import os
import time
import json
import httpx
import subprocess
from datetime import datetime
from pathlib import Path

import psutil  # requirements.txt 에 추가 필요

# -----------------------
# 헬스체크 대상 URL
# -----------------------
BACKEND_HEALTH = os.getenv("BACKEND_HEALTH", "http://waai-backend:8000/health")
MCP_HEALTH = os.getenv("MCP_HEALTH", "http://mcp-diary:7002/health")
MCP_FS_HEALTH = os.getenv("MCP_FS_HEALTH", "http://mcp-filesystem:7001/health")
DIARY_BOT_HEALTH = os.getenv("DIARY_BOT_HEALTH", "http://diary-format-bot:8001/health")

HEALTH_TARGETS = [
    ("waai-backend", BACKEND_HEALTH),
    ("mcp-diary", MCP_HEALTH),
    ("mcp-filesystem", MCP_FS_HEALTH),
    ("diary-format-bot", DIARY_BOT_HEALTH),
]

# -----------------------
# 환경변수 & 기본 설정
# -----------------------
INTERVAL = int(os.getenv("INTERVAL", "60"))  # 초

# CPU / GPU 임계치 (%)
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "80"))
GPU_THRESHOLD = float(os.getenv("GPU_THRESHOLD", "80"))

# md 파일 위치 (diary-format-bot 이 만들어주는 경로와 동일하게 마운트)
DIARY_DIR = Path(os.getenv("DIARY_DIR", "/waai/data/diary"))

# 로그 파일로도 남기고 싶으면 경로 설정 (없으면 콘솔만)
LOG_FILE = os.getenv("MONITOR_LOG_FILE", "")

# 이미 감지한 md 파일 목록 (신규 생성 감지용)
known_md_files: set[str] = set()


# -----------------------
# 로깅 유틸
# -----------------------
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{level}] {ts} {msg}"
    print(line, flush=True)

    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # 로그 파일 에러는 모니터가 죽지 않도록 무시
            pass


# -----------------------
# 1. /health 결과 기반 헬스체크
# -----------------------
def check_one_health(name: str, url: str):
    try:
        r = httpx.get(url, timeout=5.0)
    except Exception as e:
        log(f"[HEALTH] {name} 접속 실패: {e}", level="ERROR")
        return

    if r.status_code != 200:
        # 비정상 상태
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        log(
            f"[HEALTH] {name} 비정상 상태: status={r.status_code}, detail={detail}",
            level="ERROR",
        )
        return

    # 200 OK 인 경우
    try:
        data = r.json()
        log(f"[HEALTH] {name} OK: {data}", level="INFO")
    except json.JSONDecodeError:
        log(f"[HEALTH] {name} OK (non-JSON): {r.text}", level="INFO")


def check_all_health():
    for name, url in HEALTH_TARGETS:
        check_one_health(name, url)


# -----------------------
# 2. CPU / GPU 사용률 체크
# -----------------------
def check_cpu_usage():
    try:
        # psutil.cpu_percent는 첫 호출에 0 나올 수 있으므로, 약간의 간격을 줄 수도 있음
        cpu = psutil.cpu_percent(interval=0.1)
        if cpu >= CPU_THRESHOLD:
            log(
                f"[CPU ALERT] CPU 사용률 {cpu:.1f}% (임계치 {CPU_THRESHOLD}%) 초과",
                level="WARN",
            )
        else:
            log(f"[CPU] 현재 CPU 사용률 {cpu:.1f}%", level="INFO")
    except Exception as e:
        log(f"CPU 사용률 조회 실패: {e}", level="WARN")


def get_gpu_usages():
    """
    nvidia-smi가 있을 때 GPU 사용률(%) 리스트 반환.
    없으면 None 반환.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # 컨테이너에 nvidia-smi 없는 경우
        return None
    except Exception as e:
        log(f"GPU 사용률 조회 실패: {e}", level="WARN")
        return None

    usages = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            usages.append(float(line))
        except ValueError:
            continue
    return usages


def check_gpu_usage():
    usages = get_gpu_usages()
    if usages is None:
        log(
            "GPU 사용률 체크: nvidia-smi 명령을 찾을 수 없음 "
            "(GPU 모니터링 비활성화 상태, 필요 시 nvidia-smi 제공 이미지/마운트 필요)",
            level="INFO",
        )
        return

    if not usages:
        log("GPU 사용률 체크: GPU 정보 없음", level="INFO")
        return

    max_usage = max(usages)
    if max_usage >= GPU_THRESHOLD:
        log(
            f"[GPU ALERT] GPU 사용률 최대 {max_usage:.1f}% (임계치 {GPU_THRESHOLD}%) 초과",
            level="WARN",
        )
    else:
        log(f"[GPU] GPU 사용률들: {usages}, 최대 {max_usage:.1f}%", level="INFO")


# -----------------------
# 3. 신규 md 파일 생성 알림
# -----------------------
def init_known_md_files():
    """
    모니터 시작 시점에 이미 존재하는 md 파일 목록을 기준으로
    '기존 것'으로 간주하고, 이후에 생기는 것만 신규로 알림.
    """
    if not DIARY_DIR.exists():
        log(
            f"DIARY_DIR {DIARY_DIR} 가 존재하지 않습니다. 볼륨 마운트를 확인하세요.",
            level="WARN",
        )
        return

    for p in DIARY_DIR.glob("*.md"):
        known_md_files.add(p.name)

    log(f"초기 md 파일 개수: {len(known_md_files)}개", level="INFO")


def check_new_md_files():
    """
    /waai/data/diary (또는 DIARY_DIR) 하위의 .md 파일 목록을 보고
    새로 생긴 파일이 있으면 'txt→md 변환 완료'로 보고 알림.
    """
    if not DIARY_DIR.exists():
        # 이미 초기화 단계에서 경고했으므로 여기서는 조용히 리턴
        return

    current_files = set(p.name for p in DIARY_DIR.glob("*.md"))
    new_files = current_files - known_md_files

    for fname in sorted(new_files):
        log(f"[DIARY] 신규 md 파일 생성 감지 (포맷팅 완료): {fname}", level="INFO")

    if new_files:
        known_md_files.update(new_files)


# -----------------------
# 메인 루프
# -----------------------
def main():
    log(
        f"모니터 시작: INTERVAL={INTERVAL}s, DIARY_DIR={DIARY_DIR}",
        level="INFO",
    )
    log(
        f"CPU_THRESHOLD={CPU_THRESHOLD}%, GPU_THRESHOLD={GPU_THRESHOLD}%",
        level="INFO",
    )
    log(
        f"HEALTH_TARGETS={HEALTH_TARGETS}",
        level="INFO",
    )

    init_known_md_files()

    while True:
        try:
            log("=" * 30 + " 주기 모니터링 시작 " + "=" * 30, level="INFO")

            # 1) 헬스체크 (backend / mcp-diary / diary-format-bot 각각)
            check_all_health()

            # 2) CPU / GPU 사용률 체크
            check_cpu_usage()
            check_gpu_usage()

            # 3) 신규 md 파일 생성 감지 (txt→md 포맷팅 완료 알림)
            check_new_md_files()

        except Exception as e:
            # 모니터 자체에서 잡히지 않은 예외가 터지면 여기서 최종 알림
            log(f"[MONITOR ERROR] 예기치 못한 에러 발생: {e}", level="ERROR")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
