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
MCP_HEALTH = os.getenv("MCP_HEALTH", "http://mcp-bridge:7002/health")
MCP_FS_HEALTH = os.getenv("MCP_FS_HEALTH", "http://mcp-filesystem:7001/health")
DIARY_BOT_HEALTH = os.getenv("DIARY_BOT_HEALTH", "http://data-format-bot:8001/health")

HEALTH_TARGETS = [
    ("waai-backend", BACKEND_HEALTH),
    ("mcp-bridge", MCP_HEALTH),
    ("mcp-filesystem", MCP_FS_HEALTH),
    ("data-format-bot", DIARY_BOT_HEALTH),
]

# -----------------------
# 환경변수 & 기본 설정
# -----------------------
INTERVAL = int(os.getenv("INTERVAL", "60"))  # 초

# CPU / GPU 임계치 (%)
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "80"))
GPU_THRESHOLD = float(os.getenv("GPU_THRESHOLD", "80"))

# md 파일 위치 (data-format-bot 이 만들어주는 경로와 동일하게 마운트)
DIARY_DIR = Path(os.getenv("DIARY_DIR", "/waai/data/diary"))

# 합평 원고 인입 디렉터리 (새 파일 생성 시 /api/critique 호출)
CRITIQUE_INBOX_DIR = Path(os.getenv("CRITIQUE_INBOX_DIR", "/home/witness/waai/data/objects"))
CRITIQUE_PROCESSED_DIR = Path(os.getenv("CRITIQUE_PROCESSED_DIR", "/waai/data/critique/processed"))
CRITIQUE_API_URL = os.getenv("CRITIQUE_API_URL", "http://waai-backend:8000/api/critique")
CRITIQUE_EXTENSIONS = {".txt", ".md"}
CRITIQUE_CHUNK_AUTO_CHARS = int(os.getenv("CRITIQUE_CHUNK_AUTO_CHARS", "12000"))

# 로그 파일로도 남기고 싶으면 경로 설정 (없으면 콘솔만)
LOG_FILE = os.getenv("MONITOR_LOG_FILE", "")

# 이미 감지한 md 파일 목록 (신규 생성 감지용)
known_md_files: set[str] = set()
known_critique_files: set[str] = set()
critique_failures: dict[str, int] = {}


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
# 4. 신규 합평 원고 파일 감지 -> /api/critique 호출
# -----------------------
def init_known_critique_files():
    if not CRITIQUE_INBOX_DIR.exists():
        log(
            f"CRITIQUE_INBOX_DIR {CRITIQUE_INBOX_DIR} 가 존재하지 않습니다. 볼륨 마운트를 확인하세요.",
            level="WARN",
        )
        return

    for p in CRITIQUE_INBOX_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in CRITIQUE_EXTENSIONS:
            known_critique_files.add(p.name)

    log(f"초기 합평 원고 파일 개수: {len(known_critique_files)}개", level="INFO")


def _read_text_when_stable(path: Path, retries: int = 4, delay: float = 0.5) -> str:
    last_size = -1
    for _ in range(retries):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(delay)
            continue
        if size == last_size:
            return path.read_text(encoding="utf-8", errors="ignore")
        last_size = size
        time.sleep(delay)
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            header = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1:]).lstrip()
            meta: dict[str, str] = {}
            for line in header.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    meta[key] = value
            return meta, body
    return {}, text


def _build_critique_payload(path: Path, raw_text: str) -> dict[str, object] | None:
    fallback_title = path.stem
    meta, body = _parse_frontmatter(raw_text)
    if (meta.get("type") or "").strip().lower() == "objects":
        return None
    title = meta.get("title") or fallback_title
    content = (body or "").strip()
    if not content:
        return None
    payload: dict[str, object] = {"title": title, "content": content}
    if len(content) >= CRITIQUE_CHUNK_AUTO_CHARS:
        payload["options"] = {"chunked_critique": True}
    return payload


def _post_critique_request(payload: dict[str, str], source_name: str) -> bool:
    try:
        r = httpx.post(CRITIQUE_API_URL, json=payload, timeout=120.0)
    except Exception as e:
        log(f"[CRITIQUE] API 호출 실패 ({source_name}): {e}", level="ERROR")
        return False

    if r.status_code != 200:
        log(
            f"[CRITIQUE] API 오류 ({source_name}): status={r.status_code}, detail={r.text}",
            level="ERROR",
        )
        return False

    try:
        data = r.json()
    except Exception:
        log(f"[CRITIQUE] API 응답 파싱 실패 ({source_name})", level="WARN")
        return True

    critique_path = data.get("data", {}).get("critique_file_path")
    log(f"[CRITIQUE] 합평 완료 ({source_name}) -> {critique_path}", level="INFO")
    return True


def _move_to_processed(path: Path) -> None:
    try:
        CRITIQUE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = CRITIQUE_PROCESSED_DIR / path.name
        if dest.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = CRITIQUE_PROCESSED_DIR / f"{path.stem}_{stamp}{path.suffix}"
        path.replace(dest)
        log(f"[CRITIQUE] 원고 이동 완료 -> {dest}", level="INFO")
    except Exception as e:
        log(f"[CRITIQUE] 원고 이동 실패 ({path.name}): {e}", level="WARN")


def check_new_critique_files():
    if not CRITIQUE_INBOX_DIR.exists():
        return

    current_files = set()
    for p in CRITIQUE_INBOX_DIR.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in CRITIQUE_EXTENSIONS:
            current_files.add(p.name)

    new_files = current_files - known_critique_files
    for fname in sorted(new_files):
        path = CRITIQUE_INBOX_DIR / fname
        raw_text = _read_text_when_stable(path)
        payload = _build_critique_payload(path, raw_text)
        if not payload:
            log(f"[CRITIQUE] 합평 대상 제외 또는 본문 없음: {fname}", level="WARN")
            known_critique_files.add(fname)
            continue

        ok = _post_critique_request(payload, fname)
        if ok:
            known_critique_files.add(fname)
            critique_failures.pop(fname, None)
            _move_to_processed(path)
        else:
            critique_failures[fname] = critique_failures.get(fname, 0) + 1
            log(
                f"[CRITIQUE] 재시도 대기 ({fname}) 실패 횟수={critique_failures[fname]}",
                level="WARN",
            )


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
    init_known_critique_files()

    while True:
        try:
            log("=" * 30 + " 주기 모니터링 시작 " + "=" * 30, level="INFO")

            # 1) 헬스체크 (backend / mcp-bridge / data-format-bot 각각)
            check_all_health()

            # 2) CPU / GPU 사용률 체크
            check_cpu_usage()
            check_gpu_usage()

            # 3) 신규 md 파일 생성 감지 (txt→md 포맷팅 완료 알림)
            check_new_md_files()

            # 4) 신규 합평 원고 파일 감지 -> /api/critique 호출
            check_new_critique_files()

        except Exception as e:
            # 모니터 자체에서 잡히지 않은 예외가 터지면 여기서 최종 알림
            log(f"[MONITOR ERROR] 예기치 못한 에러 발생: {e}", level="ERROR")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
