import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import threading
import http.server
import socketserver

import requests

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8001"))    # /health port
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))  # 초 단위

# -----------------------------
# 헬스 상태 전역 변수
# -----------------------------
STATE = {
    "running": False,
    "processed_files": 0,
    "last_processed_file": None,
    "last_processed_at": None,
    "last_error": None,
}


def update_state_on_success(filename: str):
    STATE["running"] = True
    STATE["processed_files"] += 1
    STATE["last_processed_file"] = filename
    STATE["last_processed_at"] = datetime.now().isoformat()
    STATE["last_error"] = None


def update_state_on_error(err: Exception):
    STATE["running"] = True
    STATE["last_error"] = str(err)


INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
PROCESSED_DIR = INPUT_DIR / "_processed"
PROCESSED_RECORD = INPUT_DIR / ".processed_files.json"

LLM_API_URL = os.getenv("LLM_API_URL", "http://waai-backend:8000/api/diary/format")
LLM_FIX_URL = os.getenv("LLM_FIX_URL", "http://waai-backend:8000/api/diary/reformat-md")

# ✅ 새 옵션: 기존 처리 내역 무시 후 전체 재처리
IGNORE_PROCESSED = os.getenv("IGNORE_PROCESSED", "false").lower() == "true"


# -----------------------------
# 파일 기록 유틸
# -----------------------------

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_records():
    """
    /input/.processed_files.json 에서 이미 처리한 파일 이름 목록을 읽어온다.
    IGNORE_PROCESSED=true 이면 항상 빈 집합 반환.
    """
    if IGNORE_PROCESSED:
        print("[INFO] IGNORE_PROCESSED=true -> 기존 처리 기록 무시")
        return set()

    if PROCESSED_RECORD.exists():
        try:
            with open(PROCESSED_RECORD, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data)
        except Exception as e:
            print(f"[WARN] failed to load processed records: {e}")
            return set()
    return set()


def save_processed_records(processed_set):
    """
    처리된 파일 이름들을 /input/.processed_files.json 에 저장.
    IGNORE_PROCESSED=true 인 경우에도 기록은 갱신해 둔다.
    """
    try:
        with open(PROCESSED_RECORD, "w", encoding="utf-8") as f:
            json.dump(list(processed_set), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to save processed records: {e}")


# -----------------------------
# 파일명 파싱 / LLM 호출
# -----------------------------

def parse_metadata_from_filename(file_path: Path):
    """
    예: 2025-12-09_23-15_tags1-tags2.txt
    """
    name = file_path.stem
    parts = name.split("_", 3)
    date = parts[0] if len(parts) > 0 else ""
    time_str = parts[1] if len(parts) > 1 else "00-00"
    title_raw = parts[2] if len(parts) > 2 else name

    time_formatted = time_str.replace("-", ":")  # 23-15 -> 23:15
    title = title_raw.replace("-", " ")

    return date, time_formatted, title


def call_llm_for_format(date: str, time_str: str, title: str, raw_text: str) -> str:
    """
    waai-backend 의 /api/diary/format 호출.
    """
    payload = {
        "date": date,
        "time": time_str,
        "title": title,
        "raw_text": raw_text,
    }
    resp = requests.post(LLM_API_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["result"]  # DiaryFormatResponse.result


def call_llm_for_metadata_fix(md_text: str) -> str:
    """
    이미 생성된 md 텍스트를 /api/diary/fix-metadata 로 보내
    메타데이터(tags, projects 등)를 보정하는 함수
    """
    payload = {"markdown": md_text}
    resp = requests.post(LLM_FIX_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["result"]


def extract_tags_from_md(md_text: str):
    """
    md_text 안에서 'tags: [ ... ]' 라인을 찾아 리스트로 파싱.
    예: tags: [아내, 빚, 회복, 신앙, 가족]
    """
    for line in md_text.splitlines():
        striped = line.strip()
        if striped.startswith("tags:"):
            start = striped.find("[")
            end = striped.find("]", start)
            if start != -1 and end != -1:
                inner = striped[start + 1 : end]
                parts = [
                    p.strip().strip("'\"")
                    for p in inner.split(",")
                    if p.strip()
                ]
                return parts
    return []


def slugify(text: str) -> str:
    """
    파일명용 slug: 공백 제거, 위험한 문자 치환.
    (한글은 그대로 두고, / 등만 - 로 치환)
    """
    text = text.strip()
    text = text.replace(" ", "")
    for ch in "/\\?%*:|\"<>":
        text = text.replace(ch, "-")
    return text or "note"


def make_output_filename(date: str, md_text: str) -> str:
    """
    tags 에서 subject1-subject2 를 만드는 로직.
    - tags가 2개 이상: 첫 두 개
    - 1개: 하나 + 'diary'
    - 0개: 'diary-note'
    """
    tags = extract_tags_from_md(md_text)

    if len(tags) >= 2:
        subj1, subj2 = tags[0], tags[1]
    elif len(tags) == 1:
        subj1, subj2 = tags[0], "diary"
    else:
        subj1, subj2 = "diary", "note"

    s1 = slugify(subj1)
    s2 = slugify(subj2)

    return f"{date}_{s1}-{s2}.md"


# -----------------------------
# 처리 로직
# -----------------------------

def handle_new_file(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".txt":
        return False

    print(f"[INFO] New file detected: {file_path}")
    date, time_str, title = parse_metadata_from_filename(file_path)
    raw_text = file_path.read_text(encoding="utf-8")

    try:
        # 1차: 기본 포맷 (지금 쓰는 /api/diary/format)
        md_text = call_llm_for_format(date, time_str, title, raw_text)

        # 2차: 메타데이터 보정 (옵션)
        try:
            md_text = call_llm_for_metadata_fix(md_text)
            print("[INFO] Metadata refined by LLM")
        except Exception as e:
            print(f"[WARN] 메타데이터 보정 실패, 1차 결과 그대로 사용: {e}")

    except Exception as e:
        update_state_on_error(e)
        print(f"[ERROR] LLM 호출 실패: {e}")
        return False

    out_name = make_output_filename(date, md_text)
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(md_text, encoding="utf-8")
    print(f"[INFO] Saved formatted diary: {out_path}")

    processed_path = PROCESSED_DIR / file_path.name
    shutil.move(str(file_path), str(processed_path))
    print(f"[INFO] Moved original to: {processed_path}")

    update_state_on_success(file_path.name)
    return True


def scan_once(processed_files: set[str]) -> set[str]:
    for path in INPUT_DIR.glob("*.txt"):
        filename = path.name
        if (not IGNORE_PROCESSED) and (filename in processed_files):
            continue

        success = handle_new_file(path)
        if success:
            processed_files.add(filename)
            save_processed_records(processed_files)
    return processed_files


# -----------------------------
# /health HTTP 서버 (main 위에)
# -----------------------------
class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # 상태 구성
        status = "ok"
        if STATE["last_error"] is not None:
            status = "error"

        body = {
            "status": status,
            "running": STATE["running"],
            "processed_files": STATE["processed_files"],
            "last_processed_file": STATE["last_processed_file"],
            "last_processed_at": STATE["last_processed_at"],
            "last_error": STATE["last_error"],
        }
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    # 로그 너무 시끄럽지 않게
    def log_message(self, format, *args):  # noqa: A003 (BaseHTTPRequestHandler hook name)
        return


def start_health_server():
    with socketserver.TCPServer(("", HEALTH_PORT), HealthHandler) as httpd:
        print(f"[INFO] diary-format-bot health server started on port {HEALTH_PORT}")
        httpd.serve_forever()


# -----------------------------
# 메인
# -----------------------------

def main():
    ensure_dirs()
    processed_files = load_processed_records()

    STATE["running"] = True

    print("[INFO] diary-format-bot started")
    print(f"[INFO] INPUT_DIR  = {INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"[INFO] already processed files: {len(processed_files)}")
    if IGNORE_PROCESSED:
        print("[INFO] IGNORE_PROCESSED=true -> 모든 txt 재처리")

    # 헬스 서버 쓰레드 시작
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    # 시작 시 한 번 스캔
    processed_files = scan_once(processed_files)

    # 이후 주기적 스캔
    while True:
        time.sleep(SCAN_INTERVAL)
        processed_files = scan_once(processed_files)


if __name__ == "__main__":
    main()
