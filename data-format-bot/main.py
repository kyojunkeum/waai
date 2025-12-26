from __future__ import annotations
import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import re
import uuid
import threading
import http.server
import socketserver
import requests
import yaml
from typing import Any, Dict, Optional, Tuple


HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8001"))    # /health port
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))  # 초 단위
MONITOR_LOG_DIR = Path(os.getenv("MONITOR_LOG_DIR", "/data/_logs"))
MONITOR_LOG_FILE = MONITOR_LOG_DIR / "data-format-bot.jsonl"


def monitor_log(category: str, message: str, payload: Dict[str, Any] | None = None) -> None:
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "module": "data-format-bot",
        "category": category,
        "message": message,
        "payload": payload or {},
    }
    try:
        MONITOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with MONITOR_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

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

# =========================
# Schema score (stats)
# =========================

STATS_DIR = Path(os.getenv("STATS_DIR", "/home/witness/waai/data/_stats"))
STATS_HISTORY_PATH = STATS_DIR / "stats_history.jsonl"
STATS_MAX_LINES = int(os.getenv("STATS_MAX_LINES", "2000"))

DEFAULT_SCAN_ROOTS = [
    "/home/witness/waai/data/diary",
    "/home/witness/waai/data/ideas",
    "/home/witness/waai/data/works",
    "/home/witness/waai/data/bible",
    "/home/witness/waai/data/web_research",
]


def _parse_scan_roots() -> list[Path]:
    raw_roots = os.getenv("SCAN_ROOTS")
    if raw_roots:
        roots = [r.strip() for r in raw_roots.split(",") if r.strip()]
        return [Path(r) for r in roots]
    legacy_root = os.getenv("SCAN_ROOT")
    if legacy_root:
        return [Path(legacy_root)]
    return [Path(r) for r in DEFAULT_SCAN_ROOTS]


SCAN_ROOTS = _parse_scan_roots()
TAGS_MIN =int(os.getenv("TAGS_MIN","1"))
TAGS_MAX =int(os.getenv("TAGS_MAX","10"))
SUMMARY_MIN_LEN =int(os.getenv("SUMMARY_MIN_LEN","10"))

REQUIRED_DIARY_KEYS = os.getenv(
"REQUIRED_DIARY_KEYS",
"type,mood_score,title",
).split(",")

REQUIRED_DATA_KEYS = os.getenv(
"REQUIRED_DATA_KEYS",
"type,title,tags",
).split(",")


def _extract_front_matter(md_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not md_text.startswith("---"):
        return None, md_text
    parts = md_text.split("\n---\n", 1)
    if len(parts) != 2:
        return None, md_text
    try:
        meta = yaml.safe_load(parts[0][3:].strip())
        if not isinstance(meta, dict):
            return None, md_text
        return meta, parts[1]
    except Exception:
        return None, md_text


def _doc_kind(meta: Dict[str, Any], path: Path) -> str:
    t = str(meta.get("type", "")).strip()
    if t:
        return t
    if "diary" in path.parts:
        return "diary"
    return "data"


def _validate(meta: Dict[str, Any], kind: str) -> Dict[str, bool]:
    required = REQUIRED_DIARY_KEYS if kind == "diary" else REQUIRED_DATA_KEYS

    missing_required = any(
        meta.get(k.strip()) in [None, "", []] for k in required
    )

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags_violate = not (isinstance(tags, list) and TAGS_MIN <= len(tags) <= TAGS_MAX)

    summary = str(meta.get("summary", "")).strip()
    summary_short = len(summary) < SUMMARY_MIN_LEN

    return {
        "missing_required": missing_required,
        "tags_violate": tags_violate,
        "summary_short": summary_short,
    }


def scan_and_score(roots: list[Path]) -> Dict[str, Any]:
    md_files = []
    for root in roots:
        md_files.extend(root.rglob("*.md"))

    total = 0
    no_front_matter = 0
    missing_required = 0
    tags_violate = 0
    summary_short = 0

    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        total += 1
        meta, _ = _extract_front_matter(text)
        if not meta:
            no_front_matter += 1
            continue

        kind = _doc_kind(meta, p)
        r = _validate(meta, kind)

        if r["missing_required"]:
            missing_required += 1
        if r["tags_violate"]:
            tags_violate += 1
        if r["summary_short"]:
            summary_short += 1

    def rate(x: int) -> float:
        return 0.0 if total == 0 else round((x / total) * 100, 2)

    return {
        "scan_roots": [str(r) for r in roots],
        "total_md": total,
        "no_front_matter_pct": rate(no_front_matter),
        "missing_required_pct": rate(missing_required),
        "tags_violate_pct": rate(tags_violate),
        "summary_short_pct": rate(summary_short),
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


DIARY_INPUT_DIR = Path(os.getenv("DIARY_INPUT_DIR", "/home/witness/memory/diary"))
DIARY_OUTPUT_DIR = Path("/home/witness/waai/data/diary")
DIARY_PROCESSED_DIR = DIARY_INPUT_DIR / "_processed"
DIARY_PROCESSED_RECORD = DIARY_INPUT_DIR / ".processed_files.json"

# 신규 멀티 타입 입력/출력 기본 경로
MULTI_BASE_INPUT = Path(os.getenv("MULTI_BASE_INPUT", "/home/witness/memory"))
MULTI_BASE_OUTPUT = Path(os.getenv("MULTI_BASE_OUTPUT", "/home/witness/waai/data"))

TYPE_CONFIGS = {
    "works": {
        "input": MULTI_BASE_INPUT / "works",
        "output": MULTI_BASE_OUTPUT / "works",
        "doc_type": "work",
    },
    "bible": {
        "input": MULTI_BASE_INPUT / "bible",
        "output": MULTI_BASE_OUTPUT / "bible",
        "doc_type": "bible",
    },
    "ideas": {
        "input": MULTI_BASE_INPUT / "ideas",
        "output": MULTI_BASE_OUTPUT / "ideas",
        "doc_type": "idea",
    },
    "web_research": {
        # 입력은 underscore 없이 webresearch 디렉토리, 출력은 web_research로 정규화
        "input": MULTI_BASE_INPUT / "webresearch",
        "output": MULTI_BASE_OUTPUT / "web_research",
        "doc_type": "web_research",
    },
}

LLM_FIX_URL = os.getenv("LLM_FIX_URL", "http://waai-backend:8000/api/diary/reformat-md")
DATA_REFORMAT_URL = os.getenv("DATA_REFORMAT_URL", "http://waai-backend:8000/api/data/reformat-md")

# ✅ 새 옵션: 기존 처리 내역 무시 후 전체 재처리
IGNORE_PROCESSED = os.getenv("IGNORE_PROCESSED", "false").lower() == "true"


# -----------------------------
# 파일 기록 유틸
# -----------------------------

def ensure_dirs():
    DIARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIARY_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in TYPE_CONFIGS.values():
        cfg["output"].mkdir(parents=True, exist_ok=True)
        (cfg["input"] / "_processed").mkdir(parents=True, exist_ok=True)

def append_stats_history(stats: dict):
    """
    stats dict를 JSONL로 누적 저장한다.
    최근 STATS_MAX_LINES 줄만 유지.
    """
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(stats, ensure_ascii=False)

    # append
    with open(STATS_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # trim
    try:
        lines = STATS_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
        if len(lines) > STATS_MAX_LINES:
            STATS_HISTORY_PATH.write_text("\n".join(lines[-STATS_MAX_LINES:]) + "\n", encoding="utf-8")
    except Exception:
        pass

def load_processed_records(record_path: Path):
    """
    record_path (.processed_files.json)에서 이미 처리한 파일 이름 목록을 읽어온다.
    IGNORE_PROCESSED=true 이면 항상 빈 집합 반환.
    """
    if IGNORE_PROCESSED:
        print("[INFO] IGNORE_PROCESSED=true -> 기존 처리 기록 무시")
        return set()

    if record_path.exists():
        try:
            with open(record_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data)
        except Exception as e:
            print(f"[WARN] failed to load processed records from {record_path}: {e}")
            return set()
    return set()


def save_processed_records(record_path: Path, processed_set):
    """
    처리된 파일 이름들을 record_path 에 저장.
    IGNORE_PROCESSED=true 인 경우에도 기록은 갱신해 둔다.
    """
    try:
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(list(processed_set), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to save processed records to {record_path}: {e}")


# -----------------------------
# 파일명 파싱 / LLM 호출
# -----------------------------

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


def _format_date_for_filename(date: str) -> str:
    digits = date.replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        return digits
    return datetime.now().strftime("%Y%m%d")


def _format_time_for_filename(time_str: str) -> str:
    digits = time_str.replace(":", "")
    if len(digits) == 4 and digits.isdigit():
        return digits
    return datetime.now().strftime("%H%M")


def make_output_filename(date: str, time_str: str, original_name: str) -> str:
    """
    YYYYMMDD-HHMM-원본파일명_UUID.md 형식으로 저장.
    """
    date_part = _format_date_for_filename(date)
    time_part = _format_time_for_filename(time_str)
    original_part = slugify(original_name)
    short_id = uuid.uuid4().hex[:8]
    return f"{date_part}-{time_part}-{original_part}_{short_id}.md"


def make_generic_output_filename(prefix: str, title: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slugify(title or prefix)}.md"

def ensure_unique_path(directory: Path, filename: str) -> Path:
    """
    같은 파일명이 이미 있으면 짧은 UUID를 붙여 충돌을 피한다.
    """
    base_path = directory / filename
    if not base_path.exists():
        return base_path

    stem, suffix = base_path.stem, base_path.suffix
    short_id = uuid.uuid4().hex[:8]
    return directory / f"{stem}_{short_id}{suffix}"

# -----------------------------
# 처리 로직
# -----------------------------

def build_initial_yaml(doc_type: str, title: str) -> str:
    now = datetime.now()
    now_iso = now.isoformat(timespec="seconds")

    yaml_obj = {
        "type": doc_type,
        "title": title,

        # ✅ 시스템 메타 (유지)
        "created_at": now_iso,
        "updated_at": now_iso,

        # ✅ diary 의미 메타 (파생 생성)
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),

        # 공통 필드
        "tags": [],
        "topics": [],
        "people": [],
        "locations": [],
        "source": None,
        "usage": [],
        "summary": "",
    }

    return yaml.safe_dump(yaml_obj, allow_unicode=True, sort_keys=False)


def build_initial_md(doc_type: str, title: str, raw_text: str) -> str:
    header = build_initial_yaml(doc_type, title)
    return (
        f"---\n{header}---\n\n"
        f"# 원문 초안\n"
        f"```text\n{raw_text}\n```\n"
    )


def sanitize_markdown(md_text: str, header_yaml: str) -> str:
    """
    - '---' 이전에 붙는 설명/주석 제거
    - YAML 프론트매터가 없으면 header_yaml 추가
    - YAML 프론트매터가 있으면 header_yaml 기준으로 merge하여
      date/time 같은 핵심 메타가 유지/주입되도록 보장
    """
    text = (md_text or "").strip()
    if not text:
        return f"---\n{header_yaml}---\n"

    # LLM이 실수로 빈 프론트매터(--- 후 바로 ---)를 붙이는 경우 제거
    text = re.sub(r"^\s*---\s*\r?\n\s*---\s*", "---\n", text)

    # 첫 '---' 이전 제거
    idx = text.find("---")
    if idx == -1:
        return f"---\n{header_yaml}---\n\n{text}"
    text = text[idx:]

    # 프론트매터 블록 추출
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", text, flags=re.DOTALL)
    if not m:
        # 형식이 애매하면 그냥 header_yaml로 감싸서 안전하게
        return f"---\n{header_yaml}---\n\n{text}"

    fm_text = m.group(1)
    body = m.group(2) or ""

    # YAML 파싱
    try:
        base = yaml.safe_load(header_yaml) or {}
    except Exception:
        base = {}

    try:
        current = yaml.safe_load(fm_text) or {}
    except Exception:
        current = {}

    if not isinstance(base, dict):
        base = {}
    if not isinstance(current, dict):
        current = {}

    # 1) 기본은 base(스키마/기본값) 위에 current(LLM결과)를 얹는다 => LLM 값 우선
    merged = {**base, **current}

    # 2) 하지만 diary 핵심 메타는 header_yaml 값으로 "강제" (원하면 목록 수정)
    FORCE_KEYS = {"type", "title", "date", "time"}
    for k in FORCE_KEYS:
        if k in base:
            merged[k] = base[k]

    # YAML 재직렬화
    new_fm = yaml.safe_dump(merged, allow_unicode=True, sort_keys=False).rstrip()

    return f"---\n{new_fm}\n---\n\n{body.lstrip()}"


def call_llm_for_data_reformat(doc_type: str, md_text: str) -> str:
    payload = {"doc_type": doc_type, "markdown": md_text}
    monitor_log(
        "api_call",
        "data reformat start",
        {"doc_type": doc_type, "url": DATA_REFORMAT_URL, "chars": len(md_text)},
    )
    try:
        resp = requests.post(DATA_REFORMAT_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(data.get("error") or "reformat failed")
        result = data["data"]["result"]
        monitor_log(
            "api_call",
            "data reformat ok",
            {"doc_type": doc_type, "status_code": resp.status_code},
        )
        return result
    except Exception as e:
        monitor_log(
            "api_call",
            "data reformat failed",
            {"doc_type": doc_type, "url": DATA_REFORMAT_URL, "error": str(e)},
        )
        raise

def build_draft_md(raw_text: str, title: str) -> str:
    raw_text = raw_text.rstrip("\n")

    # ✅ 2번 핵심: draft도 build_initial_yaml 스키마 기반으로 헤더를 만든다
    base_header_yaml = build_initial_yaml("diary", title)

    # yaml 문자열 → dict 로드 → date/time 주입 → 다시 yaml로 dump
    header_obj = yaml.safe_load(base_header_yaml) or {}

    header_yaml = yaml.safe_dump(header_obj, allow_unicode=True, sort_keys=False).rstrip()

    body = "\n".join(
        [
            "# 오늘 요약",
            "",
            "# 오늘의 사건",
            "",
            "# 감정 / 생각",
            "",
            "# 배운 것 / 통찰",
            "",
            "# 소설 아이디어 메모",
            "",
            "# TODO / 다음에 이어서 쓸 것",
        ]
    ).rstrip()

    return f"---\n{header_yaml}\n---\n\n{body}\n"

def _pick_code_fence(text: str, base: str = "```") -> str:
    # 원문에 ```이 있으면 `````, `````` …처럼 더 긴 fence를 선택
    fence = base
    while fence in (text or ""):
        fence += "`"
    return fence

def append_raw_text(final_md: str, raw_text: str) -> str:
    raw_text = raw_text.rstrip("\n")
    fence = _pick_code_fence(raw_text, base="```")

    return (
        final_md.rstrip()
        + "\n\n---\n\n"
        + "## 원본 텍스트 (자동 보존 · 수정 금지)\n"
        + f"{fence}text\n"
        + raw_text
        + f"\n{fence}\n"
    )


def call_llm_for_reformat(md_text: str) -> str:
    """
    draft md 텍스트를 백엔드 API로 보내 최종 메타데이터+본문을 생성/보정한다.
    (raw_text는 md_text에 포함되지 않도록 설계되어 있음)
    """
    payload = {"markdown": md_text}
    monitor_log(
        "api_call",
        "diary reformat start",
        {"url": LLM_FIX_URL, "chars": len(md_text)},
    )
    try:
        resp = requests.post(LLM_FIX_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        monitor_log(
            "api_call",
            "diary reformat ok",
            {"status_code": resp.status_code},
        )
        return data["result"]
    except Exception as e:
        monitor_log(
            "api_call",
            "diary reformat failed",
            {"url": LLM_FIX_URL, "error": str(e)},
        )
        raise

    
  
def handle_new_file(file_path: Path) -> bool:
    # diary 전용(.txt) — handle_new_generic_file 흐름(단일 try/except + 선 header_yaml + sanitize)을 따라가되,
    # diary는 "output 1개만 저장" 요구가 있어서 LLM 실패 시 draft로 폴백(=기존 정책) 유지.
    if file_path.suffix.lower() != ".txt":
        return False
    doc_type = "diary"
    print(f"[INFO] Ne0w file detected: {file_path}")
    monitor_log("file_list", "new diary file", {"path": str(file_path)})

    try:
        # 1) 원문 로드
        raw_text = file_path.read_text(encoding="utf-8")
        title = file_path.stem.replace("-", " ").strip() or doc_type
        header_yaml = build_initial_yaml(doc_type, title)
 
        # 2) LLM 정제 → sanitize (generic과 동일한 순서)
        draft_md = build_draft_md(raw_text, title)
        try:
            refined_md = call_llm_for_reformat(draft_md)
            refined_md = sanitize_markdown(refined_md, header_yaml)
            print("[INFO] Metadata refined by LLM")
        except Exception as e:
            # diary는 "output은 1개만" 정책 때문에 LLM 실패 시에도 draft를 최종으로 저장
            # (generic처럼 그냥 False로 끝내면 txt가 남고 다음 워치에서 재처리될 수 있음)
            print(f"[WARN] LLM 호출 실패(diary) → draft로 폴백: {e}")
            refined_md = draft_md

        # 5) 최종 저장 직전에 raw_text를 시스템이 맨 아래에 붙임
        final_md = append_raw_text(refined_md, raw_text)

        # 6) 저장 (output에는 최종 파일 1개만)
        out_name = make_generic_output_filename(doc_type, title)
        out_path = ensure_unique_path(DIARY_OUTPUT_DIR, out_name)
        out_path.write_text(final_md, encoding="utf-8")
        print(f"[INFO] Saved formatted diary: {out_path}")
        monitor_log("file_write", "saved diary", {"path": str(out_path)})

        # 7) 원본 txt 이동
        processed_path = DIARY_PROCESSED_DIR / file_path.name
        shutil.move(str(file_path), str(processed_path))
        print(f"[INFO] Moved original to: {processed_path}")
        monitor_log("file_write", "moved diary original", {"path": str(processed_path)})

        update_state_on_success(file_path.name)
        return True

    except Exception as e:
        update_state_on_error(e)
        print(f"[ERROR] handle_new_file failed: {e}")
        monitor_log("error", "handle_new_file failed", {"error": str(e)})
        return False


def handle_new_generic_file(file_path: Path, cfg: dict) -> bool:
    if file_path.suffix.lower() != ".txt":
        return False
    doc_type = cfg["doc_type"]
    print(f"[INFO] New file detected ({doc_type}): {file_path}")
    monitor_log("file_list", "new generic file", {"doc_type": doc_type, "path": str(file_path)})

    raw_text = file_path.read_text(encoding="utf-8")
    title = file_path.stem.replace("-", " ").strip() or doc_type
    header_yaml = build_initial_yaml(doc_type, title)
    try:
        draft_md = build_initial_md(doc_type, title, raw_text)
        formatted_md = call_llm_for_data_reformat(doc_type, draft_md)
        formatted_md = sanitize_markdown(formatted_md, header_yaml)
    except Exception as e:
        update_state_on_error(e)
        print(f"[ERROR] LLM 호출 실패 ({doc_type}): {e}")
        monitor_log("error", "llm reformat failed", {"doc_type": doc_type, "error": str(e)})
        return False

    out_name = make_generic_output_filename(doc_type, title)
    out_path = cfg["output"] / out_name
    out_path.write_text(formatted_md, encoding="utf-8")
    print(f"[INFO] Saved formatted {doc_type}: {out_path}")
    monitor_log("file_write", "saved formatted", {"doc_type": doc_type, "path": str(out_path)})

    processed_dir = cfg["input"] / "_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / file_path.name
    shutil.move(str(file_path), str(processed_path))
    print(f"[INFO] Moved original to: {processed_path}")
    monitor_log("file_write", "moved original", {"doc_type": doc_type, "path": str(processed_path)})

    update_state_on_success(file_path.name)
    return True


def scan_diary(processed_files: set[str]) -> set[str]:
    candidates = list(DIARY_INPUT_DIR.glob("*.txt"))
    pending = [
        p.name for p in candidates if IGNORE_PROCESSED or p.name not in processed_files
    ]
    if pending:
        monitor_log(
            "file_list",
            "scan diary",
            {"input_dir": str(DIARY_INPUT_DIR), "pending": len(pending)},
        )

    for path in candidates:
        filename = path.name
        if (not IGNORE_PROCESSED) and (filename in processed_files):
            continue

        success = handle_new_file(path)
        if success:
            processed_files.add(filename)
            save_processed_records(DIARY_PROCESSED_RECORD, processed_files)
    return processed_files


def scan_generic(processed_map: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    works/bible/ideas/web_research 입력 디렉토리를 스캔.
    """
    for name, cfg in TYPE_CONFIGS.items():
        input_dir = cfg["input"]
        record_path = input_dir / ".processed_files.json"
        processed = processed_map.get(name) or set()

        if not input_dir.exists():
            continue

        candidates = list(input_dir.glob("*.txt"))
        pending = [
            p.name for p in candidates if IGNORE_PROCESSED or p.name not in processed
        ]
        if pending:
            monitor_log(
                "file_list",
                "scan generic",
                {"doc_type": cfg["doc_type"], "input_dir": str(input_dir), "pending": len(pending)},
            )

        for path in candidates:
            filename = path.name
            if (not IGNORE_PROCESSED) and (filename in processed):
                continue

            success = handle_new_generic_file(path, cfg)
            if success:
                processed.add(filename)
                save_processed_records(record_path, processed)

        processed_map[name] = processed

    return processed_map


# -----------------------------
# /health HTTP 서버 (main 위에)
# -----------------------------
class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/stats":
            try:
                stats = scan_and_score(SCAN_ROOTS)
                append_stats_history(stats)
                body_bytes = json.dumps(stats, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
            except Exception as e:
                err_body = {"error": "stats_failed", "detail": str(e)}
                body_bytes = json.dumps(err_body, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
            return

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

class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
def start_health_server():
    with ReuseTCPServer(("", HEALTH_PORT), HealthHandler) as httpd:
        print(f"[INFO] data-format-bot health server started on port {HEALTH_PORT}")
        monitor_log("server", "health server started", {"port": HEALTH_PORT})
        httpd.serve_forever()

# -----------------------------
# 메인
# -----------------------------

def main():
    ensure_dirs()
    processed_files = load_processed_records(DIARY_PROCESSED_RECORD)
    processed_generic: dict[str, set[str]] = {}
    for name, cfg in TYPE_CONFIGS.items():
        record_path = cfg["input"] / ".processed_files.json"
        processed_generic[name] = load_processed_records(record_path)

    STATE["running"] = True

    print("[INFO] data-format-bot started")
    monitor_log(
        "server",
        "data-format-bot started",
        {"scan_interval": SCAN_INTERVAL, "scan_roots": [str(r) for r in SCAN_ROOTS]},
    )
    print(f"[INFO] DIARY_INPUT_DIR  = {DIARY_INPUT_DIR}")
    print(f"[INFO] DIARY_OUTPUT_DIR = {DIARY_OUTPUT_DIR}")
    for name, cfg in TYPE_CONFIGS.items():
        print(f"[INFO] {name} INPUT={cfg['input']} OUTPUT={cfg['output']}")
    print(f"[INFO] already processed files: {len(processed_files)}")
    if IGNORE_PROCESSED:
        print("[INFO] IGNORE_PROCESSED=true -> 모든 txt 재처리")

    # 헬스 서버 쓰레드 시작
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    # 시작 시 한 번 스캔
    processed_files = scan_diary(processed_files)
    processed_generic = scan_generic(processed_generic)

    # 이후 주기적 스캔
    while True:
        time.sleep(SCAN_INTERVAL)
        processed_files = scan_diary(processed_files)
        processed_generic = scan_generic(processed_generic)



if __name__ == "__main__":
    main()
