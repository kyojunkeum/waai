from __future__ import annotations

import asyncio
import calendar
import json
import logging
import os
import re
from urllib.parse import quote_plus
from datetime import datetime, date
from pathlib import Path
from typing import Any, List
from collections import deque
from typing import Tuple, Optional, Callable
import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, ValidationError
from starlette.templating import Jinja2Templates

from mcp_client import (
    get_mood_stats,
    get_project_timeline,
    select_and_summarize,
)
from utils import ensure_dir, normalize_query, save_txt

# 데이터 경로를 환경변수 → /data → 로컬 repo/data → 홈 경로 순서로 해석
def _resolve_data_path(env_var: str, default_subpath: str, require_writable: bool = False) -> Path:
    candidates: list[Path] = []

    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))

    candidates.append(Path("/data") / default_subpath)
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "data" / default_subpath)

    if require_writable:
        candidates.append(Path.home() / ".waai" / default_subpath)

    if require_writable:
        for path in candidates:
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                continue
        return candidates[-1]

    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]

DIARY_ROOT = str(_resolve_data_path("DIARY_ROOT", "diary", require_writable=True))
DIARY_OUTPUT_DIR = Path(DIARY_ROOT)
DIARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IDEAS_ROOT = str(_resolve_data_path("IDEAS_ROOT", "ideas"))
WEB_RESEARCH_ROOT = str(_resolve_data_path("WEB_RESEARCH_ROOT", "web_research", require_writable=True))
WEBRESEARCH_OUT_DIR = os.environ.get("WEBRESEARCH_OUT_DIR", "/memory/webresearch")
WORKS_ROOT = str(_resolve_data_path("WORKS_ROOT", "works"))
BIBLE_ROOT = str(_resolve_data_path("BIBLE_ROOT", "bible"))
CRITIQUE_OBJECTS_ROOT = _resolve_data_path("CRITIQUE_OBJECTS_ROOT", "critique/objects", require_writable=True)
CRITIQUE_RESULTS_ROOT = _resolve_data_path("CRITIQUE_RESULTS_ROOT", "critique/results", require_writable=True)
CRITIQUE_CRITERIA_PATH = _resolve_data_path("CRITIQUE_CRITERIA_PATH", "critique/criteria/합평기준규칙.md")
CRITIQUE_CHUNK_MAX_CHARS = int(os.environ.get("CRITIQUE_CHUNK_MAX_CHARS", "6000"))
CRITIQUE_CHUNK_MAX_PARTS = int(os.environ.get("CRITIQUE_CHUNK_MAX_PARTS", "20"))

PLAYWRIGHT_SCHEDULE_PATH = Path(os.environ.get("PLAYWRIGHT_SCHEDULE_PATH", "/data/web_research/playwright_schedule.json"))
OUTPUT_ROOT = str(_resolve_data_path("OUTPUT_ROOT", "outputs", require_writable=True))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
Path(WEB_RESEARCH_ROOT).mkdir(parents=True, exist_ok=True)
ensure_dir(WEBRESEARCH_OUT_DIR)
CRITIQUE_OBJECTS_ROOT.mkdir(parents=True, exist_ok=True)
CRITIQUE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
PLAYWRIGHT_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("waai-backend")

# 데이터 로드 안전장치
MAX_FILES_PER_TYPE = int(os.environ.get("MAX_FILES_PER_TYPE", "10"))

# 🔹 공통 LLM 설정
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# Ollama용
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2:7b")

# OpenAI / 호환 서버용 (선택)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# docker-compose 기본 포트는 7003. 필요시 환경변수로 오버라이드.
PLAYWRIGHT_MCP_URL = os.environ.get("PLAYWRIGHT_MCP_URL", "http://mcp-playwright:7003")
SEARXNG_URL = os.environ.get("SEARXNG_URL")

# YAML Front Matter 검증시 사용
FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.S)

# 재시도 검증 시 사용
LLM_VALIDATE_RETRIES=2
PLAN_MIN_CHARS=800
CRITIQUE_MIN_CHARS=600

class DiaryFormatRequest(BaseModel):
    date: str       # "2025-12-10"
    time: str       # "23:15"
    title: str
    raw_text: str


class DiaryFormatResponse(BaseModel):
    result: str     # 완성된 Markdown (YAML 헤더 + 본문 + 원본텍스트)


class DiaryReformatRequest(BaseModel):
    markdown: str   # 기존 md 전체 텍스트


class DiaryReformatResponse(BaseModel):
    result: str     # 보정된 md 전체 텍스트


class DataReformatRequest(BaseModel):
    doc_type: str   # idea | work | web_research | bible
    markdown: str   # 기존 md 전체 텍스트


class DataReformatResponse(BaseModel):
    result: str     # 보정된 md 전체 텍스트

class PlanGenerateRequest(BaseModel):
    """
    Open WebUI에서 이 API를 호출할 때 넘길 수 있는 옵션들입니다.
    아무것도 안 넘기면 '전체 일기 기반 기획서'를 만든다고 생각하면 됩니다.
    """
    topic: str | None = None              # 기획서 제목/주제 (예: "최근 일기 기반 단편소설 기획서")
    keyword: str | None = None            # 특정 키워드 기반으로 다루고 싶을 때
    start_date: str | None = None         # "2025-01-01" 이런 식
    end_date: str | None = None           # "2025-03-31"
    mode: str = "outline"                 # mcp-bridge summarize 모드 (outline/summary 등)
    output_format: str = "md"             # "txt" 또는 "md"
    extra_instruction: str | None = None  # "가족/신앙 비중을 더 강조해줘" 같은 추가 지시


class PlanGeneratePromptOnlyRequest(BaseModel):
    prompt: str
    include: list[str] | None = None  # 옵션: 포함할 데이터 타입 지정


class PlanGenerateResponse(BaseModel):
    title: str        # 기획서 제목
    content: str      # 기획서 본문 (Open WebUI에서 바로 보여줄 내용)
    file_path: str    # /waai/data/outputs/... 저장된 경로
    sources: list[dict[str, Any]] = Field(default_factory=list)   # 근거 목록 (필수, 비어도 포함)


class PlanFromDataRequest(BaseModel):
    goal: str = "단편소설 기획서 제작"
    include: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    keyword: str | None = None
    extra_instruction: str | None = None
    prompt: str | None = None  # 자연어 프롬프트로 날짜/키워드/goal 추출용 (선택)


class PlanGenerateEnvelope(BaseModel):
    success: bool = True
    message: str = "ok"
    data: PlanGenerateResponse | None = None
    error: Any | None = None


class PlaywrightCrawlRequest(BaseModel):
    prompt: str | None = None
    keywords: list[str] = Field(default_factory=list, max_items=5)
    per_keyword: int = 2
    url: str | None = None
    timeout_ms: int | None = 20000


class PlaywrightCrawlResponse(BaseModel):
    saved_files: list[str] = Field(default_factory=list)
    count: int = 0
    keywords: list[str] = Field(default_factory=list)
    articles: list[dict[str, Any]] = Field(default_factory=list)


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    engine: str = "google_news_rss"


class WebSearchFetchRequest(BaseModel):
    query: str
    max_results: int = 5
    engine: str = "google_news_rss"
    timeout_ms: int = 20000


class PlaywrightScheduleConfig(BaseModel):
    enabled: bool = False
    interval_minutes: int = 60
    keywords: list[str] = Field(default_factory=list, max_items=5)
    per_keyword: int = 2
    last_run: str | None = None
    last_error: str | None = None
    last_count: int | None = None


class CritiqueOptions(BaseModel):
    save_critique: bool = True
    save_work: bool | None = None
    chunked_critique: bool = False
    chunk_max_chars: int | None = None
    max_parts: int | None = None


class CritiqueRequest(BaseModel):
    title: str
    content: str
    options: CritiqueOptions | None = None
    extra_instruction: str | None = None


class CritiqueResponse(BaseModel):
    path: str
    critique: str

## 데이터 폼 유효성 검증 스키마
class ReformatResult(BaseModel):
    front_matter: dict
    body: str
    tags: list[str]

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Static files (CSS 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS for Open WebUI & dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://open-webui:8080",
        "http://open-webui:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_front_matter(md: str) -> Tuple[dict, str, bool]:
    """
    Returns: (meta_dict, body_text, has_front_matter)
    """
    if not md:
        return {}, "", False
    m = FRONT_MATTER_RE.match(md)
    if not m:
        return {}, md, False

    raw_yaml = m.group(1)
    body = md[m.end():]
    try:
        meta = yaml.safe_load(raw_yaml) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}
    return meta, body, True


def _iso_datetime_like(val: Any) -> bool:
    if isinstance(val, datetime):
        return True
    if isinstance(val, str):
        try:
            datetime.fromisoformat(val)
            return True
        except Exception:
            return False
    return False


def _is_list_of_str(val: Any, min_len: int = 0, max_len: int | None = None) -> bool:
    if not isinstance(val, list):
        return False
    if any(not isinstance(x, (str, int, float, bool)) for x in val):
        return False
    # 문자열화는 허용하되, 길이 체크
    if len(val) < min_len:
        return False
    if max_len is not None and len(val) > max_len:
        return False
    return True


def validate_diary_front_matter(meta: dict) -> None:
    """
    /api/diary/reformat-md 결과물(일기 md) 검증
    - front-matter 존재
    - date/time/type/mood/mood_score/tags/summary 등 핵심 필드 검증
    """
    # required keys
    required = ["date", "time", "title", "type", "mood", "mood_score", "tags", "summary"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    # date format
    try:
        datetime.strptime(str(meta["date"]), "%Y-%m-%d")
    except Exception:
        raise ValueError("date must be YYYY-MM-DD")

    # time format "HH:MM"
    if not re.match(r"^\d{2}:\d{2}$", str(meta["time"])):
        raise ValueError("time must be HH:MM string")

    # type fixed
    if str(meta["type"]).strip().lower() != "diary":
        raise ValueError("type must be 'diary'")

    # mood_score range + 1 decimal
    try:
        ms = float(meta["mood_score"])
    except Exception:
        raise ValueError("mood_score must be a float")
    if ms < -1.0 or ms > 1.0:
        raise ValueError("mood_score must be between -1.0 and 1.0")
    if round(ms, 1) != ms:
        raise ValueError("mood_score must have at most 1 decimal place")

    # tags 3~7
    if not _is_list_of_str(meta.get("tags"), min_len=3, max_len=7):
        raise ValueError("tags must be list with 3~7 items")

    # summary length (너무 짧으면 품질상 실패로 간주)
    if len(str(meta.get("summary") or "").strip()) < 10:
        raise ValueError("summary too short")


def validate_data_front_matter(meta: dict, doc_type: str) -> None:
    """
    /api/data/reformat-md 결과물(idea/work/web_research/bible) 검증
    """
    required = ["type", "title", "created_at", "updated_at", "tags", "topics", "people", "locations", "usage", "summary", "source"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    if str(meta["type"]).strip().lower() != doc_type:
        raise ValueError(f"type must be '{doc_type}'")

    if not _iso_datetime_like(meta.get("created_at")):
        raise ValueError("created_at must be ISO datetime")
    if not _iso_datetime_like(meta.get("updated_at")):
        raise ValueError("updated_at must be ISO datetime")

    if not _is_list_of_str(meta.get("tags"), min_len=1, max_len=20):
        raise ValueError("tags must be a list (min 1)")
    if not _is_list_of_str(meta.get("topics"), min_len=1, max_len=20):
        raise ValueError("topics must be a list (min 1)")
    if not _is_list_of_str(meta.get("people"), min_len=0, max_len=30):
        raise ValueError("people must be a list")
    if not _is_list_of_str(meta.get("locations"), min_len=0, max_len=30):
        raise ValueError("locations must be a list")
    if not _is_list_of_str(meta.get("usage"), min_len=0, max_len=10):
        raise ValueError("usage must be a list (min 0)")

    # source can be null/None/""/url/text
    src = meta.get("source")
    if src is not None and not isinstance(src, (str, dict, list)):
        raise ValueError("source must be string or null")


def validate_plan_front_matter(meta: dict) -> None:
    required = ["type", "title", "goal", "include", "created_at", "updated_at", "usage", "sources"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    if str(meta["type"]).strip().lower() != "plan":
        raise ValueError("type must be 'plan'")
    if not _iso_datetime_like(meta.get("created_at")):
        raise ValueError("created_at must be ISO datetime")
    if not _iso_datetime_like(meta.get("updated_at")):
        raise ValueError("updated_at must be ISO datetime")
    if not _is_list_of_str(meta.get("usage"), min_len=1):
        raise ValueError("usage must be a list (min 1)")

    inc = meta.get("include")
    if not isinstance(inc, list) or not inc:
        raise ValueError("include must be a non-empty list")


def validate_critique_front_matter(meta: dict) -> None:
    required = ["type", "object_title", "created_at", "updated_at", "source_object_file", "usage"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    if str(meta["type"]).strip().lower() != "critique":
        raise ValueError("type must be 'critique'")
    if not _iso_datetime_like(meta.get("created_at")):
        raise ValueError("created_at must be ISO datetime")
    if not _iso_datetime_like(meta.get("updated_at")):
        raise ValueError("updated_at must be ISO datetime")
    if not _is_list_of_str(meta.get("usage"), min_len=1):
        raise ValueError("usage must be a list (min 1)")


async def call_llm_with_front_matter_retry(
    prompt: str,
    validate_meta_fn: Callable[[dict], None],
    retries: int = 2,
    must_have_front_matter: bool = True,
) -> str:
    """
    LLM이 Markdown을 반환한다고 가정.
    - front-matter 존재 여부 + meta 검증에 실패하면 재시도
    - 실패 사유를 다음 프롬프트에 피드백으로 붙임
    """
    last_md = ""
    last_err = ""

    for attempt in range(retries + 1):
        md = await call_llm(prompt)
        last_md = md

        meta, body, has_fm = extract_front_matter(md)

        try:
            if must_have_front_matter and not has_fm:
                raise ValueError("front-matter is missing. Output must start with YAML front-matter.")

            validate_meta_fn(meta)

            # 통과하면 그대로 반환
            return md

        except Exception as exc:
            last_err = str(exc)
            if attempt >= retries:
                # 마지막 실패는 원문 그대로 반환하지 말고, 호출자 쪽에서 에러 처리할지 선택 가능
                raise ValueError(f"LLM output validation failed after retries: {last_err}")

            # 다음 시도용 프롬프트 강화
            prompt = (
                prompt
                + "\n\n"
                + "[검증 실패 피드백]\n"
                + f"- 실패 사유: {last_err}\n"
                + "- 반드시 YAML front-matter를 최상단에 1개만 만들 것.\n"
                + "- 키 누락/형식 불일치(날짜/시간/score/리스트)를 수정해 재출력할 것.\n"
                + "- 설명 금지, 최종 Markdown만 출력.\n"
            )

    # 여기 도달하지 않음
    return last_md

class DiarySchema(BaseModel):
    date: str                     # YYYY-MM-DD
    mood_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="감정 점수 (-1.0 ~ 1.0, 소수점 1자리)"
    )
    summary: str
    body: str
    tags: List[str]

    @validator("date")
    def validate_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @validator("mood_score")
    def validate_mood_score_precision(cls, v):
        if round(v, 1) != v:
            raise ValueError("mood_score must have at most 1 decimal place")
        return v


async def call_llm(prompt: str) -> str:
    """
    WAAI 백엔드용 공통 LLM 호출 함수
    - LLM_BACKEND=ollama  : Ollama /api/generate
    - LLM_BACKEND=openai  : OpenAI 또는 호환 서버 /v1/chat/completions
    """
    backend = LLM_BACKEND

    # 1) Ollama
    if backend == "ollama":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    # 2) OpenAI / 호환 서버
    elif backend == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 가 설정되어 있지 않습니다.")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    else:
        raise RuntimeError(f"지원하지 않는 LLM_BACKEND: {backend}")


def standard_response(success: bool = True, message: str = "ok", data: Any = None, error: Any = None):
    """
    OpenWebUI HTTP Tool 연동을 위한 공통 응답 포맷.
    - success: bool
    - message: 한 줄 설명
    - data: 주요 응답 페이로드 (dict/모델)
    - error: 에러 상세 (없으면 None)
    """
    return {
        "success": success,
        "message": message,
        "data": data,
        "error": error,
    }


def save_plan_output(content: str) -> str:
    """
    V3.1 요구사항: /data/outputs/plan_YYYYMMDD.md 형식으로 저장.
    - 동일 날짜에 여러 번 생성 시에는 중복을 피하기 위해 _HHMMSS 를 붙인다.
    """
    today = datetime.now()
    date_part = today.strftime("%Y%m%d")
    base_name = f"plan_{date_part}.md"
    path = Path(OUTPUT_ROOT) / base_name
    if path.exists():
        suffix = today.strftime("%H%M%S")
        path = Path(OUTPUT_ROOT) / f"plan_{date_part}_{suffix}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def extract_json_block(text: str) -> str:
    """
    LLM이 앞뒤로 멘트를 붙이는 경우를 대비해 JSON 블록만 추출.
    """
    match = re.search(r"\{.*\}", text, re.S)
    return match.group(0) if match else "{}"


def _format_date(year: int, month: int, day: int | None = None, month_end: bool = False) -> str:
    if day is None:
        day = calendar.monthrange(year, month)[1] if month_end else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_keyword_from_prompt(user_prompt: str) -> str | None:
    kw_match = re.search(r"[\"'“”‘’]([^\"'“”‘’]{1,20})[\"'“”‘’]", user_prompt)
    if kw_match:
        keyword = kw_match.group(1).strip()
        if keyword:
            return keyword
    return None


def _extract_keywords_from_prompt(prompt: str, limit: int = 5) -> list[str]:
    """
    간단한 키워드 추출: 따옴표/쉼표/공백 기준으로 잘라 최대 limit개.
    """
    if not prompt:
        return []
    # 따옴표로 감싼 표현 우선
    quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,30})[\"'“”‘’]", prompt)
    words: list[str] = []
    words.extend([w.strip() for w in quoted if w.strip()])
    # 쉼표/공백 스플릿
    parts = re.split(r"[,\s]+", prompt)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 길이 1 글자는 무시
        if len(p) < 2:
            continue
        words.append(p)
    # 중복 제거, 최대 limit
    deduped: list[str] = []
    for w in words:
        if w not in deduped:
            deduped.append(w)
        if len(deduped) >= limit:
            break
    return deduped


def rule_based_plan_parse(user_prompt: str) -> dict[str, str | None]:
    """
    빠르게 파싱할 수 있는 요소(날짜/키워드)는 룰 기반으로 먼저 추출.
    - 명확한 날짜 범위가 보이면 바로 사용
    - 애매한 표현은 LLM 파서가 보완
    """
    keyword = _extract_keyword_from_prompt(user_prompt)

    # 1) "2025년 10월부터 11월까지" 처럼 월 범위
    month_range = re.search(
        r"(?P<start_year>\d{4})년\s*(?P<start_month>\d{1,2})월\s*(?:부터|~|-|–)?\s*(?:(?P<end_year>\d{4})년\s*)?(?P<end_month>\d{1,2})월",
        user_prompt,
    )
    if month_range:
        start_year = int(month_range.group("start_year"))
        start_month = int(month_range.group("start_month"))
        end_year = int(month_range.group("end_year") or start_year)
        end_month = int(month_range.group("end_month"))
        return {
            "start_date": _format_date(start_year, start_month),
            "end_date": _format_date(end_year, end_month, month_end=True),
            "keyword": keyword,
        }

    # 2) 2025-10-01 ~ 2025-11-30 같은 날짜 범위
    date_range = re.search(
        r"(?P<start_year>\d{4})[./-](?P<start_month>\d{1,2})[./-](?P<start_day>\d{1,2})\s*(?:부터|~|-|–|to)\s*(?:(?P<end_year>\d{4})[./-])?(?P<end_month>\d{1,2})[./-](?P<end_day>\d{1,2})",
        user_prompt,
    )
    if date_range:
        start_year = int(date_range.group("start_year"))
        start_month = int(date_range.group("start_month"))
        start_day = int(date_range.group("start_day"))
        end_year = int(date_range.group("end_year") or start_year)
        end_month = int(date_range.group("end_month"))
        end_day = int(date_range.group("end_day"))
        return {
            "start_date": _format_date(start_year, start_month, start_day),
            "end_date": _format_date(end_year, end_month, end_day),
            "keyword": keyword,
        }

    # 2-1) 12월 10일부터 12월 18일까지 같은 '연도 없는' 날짜 범위
    day_range = re.search(
        r"(?P<start_month>\d{1,2})월\s*(?P<start_day>\d{1,2})일?\s*(?:부터|~|-|–|to)\s*(?P<end_month>\d{1,2})월\s*(?P<end_day>\d{1,2})일?",
        user_prompt,
    )
    if not day_range:
        day_range = re.search(
            r"(?P<start_month>\d{1,2})[./-](?P<start_day>\d{1,2})\s*(?:부터|~|-|–|to)\s*(?P<end_month>\d{1,2})[./-](?P<end_day>\d{1,2})",
            user_prompt,
        )
    if day_range:
        year = datetime.now().year
        start_month = int(day_range.group("start_month"))
        start_day = int(day_range.group("start_day"))
        end_month = int(day_range.group("end_month"))
        end_day = int(day_range.group("end_day"))
        end_year = year + 1 if end_month < start_month else year
        return {
            "start_date": _format_date(year, start_month, start_day),
            "end_date": _format_date(end_year, end_month, end_day),
            "keyword": keyword,
        }

    # 3) 단일 날짜나 월만 지정된 경우 → 월 전체 범위로 간주
    single_date = re.search(
        r"(?P<year>\d{4})년\s*(?P<month>\d{1,2})월(?:\s*(?P<day>\d{1,2})일)?",
        user_prompt,
    )
    if not single_date:
        single_date = re.search(
            r"(?P<year>\d{4})[./-](?P<month>\d{1,2})(?:[./-](?P<day>\d{1,2}))?",
            user_prompt,
        )

    if single_date:
        year = int(single_date.group("year"))
        month = int(single_date.group("month"))
        day_str = single_date.group("day")
        if day_str:
            day = int(day_str)
            start_date = _format_date(year, month, day)
            end_date = _format_date(year, month, day)
        else:
            start_date = _format_date(year, month)
            end_date = _format_date(year, month, month_end=True)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "keyword": keyword,
        }

    # 3-1) 연도 없이 월/일만 지정된 경우는 현재 연도로 보정
    month_day = re.search(
        r"(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일?",
        user_prompt,
    )
    if not month_day:
        month_day = re.search(
            r"(?P<month>\d{1,2})[./-](?P<day>\d{1,2})",
            user_prompt,
        )
    if month_day:
        year = datetime.now().year
        month = int(month_day.group("month"))
        day = int(month_day.group("day"))
        return {
            "start_date": _format_date(year, month, day),
            "end_date": _format_date(year, month, day),
            "keyword": keyword,
        }

    return {
        "start_date": None,
        "end_date": None,
        "keyword": keyword,
    }


def save_plan_parse_log(user_prompt: str, raw_response: str, parsed: PlanGenerateRequest, rule_hints: dict[str, str | None]):
    """
    파서 결과를 /data/outputs 쪽에 남겨 운영 시 추적 가능하게.
    """
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(OUTPUT_ROOT) / f"plan_parse_log_{ts}.json"
        payload = {
            "user_prompt": user_prompt,
            "rule_hints": rule_hints,
            "llm_raw": raw_response,
            "parsed": parsed.dict(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
    except Exception:
        return None


async def parse_plan_request_with_llm(user_prompt: str) -> tuple[PlanGenerateRequest, str]:
    parser_prompt = f"""
너는 사용자의 요청 문장을 PlanGenerateRequest JSON으로 변환하는 파서다.
반드시 아래 JSON 스키마의 키만 사용해서 JSON만 출력해라(설명 금지).

스키마 키:
topic, keyword, start_date, end_date, mode, output_format, extra_instruction

규칙:
- 날짜는 YYYY-MM-DD 형식으로.
- 날짜가 없으면 start_date/end_date는 null.
- topic은 기획서 제목/주제로 가장 적절한 짧은 문장.
- keyword는 대표 키워드 1개(없으면 null).
- extra_instruction에는 강조점/톤/제외요소 등 추가 지시를 넣어라.
- mode는 outline/summary 중 하나(없으면 outline).
- output_format은 md/txt 중 하나(없으면 md).

사용자 요청:
\"\"\"{user_prompt}\"\"\"
""".strip()

    raw = await call_llm(parser_prompt)
    js = extract_json_block(raw)

    try:
        data = json.loads(js)
    except json.JSONDecodeError:
        data = {}

    data.setdefault("mode", "outline")
    data.setdefault("output_format", "md")

    try:
        return PlanGenerateRequest(**data), raw
    except ValidationError:
        return PlanGenerateRequest(
            topic="일기 기반 단편소설 기획서",
            extra_instruction=user_prompt,
        ), raw


async def parse_plan_request(user_prompt: str) -> PlanGenerateRequest:
    """
    1) 룰 기반으로 명확한 날짜/키워드 먼저 잡기
    2) 나머지는 LLM 파서에게 JSON 스키마로 강제
    3) 로그를 남겨 운영 중 파서 품질 추적
    """
    rule_hints = rule_based_plan_parse(user_prompt)
    raw_response = ""

    try:
        parsed, raw_response = await parse_plan_request_with_llm(user_prompt)
    except Exception:
        parsed = PlanGenerateRequest(
            topic="일기 기반 단편소설 기획서",
            extra_instruction=user_prompt,
        )

    # 룰 기반 결과가 있으면 우선 적용 (LLM이 애매하게 잡는 경우 덮어쓰기)
    updates = {k: v for k, v in rule_hints.items() if v}
    if updates:
        parsed = parsed.copy(update=updates)

    save_plan_parse_log(user_prompt, raw_response, parsed, rule_hints)
    return parsed


# =========================
# ✅ NEW: 일기 포맷팅 API
# =========================

def build_diary_format_prompt(req: DiaryFormatRequest) -> str:
    """
    Qwen 계열 모델에 최적화된 고도화 프롬프트.
    - 감정 분석
    - tags 자동 추출 (3~7개)
    - people 추출
    - location 추정
    - projects 자동 분류 (소설아이디어/NGO/IT/가족 중 최대 2개)
    - scene_potential 자동 여부
    - summary 1문장 생성
    - Markdown + YAML 완성
    """

    return f"""
당신은 \"개인 일기 분석 & 구조화 전문가\"이자 \"Qwen 최적화 LLM\"입니다.
당신의 역할은 사용자의 원본 일기를 읽고 다음 규칙에 따라 구조화된 일기 Markdown 파일을 생성하는 것입니다.

========================================================
🎯 출력 규칙(아주 중요): 반드시 지켜야 합니다
========================================================

1) 반드시 YAML 프론트매터부터 시작해야 합니다. (`---` 로 시작하고 `---` 로 닫음)

2) YAML 필드 규칙:
   - date: {req.date} (변경 금지)
   - time: \"{req.time}\" (변경 금지)
   - title: 원본 제목을 기반으로 자연스럽게 정리하되 과한 창작은 금지
   - mood: 영어 소문자 스네이크케이스 (예: happy, sad, anxious_relief, exhausted, mixed_hopeful)
   - mood_score: -1.0 ~ +1.0 실수 (매우 우울 -1.0 / 평범 0.0 / 매우 긍정적 +1.0)
   - tags: 원본 일기의 핵심 키워드 3~7개를 한국어 배열로
   - people: 등장한 인물 또는 관계(아내, 딸, 부모님 등)
   - location: \"home\", \"office\", \"cafe\", \"outdoor\" 등 한 단어로 요약
   - type: \"diary\" 고정
   - projects: \"소설아이디어\", \"NGO\", \"IT\", \"가족\" 중 일기와 가장 관련 높은 1~2개 선택
   - scene_potential: 원본 일기가 소설 장면으로 확장할 가치가 있으면 true 아니면 false
   - summary: 일기를 한 문장으로 정교하게 요약

3) YAML 아래에는 반드시 다음 Markdown 섹션을 포함합니다:
   # 오늘 요약 (3줄)
   # 오늘의 사건
   # 감정 / 생각
   # 배운 것 / 통찰
   # 소설 아이디어 메모 (옵션)
   # TODO / 다음에 이어서 쓸 것 (옵션)

4) 마지막에 원본 텍스트를 그대로 보존해야 합니다:
```text
{req.raw_text}
```
""".strip()


def build_plan_prompt(topic: str, diary_summary: str, extra_instruction: str | None = None) -> str:
    """
    qwen 계열 / 일반 LLM 모두 잘 먹게 설계한 기획서 프롬프트.
    diary_summary에는 mcp-bridge가 만든 다중 소스(diary/ideas/web_research/works/bible 등) 요약/통계 결과를 넣습니다.
    """
    extra = extra_instruction.strip() if extra_instruction else ""

    return f"""
당신은 사용자의 일기와 창작 노트를 기반으로
'단편소설 기획 전문 에디터 & 스토리 컨설턴트'입니다.

아래는 사용자가 작성한 다중 소스(md) 데이터와 요약입니다.
일기뿐 아니라 아이디어(ideas), 웹 리서치(web_research), 기존 작품(works), 성경 메모(bible)까지 활용해 **단편소설 기획서**를 만들어주세요.

[요청 단편소설 기획 주제]
- {topic}

[다중 소스 창작 데이터 요약]
{diary_summary}

--------------------------------------
[작성 규칙 — 반드시 지킬 것]
--------------------------------------

전체 기획서는 아래 구조로 작성합니다.
한국어로, 문학적이지만 과하지 않은 톤을 유지하세요.
패턴·모티프·정서의 흐름을 분석하여 ‘소설적인 의미’를 부여하는 것이 핵심입니다.

# 1. 소설 개요(Concept Overview)
- 이 일기 데이터에서 도출된 핵심 테마 한 문단 요약
- 이야기의 주제(Theme) 제안 1~2개
- 소설의 감정적 색채(톤 & 무드)

# 2. 핵심 메시지 / 주제 의식 (Theme)
- 사용자의 삶에서 반복되는 핵심 정서·통찰을 문학적 주제로 정리
- 이야기의 중심 질문(Central Question) 제안
- 메시지가 독자에게 전달할 감정적 효과

# 3. 등장인물 설계(Character Design)
- 주인공(Protagonist): 성격·결핍·갈망·핵심 상처
- 주요 인물: 아내/가족/또는 상징적 인물 등 일기 기반으로 설계
- 인물 간 관계의 긴장 구조
- 감정적 변화 아크(Character Arc)

# 4. 세계관 및 배경(World & Setting)
- 일기 속 현실을 바탕으로 한 ‘사실적 세계’
- 기술, 감정 기술(E.V.E 같은 요소), 사회 문제(NGO/빈곤 등) 등
  - 현실·근미래·초현실 중 어떤 톤이 어울리는지 제안
- 배경이 상징하는 의미

# 5. 플롯 설계(Plot Structure)
## 5-1. 사건의 흐름(Story Beats)
- 발단 → 전개 → 전환 → 절정 → 결말의 스토리 라인 제안
- 주인공의 감정 변화가 어떻게 진행되는지 묘사

## 5-2. 갈등(Conflict)
- 외적 갈등(가족, 사회적 압박, 기술, 인간관계 등)
- 내적 갈등(두려움, 상처, 신앙, 회복, 자기 의심 등)
- 갈등이 주제와 어떻게 연결되는지 설명

## 5-3. 장면 아이디어(Scene Ideas)
- 기억·일기에서 직접 추출한 ‘장면성 있는 순간’ 3~7개
- 이 장면들을 스토리에 배치하는 방식 제안

# 6. 상징 / 모티프(Motifs & Symbols)
- 반복적으로 나타난 키워드·감정·사건을 문학적 모티프화
- 예: 침묵, 회복, 고통, 가족, 기술 vs 인간, 오해, 사랑 등
- 소설 속 상징적 장치로의 변환 제안

# 7. 작품 톤 & 문체 제안(Style Recommendation)
- 아래는 '선택사항'이며, 반드시 일기 기반 근거를 먼저 제시한 뒤에만 제안하라.
- 추천 작가/문체는 1~2개만 제시하라.
- 이 단편에 어울리는 문장 스타일
- 느린/빠른/서정적/압축적 등 문체 가이드

# 8. 독자 경험 설계(Reader Experience)
- 독자가 느낄 감정 여정
- 소설이 남길 ‘뒷맛’ 혹은 여운

# 9. 향후 발전 가능성
- 장편 확장 가능성 여부
- 동일 세계관에서의 추가 단편 아이디어
- 주인공 또는 설정을 확장할 방안

# 10. 최종 요약(One-Paragraph Logline)
- 위 기획서를 한 문단으로 요약한 로글라인(logline)

--------------------------------------
[톤 & 스타일]
--------------------------------------
- 문학적이되 난해하지 않게.
- 사용자의 삶을 “소설적 재료”로 존중하며 해석.
- 일기 속 상처·믿음·감정은 신중하게 다루고,
  희망의 방향성도 잃지 않도록.
- 스토리는 실현 가능한 구체적 형태로 제안.
--------------------------------------
--------------------------------------
[근거 기반 작성 — 반드시 지킬 것]
--------------------------------------
- 이 기획서는 '사용자의 실제 기록'에 근거해야 한다.
- diary/ideas/web_research/works/bible 모든 타입 데이터를 활용한다.
- 각 섹션마다 아래 형식의 '근거'를 최소 2개 이상 포함하라.
- 근거는 반드시 원문 또는 파일명을 인용해라. 예)
  - [source: diary/2025-12-01_x.md] "<문구>"
  - [source: ideas/…], [source: web_research/…], [source: works/…], [source: bible/…]
- 성경(BIBLE) 인용 시 [성경: 책 장:절 (번역)] 형식을 사용하고, 방향/통찰을 돕는 참고로만 활용한다.
- 인용은 과장하지 말고, 요약에 실제로 존재하는 내용만 사용하라.

[근거 표기 형식]
- 근거: (YYYY-MM-DD) "<요약에서 나온 핵심 문장/키워드>" → 왜 이 근거가 섹션을 뒷받침하는지 1문장 설명

[금지]
- 근거 없이 일반론으로만 쓰는 문장(예: '누구나 성장한다', '감동을 준다')은 금지한다.
- 근거가 빈약하면 '근거가 부족함'을 명시하고, 어떤 정보가 더 필요하다고 제안하라.

추가 참고 지시사항(있으면 반영, 없으면 무시 가능):
{extra}
""".strip()


def slugify_filename(text: str) -> str:
    text = text.strip().replace(" ", "")
    for ch in "/\\?%*:|\"<>":
        text = text.replace(ch, "-")
    return text or "note"


async def _call_playwright_crawl(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{PLAYWRIGHT_MCP_URL}/crawl", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def _call_playwright_fetch(url: str, timeout_ms: int | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """
    Call Playwright MCP fetch endpoint and return (data, error_message).
    """
    payload: dict[str, Any] = {"url": url}
    if timeout_ms is not None:
        payload["timeout_ms"] = timeout_ms

    base_timeout = max(5.0, (timeout_ms or 20000) / 1000 + 5)
    try:
        async with httpx.AsyncClient(timeout=base_timeout) as client:
            resp = await client.post(f"{PLAYWRIGHT_MCP_URL.rstrip('/')}/fetch", json=payload)
    except Exception as exc:
        return None, f"http error: {exc}"

    if resp.status_code >= 400:
        return None, f"playwright mcp returned status {resp.status_code}"

    try:
        data = resp.json()
    except Exception as exc:
        return None, f"invalid json from playwright mcp: {exc}"

    return data, None


def _extract_article_payload(raw: Any) -> tuple[str | None, str | None, str | None]:
    """
    Extract (link, title, body) from various possible MCP payload shapes.
    """
    if not isinstance(raw, dict):
        return None, None, None

    payload = raw
    if isinstance(payload.get("data"), dict):
        payload = payload["data"]

    # try direct keys
    link = payload.get("link") or payload.get("url")
    title = payload.get("title") or payload.get("pageTitle") or payload.get("name")
    body = payload.get("body") or payload.get("text") or payload.get("content")

    # fallback: nested article/result objects
    if body is None:
        for key in ("article", "result", "item"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                link = link or nested.get("link") or nested.get("url")
                title = title or nested.get("title") or nested.get("pageTitle")
                body = nested.get("body") or nested.get("text") or nested.get("content")
                if body:
                    break

    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False)
    if body is not None:
        body = str(body).strip()

    return link, title, body


ARTICLE_NOISE_KEYWORDS = [
    "공유",
    "스크랩",
    "인쇄",
    "글자 크기",
    "폰트",
    "댓글",
    "구독",
    "로그인",
    "앱에서 보기",
    "바로가기",
    "기자",
    "후원",
    "광고",
    "무단전재",
    "재배포",
    "저작권",
    "뉴스 제공",
    "기사 원문",
]

ARTICLE_NOISE_PATTERNS = [
    r"무단전재\s*/\s*재배포 금지",
    r"copyright",
    r"ⓒ",
    r"사진\s*=\s*",
    r"영상\s*=\s*",
    r"관련\s*기사",
    r"기사\s*입력",
    r"기사\s*승인",
]


def _normalize_body_text(text: str) -> str:
    normalized = re.sub(r"\r\n?", "\n", text or "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_paragraphs(text: str, limit: int = 32, max_chars: int = 6000) -> list[str]:
    paragraphs: list[str] = []
    total = 0
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        cleaned = re.sub(r"\s+", " ", block)
        if len(cleaned) < 8:
            continue
        total += len(cleaned)
        if total > max_chars:
            break
        paragraphs.append(cleaned)
        if len(paragraphs) >= limit:
            break
    return paragraphs


def _rule_based_article_cleanup(body: str | None) -> str:
    normalized = _normalize_body_text(body or "")
    if not normalized:
        return ""

    blocks = _split_paragraphs(normalized)
    cleaned: list[str] = []

    for block in blocks:
        lower = block.lower()
        if any(key in lower for key in ARTICLE_NOISE_KEYWORDS):
            continue
        if any(re.search(pat, block, re.I) for pat in ARTICLE_NOISE_PATTERNS):
            continue
        cleaned.append(block)

    # 짧은 본문이라면 원본을 그대로 사용
    if not cleaned and normalized:
        return normalized

    # 중복 단락 제거
    seen: set[str] = set()
    deduped: list[str] = []
    for block in cleaned:
        if block not in seen:
            deduped.append(block)
            seen.add(block)

    return "\n\n".join(deduped)


def _build_llm_body_prompt(title: str, url: str | None, blocks: list[str]) -> str:
    blocks_json = json.dumps(blocks, ensure_ascii=False, indent=2)
    return f"""
너는 뉴스/블로그 기사 본문 식별 및 정제기다.
주어진 후보 문단 중 기사 본문만 남기고 UI/광고/공유/폰트 안내/댓글/구독/저작권 문구를 제거한다.

[출력만 JSON으로]
{{
  "is_article": true|false,
  "clean_body": "본문만 자연스럽게 이어붙인 텍스트",
  "reason": "선택 근거 한 줄"
}}

[입력 정보]
- 제목: {title}
- URL: {url or ""}
- 후보 문단 리스트(JSON): {blocks_json}

규칙:
- clean_body에는 기사 본문 문장만 남겨라. 불필요한 공백과 중복을 없애고 문단 사이에는 빈 줄 1개만 둔다.
- 공유/스크랩/인쇄/글자크기/뉴스 제공/저작권/구독/댓글/관련기사/광고 등 UI 텍스트는 모두 제거.
- 본문이 확실하지 않으면 is_article=false로 하고 reason만 채운다.
""".strip()


async def _llm_select_article_body(title: str, url: str | None, body: str) -> tuple[str | None, str | None]:
    blocks = _split_paragraphs(body)
    if not blocks:
        return None, "no_blocks"

    prompt = _build_llm_body_prompt(title, url, blocks)
    raw = await call_llm(prompt)
    js = extract_json_block(raw)

    try:
        data = json.loads(js)
    except Exception:
        return None, "json_parse_failed"

    if data.get("is_article") is False:
        return None, str(data.get("reason") or "rejected")

    cleaned = _normalize_body_text(data.get("clean_body") or data.get("body") or "")
    if len(cleaned) < 80:
        return None, "llm_body_too_short"

    return cleaned, str(data.get("reason") or "llm_selected")


async def _select_and_clean_article_body(title: str, url: str | None, raw_body: str) -> tuple[str, str | None]:
    precleaned = _rule_based_article_cleanup(raw_body)
    llm_reason: str | None = None

    try:
        llm_body, llm_reason = await _llm_select_article_body(title, url, precleaned)
    except Exception as exc:
        logger.info("[body_clean] llm failed for url=%s err=%s", url, exc)
        llm_body = None

    final_body = llm_body or precleaned or (raw_body or "")
    return final_body, llm_reason


async def _fetch_and_save_article(
    item: dict[str, Any],
    timeout_ms: int,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """
    Fetch article via Playwright MCP and save to WEBRESEARCH_OUT_DIR.
    Returns (saved, failed) where each is a dict.
    """
    link = item.get("link") or item.get("url")
    title = item.get("title") or ""
    if not link:
        return None, {"link": "", "reason": "invalid_link"}

    async with semaphore:
        data, err = await _call_playwright_fetch(link, timeout_ms)
        if err or data is None:
            return None, {"link": link, "reason": f"playwright_failed: {err or 'no data'}"}

        link2, title2, body = _extract_article_payload(data)
        final_link = link2 or link
        final_title = (title2 or title or "").strip()
        if not body:
            return None, {"link": final_link, "reason": "no_body"}

        cleaned_body, llm_reason = await _select_and_clean_article_body(final_title, final_link, body)
        if llm_reason:
            logger.info("[body_clean] llm_reason=%s url=%s", llm_reason, final_link)

        try:
            file_path = save_txt(WEBRESEARCH_OUT_DIR, final_title, final_link, cleaned_body)
        except Exception as exc:
            return None, {"link": final_link, "reason": f"save_failed: {exc}"}

    return {"link": final_link, "title": final_title, "file_path": file_path}, None


async def _search_google_news_rss(query: str, max_results: int) -> tuple[list[dict[str, str]], str | None]:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=ko&gl=KR&ceid=KR:ko"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:
        return [], f"http error: {exc}"

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(text)
        items: list[dict[str, str]] = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            if link and title:
                items.append({"link": link, "title": title})
            if len(items) >= max_results:
                break
        return items, None
    except Exception as exc:
        return [], f"parse error: {exc}"


async def _search_searxng(query: str, max_results: int) -> tuple[list[dict[str, str]], str | None]:
    if not SEARXNG_URL:
        return [], "SEARXNG_URL not configured"

    params = {"q": query, "format": "json", "engines": "news"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(SEARXNG_URL.rstrip("/") + "/search", params=params)
    except Exception as exc:
        return [], f"http error: {exc}"

    if resp.status_code >= 400:
        return [], f"searxng status {resp.status_code}"

    try:
        data = resp.json()
    except Exception as exc:
        return [], f"invalid json: {exc}"

    results = []
    for item in data.get("results", []):
        link = item.get("url") or item.get("link")
        title = item.get("title") or ""
        if link and title:
            results.append({"link": link, "title": title})
        if len(results) >= max_results:
            break

    return results, None


@app.post("/api/plan/from-prompt", response_model=PlanGenerateEnvelope)
async def plan_from_prompt(body: PlanGeneratePromptOnlyRequest):
    """
    Open WebUI에서 자연어 한 줄(prompt)만 보내도
    - 날짜/키워드/형식 등을 자동 추출한 뒤
    - 멀티소스 데이터 로딩 → mcp-bridge select-and-summarize → 기획서 생성 흐름을 그대로 사용.
    """
    parsed_req = await parse_plan_request(body.prompt)
    # V3.1: from-prompt도 from-data 파이프라인을 사용해
    # 1) prompt → JSON 파라미터 추출
    # 2) 파일 목록 필터링(_load_markdown_entries)
    # 3) mcp-bridge select-and-summarize
    # 4) 기획서 생성 LLM 호출
    data_req = PlanFromDataRequest(
        goal=parsed_req.topic or "단편소설 기획서 제작",
        include=body.include or None,  # 없으면 from-data 내부에서 기본 5종 사용
        start_date=parsed_req.start_date,
        end_date=parsed_req.end_date,
        keyword=parsed_req.keyword,
        extra_instruction=parsed_req.extra_instruction,
        prompt=body.prompt,  # 원본 프롬프트 저장/로깅용
    )
    plan = await generate_plan_from_data_internal(data_req)
    return PlanGenerateEnvelope(success=True, message="ok", data=plan, error=None)


@app.post("/api/diary/preview", response_model=DiaryFormatResponse)
async def api_diary_preview(body: DiaryFormatRequest):
    """
    ✅ 포맷 결과 '미리보기' 전용 엔드포인트
    - 파일을 저장하거나 .txt를 이동하지 않음
    - 그냥 LLM 결과만 반환
    - Open WebUI / 별도 Web UI 에서 바로 호출해서 화면에 보여주기 용도
    """
    prompt = build_diary_format_prompt(body)
    md_text = await call_llm(prompt)
    return DiaryFormatResponse(result=md_text)


def build_diary_repair_prompt(original_md: str) -> str:
    """
    기존에 저장된 일기 Markdown을 입력받아:
    - YAML 메타데이터를 점검/보완
    - 누락된 필드를 채우고, 이상한 값은 자연스럽게 수정
    - 본문 섹션 구조는 유지하되, 약간의 다듬기는 허용
    - 이미 존재하는 mood, mood_score 값은 덮어쓰지 말고 그대로 유지
    """
    return f"""
당신은 '일기 메타데이터 검수/보정 전문가'입니다.

입력: 사용자의 기존 일기 Markdown (YAML + 본문)

작업 목표:
1. YAML front-matter를 점검하고 아래 필드들을 모두 올바르게 채우세요.
이때 YAML fornt-matter가 파일의 최상단에 오도록 수정하세요.
   - date: YYYY-MM-DD 형식
   - time: \"HH:MM\" 형식 (문자열)
   - title: 자연스럽지만 과하지 않게
   - mood: 영어 소문자 스네이크케이스 (예: mixed_hopeful, deeply_tired, calm, anxious_relief)
   - mood_score: -1.0 ~ +1.0 실수
   - tags: 일기 내용을 가장 잘 대표하는 한국어 키워드 3~7개
   - people: 등장 인물/관계 리스트
   - location: home, office, cafe, outdoor 등
   - type: diary
   - projects: [\"소설아이디어\",\"NGO\",\"IT\",\"가족\"] 중 가장 관련 있는 것 1~2개
   - scene_potential: 소설 장면으로 쓸 만하면 true, 아니면 false
   - summary: 이 일기를 한 문장으로 요약한 한국어 문장
   - mood / mood_score 가 이미 YAML에 있을 경우 값은 절대 변경하거나 삭제하지 말고 그대로 둔다.
     존재하지 않을 때만 새로 추정하여 추가한다.

2. 본문 섹션:
   - # 오늘 요약 (3줄)
   - # 오늘의 사건
   - # 감정 / 생각
   - # 배운 것 / 통찰
   - # 소설 아이디어 메모 (옵션)
   - # TODO / 다음에 이어서 쓸 것 (옵션)
   이 섹션 구조는 유지하되, 내용이 너무 빈약하면 자연스럽게 조금 보완해도 됩니다.

3. “입력에 YAML이 있더라도 최종 결과는 YAML front-matter를 오직 1개만 만들고, 나머지 YAML 블록은 절대 남기지 마세요.”

4. “입력에 ‘참고용 원문(출력 금지)’ 섹션이 있더라도, 최종 출력에는 절대 포함하지 말고 제거하세요.”

아래는 사용자가 저장한 기존 Markdown입니다.
이를 기반으로 위 규칙에 맞는 완성된 Markdown 전체로 대체하세요.

----- 기존 Markdown 시작 -----
{original_md}
----- 기존 Markdown 끝 -----
""".strip()


def build_data_repair_prompt(original_md: str, doc_type: str) -> str:
    """
    idea/work/web_research/bible 용 범용 YAML 보정 프롬프트.
    - diary 전용 필드(mood/mood_score)는 건드리지 않는다.
    - 기존 필드는 삭제 금지, 없으면 추가만.
    """
    return f"""
당신은 'Markdown 메타데이터 검수/보정 전문가'입니다.
입력: 사용자의 기존 Markdown (YAML + 본문)

목표:
1) YAML front-matter를 파일 최상단에 배치하고 아래 필드를 모두 채우세요.
   - 출력은 반드시 '---'로 시작하는 YAML 프론트매터로 시작해야 합니다. 어떤 설명 문구도 YAML 위에 넣지 마세요.
   - type: "{doc_type}"
   - title: 자연스럽게 정리 (없으면 본문/파일명에서 추정)
   - created_at / updated_at: ISO datetime. 기존 값이 있으면 유지, 없으면 현재 시각 또는 문맥에서 추정
   - tags: 핵심 키워드 3~7개 리스트
   - topics: 주제 2~4개 리스트
   - people: 등장 인물/관계 리스트
   - locations: 위치/공간 관련 단어 리스트
   - source: 원문 출처(URL/서적 등) 또는 null
   - usage: 이 문서의 용도 리스트 (예: ["planning"], ["reference"], ["critique"])
   - summary: 본문을 한 문장으로 요약 (이미 있으면 유지)
   - 이미 존재하는 추가/커스텀 필드는 절대 삭제하지 말고 그대로 유지
   - diary 전용 필드(mood, mood_score)는 이 문서 타입에선 추가/수정하지 말 것

2) 본문 구조:
   - 기존 본문 섹션은 최대한 유지
   - 내용이 빈약하면 자연스럽게 보완 가능하나, 원본 의미를 과도하게 변형하지 말 것

3) 코드 블럭/인용문 등 원본 텍스트는 훼손하지 마세요.
   - 이미 원본 텍스트 코드블럭(예: ```text ... ```)이 있으면 그대로 보존하세요.
   - 없다면 문서 맨 아래에 "원본 텍스트 (자동 보존)" 섹션을 만들고 ```text 코드블럭``` 안에 입력 원문을 그대로 넣으세요.

4) 출력 형식:
   - 설명/해설/주석 없이 최종 Markdown만 출력하세요.
   - YAML 프론트매터 바로 뒤에 본문을 이어서 작성하세요.

----- 기존 Markdown 시작 -----
{original_md}
----- 기존 Markdown 끝 -----
""".strip()


@app.post("/api/diary/reformat-md", response_model=DiaryReformatResponse)
async def api_diary_reformat_md(body: DiaryReformatRequest):
    prompt = build_diary_repair_prompt(body.markdown)

    new_md = await call_llm_with_front_matter_retry(
        prompt=prompt,
        validate_meta_fn=validate_diary_front_matter,
        retries=int(os.environ.get("LLM_VALIDATE_RETRIES", "2")),
        must_have_front_matter=True,
    )
    return DiaryReformatResponse(result=new_md)

@app.post("/api/data/reformat-md")
async def api_data_reformat_md(body: DataReformatRequest):
    """
    범용 데이터(md) YAML 보정 API.
    - doc_type: idea | work | web_research | bible
    - diary 전용 mood/mood_score는 건드리지 않음.
    """
    allowed = {"idea", "work", "web_research", "bible"}
    doc_type = body.doc_type.strip().lower()
    if doc_type not in allowed:
        return standard_response(
            success=False,
            message="invalid doc_type",
            data=None,
            error=f"doc_type must be one of {sorted(allowed)}",
        )


    prompt = build_data_repair_prompt(body.markdown, doc_type)
    try:
        new_md = await call_llm_with_front_matter_retry(
            prompt=prompt,
            validate_meta_fn=lambda meta: validate_data_front_matter(meta, doc_type),
            retries=int(os.environ.get("LLM_VALIDATE_RETRIES", "2")),
            must_have_front_matter=True,
        )
        return standard_response(success=True, message="ok", data={"result": new_md}, error=None)
    except Exception as exc:
        return standard_response(
            success=False,
            message="reformat failed",
            data=None,
            error=str(exc),
        )


# =========================
# 🌐 Web UI 라우트
# =========================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
):
    mood_stats = await get_mood_stats(start_date=start_date, end_date=end_date)
    data_roots = {
        "diary": DIARY_ROOT,
        "ideas": IDEAS_ROOT,
        "works": WORKS_ROOT,
        "bible": BIBLE_ROOT,
        "web_research": WEB_RESEARCH_ROOT,
    }
    data_files = {
        name: _list_files_under(path, limit=MAX_FILES_PER_TYPE * 30)
        for name, path in data_roots.items()
    }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": os.environ.get("DASHBOARD_USER", "guest"),
            "mood_stats": mood_stats,
            "data_roots": data_roots,
            "data_files": data_files,
        },
    )


@app.get("/llm-info")
async def llm_info():
    return {
        "backend": LLM_BACKEND,
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL if LLM_BACKEND == "ollama" else None,
        "openai_model": OPENAI_MODEL if LLM_BACKEND == "openai" else None,
    }


# =========================
# 📚 데이터 기반 기획서 생성 (NEW)
# =========================

DATA_ROOTS = {
    "diary": Path(DIARY_ROOT),
    "ideas": Path(IDEAS_ROOT),
    "web_research": Path(WEB_RESEARCH_ROOT),
    "works": Path(WORKS_ROOT),
    "bible": Path(BIBLE_ROOT),
}


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """
    간단한 YAML front-matter 파서. 없으면 ({}, 전체 텍스트) 반환.
    """
    if not text.lstrip().startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta_raw = parts[1]
    body = parts[2]
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except Exception:
        meta = {}
    return meta, body


def _ensure_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    return [str(val)]


def _parse_meta_date(meta: dict[str, Any]) -> date | None:
    for key in ("date", "created_at", "updated_at"):
        val = meta.get(key)
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val).date()
            except Exception:
                continue
    return None


def _guess_date_from_filename(path: Path) -> date | None:
    name = path.stem
    candidates = [name[:10], name.split("_")[0]]
    for cand in candidates:
        try:
            return datetime.fromisoformat(cand).date()
        except Exception:
            pass
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except Exception:
            pass
    return None


def _matches_filter(meta: dict[str, Any], body: str, rel_path: str, start_date: str | None, end_date: str | None, keyword: str | None) -> bool:
    def parse_date_safe(s: str | None):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None

    diary_date = _parse_meta_date(meta) or _guess_date_from_filename(Path(rel_path))

    if start_date:
        s = parse_date_safe(start_date)
        if s and diary_date and diary_date < s:
            return False
    if end_date:
        e = parse_date_safe(end_date)
        if e and diary_date and diary_date > e:
            return False

    if keyword:
        meta_fields = (
            [meta.get("title") or ""]
            + _ensure_list(meta.get("tags"))
            + _ensure_list(meta.get("topics"))
            + _ensure_list(meta.get("people"))
            + _ensure_list(meta.get("locations"))
        )
        in_meta = any(keyword in t for t in meta_fields)
        if (not in_meta) and (keyword not in body):
            return False
    return True


def _load_markdown_entries(kind: str, start_date: str | None, end_date: str | None, keyword: str | None, limit: int = MAX_FILES_PER_TYPE) -> list[dict[str, Any]]:
    root = DATA_ROOTS.get(kind)
    if not root or not root.exists():
        return []

    files = sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    entries: list[dict[str, Any]] = []

    for path in files:
        if len(entries) >= limit:
            break
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _split_front_matter(text)

        if not _matches_filter(meta, body, str(path.relative_to(root)), start_date, end_date, keyword):
            continue

        entry = {
            "path": str(path),
            "rel_path": str(path.relative_to(root)),
            "title": meta.get("title") or path.stem,
            "tags": _ensure_list(meta.get("tags")),
            "topics": _ensure_list(meta.get("topics")),
            "summary": meta.get("summary") or "",
            "content": body.strip(),
            "meta": meta,
        }
        entries.append(entry)

    return entries


def _render_section(label: str, entries: list[dict[str, Any]]):
    if not entries:
        return f"[{label}]\n- (데이터 없음)\n"
    lines = [f"[{label}]"]
    for e in entries:
        meta = e.get("meta", {})
        tags = _ensure_list(meta.get("tags") or e.get("tags"))
        topics = _ensure_list(meta.get("topics") or e.get("topics"))
        meta_info = []
        if tags:
            meta_info.append(f"tags={','.join(tags)}")
        if topics:
            meta_info.append(f"topics={','.join(topics)}")
        meta_str = f" ({'; '.join(meta_info)})" if meta_info else ""
        title = e.get("title") or (meta.get("title") or e.get("rel_path"))
        rel_path = e.get("rel_path") or e.get("path") or ""
        lines.append(f"## {title} [{rel_path}] {meta_str}".strip())
        summary_line = meta.get("summary") or e.get("summary")
        if summary_line:
            lines.append(f"- summary: {summary_line}")
        content = (
            e.get("content")
            or e.get("excerpt")
            or e.get("body")
            or ""
        ).strip()
        excerpt = content[:1200]
        if len(content) > 1200:
            excerpt += "\n...[본문 길이 초과로 일부만 포함]"
        if excerpt:
            lines.append(excerpt)
        lines.append("")
    return "\n".join(lines)


def build_plan_prompt_from_data(goal: str, data: dict[str, list[dict[str, Any]]], extra_instruction: str | None = None) -> str:
    """
    데이터 소스별 섹션을 분리해 LLM에 전달.
    WORKS/BIBLE 규칙 포함.
    """
    extra = (extra_instruction or "").strip()

    prompt_parts = [
        "당신은 사용자의 다중 소스(md 파일) 데이터를 읽고 단편소설 기획서를 작성하는 전문가입니다.",
        "각 섹션을 기반으로 근거를 명확히 제시하고, 사실을 창작하지 마세요.",
        f"[요청 목표]\n- {goal}",
        "",
        _render_section("DIARY", data.get("diary", [])),
        _render_section("IDEAS", data.get("ideas", [])),
        _render_section("WEB_RESEARCH", data.get("web_research", [])),
        _render_section("WORKS", data.get("works", [])),
        _render_section("BIBLE", data.get("bible", [])),
        "",
        "[작성 규칙]",
        "- WORKS(작품)에서 아이디어를 차용할 때는 반드시 아래 형식을 포함:",
        "  [참고 작품: <파일명>]",
        "  - 해당 아이디어가 나온 이유:",
        "  - 작품의 이 아이디어가 적합한 근거:",
        "- BIBLE 데이터를 인용할 때:",
        "  - 직접 인용 시 반드시 [성경: 책 장:절 (번역)] 형식으로 표기",
        "  - 기획 방향/통찰 보조용으로만 사용하고, 사람/작품을 단정/심판하는 표현은 금지",
        "- 섹션별 원문 근거를 인용하며, 없는 경우 '근거 부족'을 명시",
        "- 기존 기획서 톤을 유지하되 데이터 근거 우선으로 작성",
    ]

    if extra:
        prompt_parts.append(f"- 추가 지시: {extra}")

    return "\n".join(prompt_parts)


async def _apply_prompt_to_data_request(req: PlanFromDataRequest) -> PlanFromDataRequest:
    """
    plan/from-data 전용: prompt가 들어오면 parse_plan_request를 통해
    start_date/end_date/keyword/goal(extra) 를 채워넣는다.
    기존 필드가 이미 주어졌다면 덮어쓰지 않는다.
    """
    if not req.prompt:
        return req
    try:
        parsed = await parse_plan_request(req.prompt)
    except Exception:
        return req

    updates: dict[str, Any] = {}
    if (not req.start_date) and parsed.start_date:
        updates["start_date"] = parsed.start_date
    if (not req.end_date) and parsed.end_date:
        updates["end_date"] = parsed.end_date
    if (not req.keyword) and parsed.keyword:
        updates["keyword"] = parsed.keyword
    if (not req.extra_instruction) and parsed.extra_instruction:
        updates["extra_instruction"] = parsed.extra_instruction
    # goal은 기본값일 때만 topic으로 대체
    if (req.goal == "단편소설 기획서 제작" or not req.goal) and parsed.topic:
        updates["goal"] = parsed.topic
    return req.copy(update=updates)


def _render_sources_list(sources: dict[str, list[str]]) -> str:
    """
    기획서 결과물에 사람이 바로 확인할 수 있는 참고 파일 목록을 추가.
    """
    if not sources:
        return "## 참고 소스 목록\n- (없음)"
    lines = ["## 참고 소스 목록"]
    for kind, items in sources.items():
        if not items:
            lines.append(f"- {kind}: (없음)")
            continue
        for path in items:
            lines.append(f"- {kind}: {path}")
    return "\n".join(lines)

def build_plan_header_yaml(title: str, goal: str, include: list[str], sources: dict[str, list[str]]) -> str:
    now_iso = datetime.now().isoformat()
    header = {
        "type": "plan",
        "title": title,
        "goal": goal,
        "include": include,
        "created_at": now_iso,
        "updated_at": now_iso,
        "usage": ["planning"],
        "sources": sources,   # ✅ validate_plan_front_matter의 sources 요구 충족
    }
    return "---\n" + yaml.safe_dump(header, allow_unicode=True, sort_keys=False) + "---\n\n"

async def generate_plan_from_data_internal(req: PlanFromDataRequest) -> PlanGenerateResponse:
    req = await _apply_prompt_to_data_request(req)
    includes = req.include or ["diary", "ideas", "web_research", "works", "bible"]
    includes = [i for i in includes if i in DATA_ROOTS]
    topic = req.goal or "단편소설 기획서 제작"

    # mcp-bridge 새 select-and-summarize 사용: 파일 선택 + 요약
    selection = await select_and_summarize(
        include=includes,
        start_date=req.start_date,
        end_date=req.end_date,
        keyword=req.keyword,
        extra_instruction=req.extra_instruction,
        limit_per_type=MAX_FILES_PER_TYPE,
        preview_chars=1200,
    )

    filtered_data = selection.get("entries", {})
    summary_text = selection.get("result", "")
    sources_dict = selection.get("sources", {})

    # 기존 build_plan_prompt 템플릿(1~10 섹션 + 톤/스타일)을 그대로 활용
    sections = []
    for kind in includes:
        label = kind.upper()
        entries = filtered_data.get(kind, [])
        sections.append(_render_section(label, entries))
    multi_source_summary = "\n\n".join(sections)
    multi_source_summary = multi_source_summary + "\n\n[멀티소스 요약]\n" + summary_text

    prompt = build_plan_prompt(
        topic=topic,
        diary_summary=multi_source_summary,
        extra_instruction=req.extra_instruction,
    )

    plan_text = await call_llm(prompt)

    # (선택) 본문 품질 규칙: 너무 짧으면 재시도 같은 간단 규칙
    if len(plan_text.strip()) < 500:
        # 재시도: (프롬프트 강화)
        plan_text = await call_llm(prompt + "\n\n[품질 기준] 최소 500자 이상, 섹션 구조를 반드시 채워라.")

    sources_section = _render_sources_list(sources_dict)
    plan_text_with_sources = f"{plan_text}\n\n---\n{sources_section}"

    header_str = build_plan_header_yaml(
        title=req.goal or "단편소설 기획서 제작",
        goal=req.goal or "단편소설 기획서 제작",
        include=includes,
        sources=sources_dict,
    )
    final_text = header_str + plan_text_with_sources

    final_text = header_str + plan_text_with_sources

    # ✅ 최종 결과(front-matter 포함) 검증
    meta, _, has_fm = extract_front_matter(final_text)
    if not has_fm:
        raise ValueError("plan output missing front-matter (server bug)")
    validate_plan_front_matter(meta)

    file_path = save_plan_output(final_text)

    return PlanGenerateResponse(
        title=req.goal,
        content=final_text,
        file_path=file_path,
        sources=[sources_dict],
    )


@app.post("/api/plan/from-data", response_model=PlanGenerateEnvelope)
async def plan_from_data(req: PlanFromDataRequest):
    """
    다중 데이터 소스(diary/ideas/web_research/works/bible)를 섹션별로 LLM에 전달해 단편소설 기획서를 생성한다.
    OpenWebUI HTTP Tool 설정 예시:
    - Method: POST
    - URL: http://waai-backend:8000/api/plan/from-data
    - Headers: Content-Type: application/json
    - Body 예시:
      {
        "prompt": "12월 가족 일기 기반으로 따뜻한 감동 단편 기획서. 12/1~12/31 사이 기록만, 희망적 결말.",
        "goal": "최근 일기 기반 단편소설 기획",
        "include": ["diary", "ideas", "web_research", "works", "bible"],
        "start_date": "2025-12-01",
        "end_date": "2025-12-31",
        "keyword": "가족",
        "extra_instruction": "희망적 결말로 마무리"
      }
    - 성공 응답 예시:
      {
        "success": true,
        "message": "ok",
        "data": {
          "title": "...",
          "content": "...(기획서 본문)...",
          "file_path": "/data/outputs/202512xx_plan.md",
          "sources": [{"diary": ["2025-12-10.md"], "ideas": ["foo.md"]}]
        },
        "error": null
      }
    - 실패 응답 예시:
      {
        "success": false,
        "message": "plan generation failed",
        "data": null,
        "error": "에러 메시지"
      }

    curl 테스트:
    curl -X POST http://waai-backend:8000/api/plan/from-data \\
      -H "Content-Type: application/json" \\
      -d '{"prompt":"12월 가족 일기 기반 기획서","include":["diary","ideas"],"keyword":"가족"}'
    """
    try:
        plan = await generate_plan_from_data_internal(req)
        return PlanGenerateEnvelope(
            success=True,
            message="ok",
            data=plan,
            error=None,
        )
    except Exception as exc:
        return PlanGenerateEnvelope(
            success=False,
            message="plan generation failed",
            data=None,
            error=str(exc),
        )


def _list_files_under(root: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    """
    /data 내부 파일을 최신 수정순으로 적당히 보여주기 위한 헬퍼.
    """
    base = Path(root)
    if not base.exists():
        return []

    files: list[tuple[float, Path]] = []
    for path in base.rglob("*"):
        if path.is_file():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            files.append((mtime, path))

    files.sort(key=lambda x: x[0], reverse=True)
    items: list[dict[str, Any]] = []

    for mtime, path in files[:limit]:
        rel = str(path.relative_to(base))
        try:
            size_kb = round(path.stat().st_size / 1024, 1)
        except OSError:
            size_kb = None
        items.append(
            {
                "rel_path": rel,
                "name": path.name,
                "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                "size_kb": size_kb,
            }
        )
    return items


# =========================
# 🌐 Playwright 웹 리서치 API
# =========================


@app.post("/api/playwright/crawl", response_model=dict, operation_id="playwright_crawl")
async def playwright_crawl(req: PlaywrightCrawlRequest):
    """
    OpenWebUI 커스텀 툴에서 직접 호출할 수 있는 Playwright 크롤링 엔드포인트.
    - url이 주어지면 Playwright MCP의 fetch 엔드포인트를 호출해 단일 페이지 본문을 반환
    - (호환성) 기존 prompt/keywords 입력은 기존 크롤링 플로우로 동작
    """
    if req.url:
        data, err = await _call_playwright_fetch(req.url, req.timeout_ms)
        if err or data is None:
            return standard_response(
                success=False,
                message="playwright fetch failed",
                data=None,
                error=err or "no data",
            )

        link, title, body = _extract_article_payload(data)
        if not body:
            return standard_response(
                success=False,
                message="playwright fetch returned no body",
                data=None,
                error="missing body/text/content",
            )

        return standard_response(
            success=True,
            message="ok",
            data={"link": link or req.url, "title": title or "", "body": body},
            error=None,
        )

    keywords = [k.strip() for k in req.keywords if k and k.strip()]
    if not keywords and req.prompt:
        keywords = _extract_keywords_from_prompt(req.prompt, limit=5)
    keywords = keywords[:5]
    if not keywords:
        return standard_response(success=False, message="keywords required", data=None, error="no keywords")

    per_keyword = max(1, min(req.per_keyword, 5))
    payload = {"keywords": keywords, "perKeyword": per_keyword}
    data = await _call_playwright_crawl(payload)
    if data is None:
        return standard_response(success=False, message="playwright crawl failed", data=None, error="call failed")

    saved = data.get("saved_files") or data.get("savedFiles") or []
    count = data.get("count") or len(saved)
    articles = data.get("articles") or []
    return standard_response(
        success=True,
        message="ok",
        data={"saved_files": saved, "count": count, "keywords": keywords, "articles": articles},
        error=None,
    )


@app.post("/api/web/search", response_model=dict, operation_id="web_search")
async def web_search(req: WebSearchRequest):
    query = normalize_query(req.query)
    if not query:
        return standard_response(success=False, message="query required", data=None, error="empty query")

    max_results = max(1, min(req.max_results or 5, 20))
    engine = (req.engine or "google_news_rss").lower()

    items: list[dict[str, str]] = []
    error: str | None = None

    if engine == "searxng" or (engine != "google_news_rss" and SEARXNG_URL):
        items, error = await _search_searxng(query, max_results)
        engine_used = "searxng"
    else:
        items, error = await _search_google_news_rss(query, max_results)
        engine_used = "google_news_rss"

    if error:
        return standard_response(success=False, message="search failed", data=None, error=error)

    return standard_response(
        success=True,
        message="ok",
        data={"query": query, "engine": engine_used, "items": items[:max_results]},
        error=None,
    )


MAX_PLAYWRIGHT_CONCURRENCY = 2


@app.post("/api/web_search/fetch", response_model=dict, operation_id="web_search_fetch")
async def web_search_fetch(req: WebSearchFetchRequest):
    query = normalize_query(req.query)
    if not query:
        return standard_response(success=False, message="query required", data=None, error="empty query")

    max_results = max(1, min(req.max_results or 5, 20))
    search_result = await web_search(WebSearchRequest(query=query, max_results=max_results, engine=req.engine))

    if not search_result.get("success"):
        return standard_response(
            success=False,
            message="search failed",
            data=None,
            error=search_result.get("error") or "search_failed",
        )

    items = (search_result.get("data") or {}).get("items") or []
    if not items:
        logger.info("[web_search_fetch] no search results for query=%s", query)
        return standard_response(
            success=False,
            message="no search results",
            data=None,
            error="no_search_results",
        )

    items = items[:max_results]
    logger.info("[web_search_fetch] query=%s engine=%s results=%d", query, req.engine, len(items))

    semaphore = asyncio.Semaphore(MAX_PLAYWRIGHT_CONCURRENCY)
    tasks = [
        _fetch_and_save_article(item, req.timeout_ms or 20000, semaphore)
        for item in items
    ]
    results = await asyncio.gather(*tasks)

    saved: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for ok, err in results:
        if ok:
            saved.append(ok)
            logger.info("[web_search_fetch] saved title=%s path=%s", ok.get("title", ""), ok.get("file_path", ""))
        elif err:
            failed.append(err)
            logger.info("[web_search_fetch] failed link=%s reason=%s", err.get("link", ""), err.get("reason", ""))

    message = f"saved {len(saved)} items"
    return standard_response(
        success=bool(saved) or not failed,  # allow partial success
        message=message,
        data={"query": query, "saved": saved, "failed": failed},
        error=None if saved or not failed else "all_failed",
    )


# =========================
# 🌐 Playwright 웹리서치 스케줄러
# =========================

_playwright_logs: deque[dict[str, Any]] = deque(maxlen=30)
_playwright_scheduler_task: asyncio.Task | None = None


def _load_playwright_schedule() -> PlaywrightScheduleConfig:
    if PLAYWRIGHT_SCHEDULE_PATH.exists():
        try:
            data = json.loads(PLAYWRIGHT_SCHEDULE_PATH.read_text(encoding="utf-8"))
            return PlaywrightScheduleConfig(**data)
        except Exception:
            pass
    return PlaywrightScheduleConfig()


def _save_playwright_schedule(cfg: PlaywrightScheduleConfig):
    PLAYWRIGHT_SCHEDULE_PATH.write_text(cfg.model_dump_json(ensure_ascii=False, indent=2), encoding="utf-8")


def _log_playwright_run(kind: str, keywords: list[str], count: int, saved: list[str], error: str | None = None):
    _playwright_logs.appendleft(
        {
            "time": datetime.now().isoformat(),
            "kind": kind,
            "keywords": keywords,
            "count": count,
            "saved_files": saved[:5],
            "error": error,
        }
    )


async def _playwright_scheduler_loop():
    while True:
        cfg = _load_playwright_schedule()
        if cfg.enabled and cfg.keywords:
            payload = {
                "keywords": cfg.keywords[:5],
                "perKeyword": max(1, min(cfg.per_keyword, 5)),
            }
            data = await _call_playwright_crawl(payload)
            now_iso = datetime.now().isoformat()
            if data is None:
                cfg.last_run = now_iso
                cfg.last_error = "playwright call failed"
                cfg.last_count = 0
                _log_playwright_run("schedule", payload["keywords"], 0, [], cfg.last_error)
            else:
                saved = data.get("saved_files") or data.get("savedFiles") or []
                count = data.get("count") or len(saved)
                cfg.last_run = now_iso
                cfg.last_error = None
                cfg.last_count = count
                _log_playwright_run("schedule", payload["keywords"], count, saved, None)
            _save_playwright_schedule(cfg)

        interval = max(1, cfg.interval_minutes) * 60
        await asyncio.sleep(interval)


@app.on_event("startup")
async def _start_playwright_scheduler():
    global _playwright_scheduler_task
    if _playwright_scheduler_task is None:
        _playwright_scheduler_task = asyncio.create_task(_playwright_scheduler_loop())


@app.get("/api/playwright/schedule", response_model=dict, operation_id="get_playwright_schedule")
async def get_playwright_schedule():
    cfg = _load_playwright_schedule()
    return standard_response(success=True, message="ok", data=cfg.dict(), error=None)


@app.post("/api/playwright/schedule", response_model=dict, operation_id="set_playwright_schedule")
async def set_playwright_schedule(cfg: PlaywrightScheduleConfig):
    cfg.interval_minutes = max(1, cfg.interval_minutes)
    cfg.per_keyword = max(1, min(cfg.per_keyword, 5))
    cfg.keywords = [k.strip() for k in cfg.keywords if k and k.strip()][:5]
    _save_playwright_schedule(cfg)
    return standard_response(success=True, message="saved", data=cfg.dict(), error=None)


@app.get("/api/playwright/status", response_model=dict, operation_id="get_playwright_status")
async def get_playwright_status():
    return standard_response(success=True, message="ok", data=list(_playwright_logs), error=None)


# =========================
# 📑 단편소설 합평 API (NEW)
# =========================

CRITIQUE_FALLBACK_RULES = """- 등장인물의 목표와 갈등이 뚜렷한가?
- 장면마다 구체적 감정/감각 묘사가 있는가?
- 사건 진행이 논리적으로 이어지는가?
- 대사가 인물 성격과 상황에 맞는가?
- 마무리가 주제 의식과 정서적 여운을 전달하는가?
"""


def _load_critique_criteria() -> str:
    if CRITIQUE_CRITERIA_PATH.exists():
        try:
            return CRITIQUE_CRITERIA_PATH.read_text(encoding="utf-8")
        except Exception:
            return CRITIQUE_FALLBACK_RULES
    return CRITIQUE_FALLBACK_RULES


def _save_critique_object(title: str, content: str) -> str:
    now_iso = datetime.now().isoformat()
    yaml_header = {
        "type": "critique_object",
        "title": title,
        "created_at": now_iso,
        "updated_at": now_iso,
        "usage": ["critique", "reference"],
    }
    md = "---\n" + yaml.safe_dump(yaml_header, allow_unicode=True, sort_keys=False) + "---\n\n" + content
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = slugify_filename(title)
    path = CRITIQUE_OBJECTS_ROOT / f"{ts}_{safe_title}.md"
    path.write_text(md, encoding="utf-8")
    return str(path)


def _save_critique_result(title: str, critique_text: str, object_path: str) -> str:
    now_iso = datetime.now().isoformat()
    yaml_header = {
        "type": "critique",
        "object_title": title,
        "created_at": now_iso,
        "updated_at": now_iso,
        "source_object_file": object_path,
        "usage": ["critique"],
    }
    md = "---\n" + yaml.safe_dump(yaml_header, allow_unicode=True, sort_keys=False) + "---\n\n" + critique_text
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CRITIQUE_RESULTS_ROOT / f"{ts}_{slugify_filename(title)}_critique.md"
    path.write_text(md, encoding="utf-8")
    return str(path)


def _build_critique_prompt(title: str, content: str, criteria: str, extra_instruction: str | None = None) -> str:
    extra = extra_instruction.strip() if extra_instruction else ""
    extra_block = f"\n\n[추가 지시사항]\n{extra}" if extra else ""
    return f"""
너는 단편소설 합평 전문 에디터다. 아래 구조를 반드시 지켜 출력하라.

[합평 기준]
{criteria}

[입력 원고 제목]
{title}

[입력 원고]
{content}
{extra_block}

--- 출력 형식 (반드시 이 순서로) ---
(1) 한 줄 총평
(2) 항목별 점수: 각 항목 10점 만점, 근거로 원고 문장/문단을 인용
(3) 개선 제안: 구체적으로 몇 라인/어느 문단을 어떤 표현·묘사·방향으로 수정할지 제안 (장면/문단 단위)
(4) 기준 준수 여부 체크리스트: 합평기준규칙.md 항목을 그대로 나열하고 각 항목에 대해 준수/미흡 + 한 줄 근거
""".strip()


def _build_critique_chunk_prompt(
    title: str,
    content: str,
    criteria: str,
    part_index: int,
    total_parts: int,
    extra_instruction: str | None = None,
) -> str:
    extra = extra_instruction.strip() if extra_instruction else ""
    extra_block = f"\n\n[추가 지시사항]\n{extra}" if extra else ""
    return f"""
너는 단편소설 합평 전문 에디터다. 아래 구조를 반드시 지켜 출력하라.
이 파트는 원고의 일부이므로, 이 파트의 내용에만 근거해 평가하라.

[합평 기준]
{criteria}

[입력 원고 제목]
{title}

[입력 원고 파트 {part_index}/{total_parts}]
{content}
{extra_block}

--- 출력 형식 (반드시 이 순서로) ---
(0) 파트 요약: 2~3문장으로 현재 파트에서 무슨 일이 벌어지는지 요약
(1) 한 줄 총평
(2) 항목별 점수: 각 항목 10점 만점, 근거로 원고 문장/문단을 인용
(3) 개선 제안: 구체적으로 몇 라인/어느 문단을 어떤 표현·묘사·방향으로 수정할지 제안 (장면/문단 단위)
(4) 기준 준수 여부 체크리스트: 합평기준규칙.md 항목을 그대로 나열하고 각 항목에 대해 준수/미흡 + 한 줄 근거
""".strip()


def _build_critique_overall_prompt(
    title: str,
    part_summaries: list[str],
    criteria: str,
    extra_instruction: str | None = None,
) -> str:
    summary_lines = "\n".join([f"- 파트 {idx}: {summary}" for idx, summary in enumerate(part_summaries, start=1)])
    extra = extra_instruction.strip() if extra_instruction else ""
    extra_block = f"\n\n[추가 지시사항]\n{extra}" if extra else ""
    return f"""
너는 단편소설 합평 전문 에디터다. 아래 구조를 반드시 지켜 출력하라.
아래 파트 요약을 바탕으로 작품 전체의 구조/전개/정서 흐름을 평가하라.

[합평 기준]
{criteria}

[입력 원고 제목]
{title}

[파트 요약]
{summary_lines}
{extra_block}

--- 출력 형식 (반드시 이 순서로) ---
(1) 한 줄 총평
(2) 항목별 점수: 각 항목 10점 만점, 근거로 파트 요약을 인용
(3) 개선 제안: 작품 전체 구조/전개/정서 흐름 중심으로 구체적 수정 방향 제안
(4) 기준 준수 여부 체크리스트: 합평기준규칙.md 항목을 그대로 나열하고 각 항목에 대해 준수/미흡 + 한 줄 근거
""".strip()


def _split_critique_chunks(text: str, max_chars: int, max_parts: int) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return [""]
    paragraphs = re.split(r"\n\s*\n", normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                if len(chunks) >= max_parts:
                    return chunks
                current = []
                current_len = 0
            start = 0
            while start < len(para):
                part = para[start:start + max_chars]
                chunks.append(part)
                if len(chunks) >= max_parts:
                    return chunks
                start += max_chars
            continue

        extra_len = len(para) + (2 if current else 0)
        if current_len + extra_len > max_chars:
            chunks.append("\n\n".join(current))
            if len(chunks) >= max_parts:
                return chunks
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += extra_len

    if current and len(chunks) < max_parts:
        chunks.append("\n\n".join(current))

    return chunks


def _extract_chunk_summary(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"^\(0\)\s*파트 요약[:：]?\s*(.+)$", text, re.M)
    if match:
        return match.group(1).strip()
    match = re.search(r"^파트 요약[:：]?\s*(.+)$", text, re.M)
    if match:
        return match.group(1).strip()
    return None


def _fallback_chunk_summary(content: str, max_chars: int = 300) -> str:
    trimmed = re.sub(r"\s+", " ", (content or "").strip())
    return trimmed[:max_chars] if trimmed else "요약 없음"


def _assemble_chunked_critique(chunk_outputs: list[str], overall_text: str) -> str:
    parts: list[str] = ["## 파트별 합평"]
    total_parts = len(chunk_outputs)
    for idx, text in enumerate(chunk_outputs, start=1):
        body = (text or "").strip()
        parts.append(f"### 파트 {idx}/{total_parts}\n{body}")
    parts.append("---\n\n## 전체 합평\n" + (overall_text or "").strip())
    return "\n\n".join(parts).strip()


@app.post("/api/critique", operation_id="api_critique")
async def api_critique(req: CritiqueRequest):
    opts = req.options or CritiqueOptions()
    # 입력 원고 저장 (항상 수행)
    work_path = _save_critique_object(req.title, req.content)

    # 기준 로드 및 LLM 합평 생성
    criteria = _load_critique_criteria()
    critique_text: str
    if opts.chunked_critique:
        max_chars = opts.chunk_max_chars or CRITIQUE_CHUNK_MAX_CHARS
        max_parts = opts.max_parts or CRITIQUE_CHUNK_MAX_PARTS
        chunks = _split_critique_chunks(req.content, max_chars=max_chars, max_parts=max_parts)
        chunk_outputs: list[str] = []
        part_summaries: list[str] = []
        total_parts = len(chunks)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_prompt = _build_critique_chunk_prompt(
                req.title,
                chunk,
                criteria,
                part_index=idx,
                total_parts=total_parts,
                extra_instruction=req.extra_instruction,
            )
            chunk_text = await call_llm(chunk_prompt)
            chunk_outputs.append(chunk_text)
            summary = _extract_chunk_summary(chunk_text) or _fallback_chunk_summary(chunk)
            part_summaries.append(summary)

        overall_prompt = _build_critique_overall_prompt(
            req.title,
            part_summaries,
            criteria,
            extra_instruction=req.extra_instruction,
        )
        overall_text = await call_llm(overall_prompt)
        critique_text = _assemble_chunked_critique(chunk_outputs, overall_text)
    else:
        prompt = _build_critique_prompt(req.title, req.content, criteria, extra_instruction=req.extra_instruction)
        critique_text = await call_llm(prompt)

    critique_path: str | None = None

    # ✅ response에 반환할 critique md(헤더 포함)를 만든다
    now_iso = datetime.now().isoformat()
    critique_yaml_header = {
        "type": "critique",
        "object_title": req.title,
        "created_at": now_iso,
        "updated_at": now_iso,
        "source_object_file": work_path,
        "usage": ["critique"],
    }
    critique_md_with_header = "---\n" + yaml.safe_dump(
        critique_yaml_header, allow_unicode=True, sort_keys=False
    ) + "---\n\n" + critique_text

    # ✅ 검증 (front-matter + 주요 필드)
    meta, _, has_fm = extract_front_matter(critique_md_with_header)
    if not has_fm:
        raise ValueError("critique output missing front-matter (server bug)")
    validate_critique_front_matter(meta)

    if opts.save_critique or (opts.save_work is True):
        # 기존 저장 로직 유지 (저장 파일도 동일한 헤더 포함)
        critique_path = _save_critique_result(req.title, critique_text, work_path)

    return standard_response(
        success=True,
        message="ok",
        data={
            "critique": critique_md_with_header,  # ✅ 이제 헤더 포함 md로 반환
            "critique_file_path": critique_path,
            "work_file_path": work_path,
        },
        error=None,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
