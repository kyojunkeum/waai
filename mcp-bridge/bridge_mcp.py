import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import yaml
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

def _resolve_data_path(env_var: str, default_subpath: str) -> Path:
    candidates: list[Path] = []

    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))

    candidates.append(Path("/data") / default_subpath)
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "data" / default_subpath)

    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


DIARY_ROOT = str(_resolve_data_path("DIARY_ROOT", "diary"))
IDEAS_ROOT = str(_resolve_data_path("IDEAS_ROOT", "ideas"))
WEB_RESEARCH_ROOT = str(_resolve_data_path("WEB_RESEARCH_ROOT", "web_research"))
WORKS_ROOT = str(_resolve_data_path("WORKS_ROOT", "works"))
BIBLE_ROOT = str(_resolve_data_path("BIBLE_ROOT", "bible"))
MCP_FILESYSTEM_URL = os.environ.get("MCP_FILESYSTEM_URL", "http://mcp-filesystem:7001")
DEFAULT_LIMIT_PER_TYPE = int(os.environ.get("LIMIT_PER_TYPE", "10"))
TYPE_LIMIT_OVERRIDES = {
    "diary": int(os.environ.get("LIMIT_DIARY", str(DEFAULT_LIMIT_PER_TYPE))),
    "ideas": int(os.environ.get("LIMIT_IDEAS", str(DEFAULT_LIMIT_PER_TYPE))),
    "web_research": int(os.environ.get("LIMIT_WEB_RESEARCH", str(DEFAULT_LIMIT_PER_TYPE))),
    "works": int(os.environ.get("LIMIT_WORKS", str(DEFAULT_LIMIT_PER_TYPE))),
    "bible": int(os.environ.get("LIMIT_BIBLE", str(DEFAULT_LIMIT_PER_TYPE))),
}

# Ollama용 (direct)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다.")
    return value

OLLAMA_REFINE_MODEL = _require_env("OLLAMA_REFINE_MODEL")


app = FastAPI()


# ---------- 공통 유틸 ----------

class DiaryEntry(BaseModel):
    path: str              # 절대 경로
    rel_path: str          # DIARY_ROOT 기준 상대 경로
    date: Optional[date]
    time: Optional[str]
    title: Optional[str]
    mood: Optional[str]
    mood_score: Optional[float]
    tags: List[str] = []
    people: List[str] = []
    location: Optional[str]
    type: Optional[str]
    projects: List[str] = []
    scene_potential: Optional[bool]
    body: str               # 본문 전체


async def list_diary_files(client: Optional[httpx.AsyncClient] = None) -> List[str]:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=30.0)

    try:
        resp = await client.get("/files")
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files", [])
        filtered = [
            f for f in files
            if isinstance(f, str) and f.lower().endswith((".txt", ".md"))
        ]
        return sorted(filtered)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mcp-filesystem /files 호출 실패: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()


async def list_all_files(client: Optional[httpx.AsyncClient] = None) -> Dict[str, List[str]]:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=30.0)
    try:
        resp = await client.get("/files-all")
        resp.raise_for_status()
        data = resp.json()
        return {
            "diary": data.get("diary", []),
            "ideas": data.get("ideas", []),
            "web_research": data.get("web_research", []),
            "works": data.get("works", []),
            "bible": data.get("bible", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mcp-filesystem /files-all 호출 실패: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()


async def read_diary_file(rel_path: str, client: Optional[httpx.AsyncClient] = None) -> str:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=30.0)

    try:
        resp = await client.get("/file", params={"name": rel_path})
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mcp-filesystem /file 호출 실패: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()


async def read_file_by_type(file_type: str, rel_path: str, client: Optional[httpx.AsyncClient] = None) -> str:
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=30.0)
    try:
        resp = await client.get("/file-by-type", params={"type": file_type, "name": rel_path})
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mcp-filesystem /file-by-type 호출 실패: {exc}") from exc
    finally:
        if own_client:
            await client.aclose()


def parse_front_matter(text: str) -> (Dict[str, Any], str):
    """
    YAML front matter를 찾아 (meta, body) 튜플로 반환.
    - 앞에 안내 문구가 있어도, '---' 이후의 YAML 덩어리를 파싱
    - 닫는 '---'가 없거나 중간에 본문이 섞인 LLM 출력도 최대한 견고하게 처리
    """
    lines = text.splitlines()

    def is_yamlish(line: str) -> bool:
        s = line.strip()
        if not s:
            return True
        if s.startswith("#"):
            return True
        if s.startswith("-"):
            return True
        # key: value 형태 감지
        first = s.split()[0]
        return ":" in first

    search_from = 0
    fallback_body = text
    while search_from < len(lines):
        # 1) '---' 시작 지점 탐색
        start_idx = None
        for i in range(search_from, len(lines)):
            if lines[i].strip() == "---":
                start_idx = i
                break

        if start_idx is None:
            break

        # 2) YAML 덩어리 수집: '---' 다음 줄부터, YAML 형태가 아닌 줄을 만나면 종료
        collected: list[str] = []
        end_idx = None
        for i in range(start_idx + 1, len(lines)):
            s = lines[i].strip()
            if s == "---":
                end_idx = i
                break
            if collected and not is_yamlish(lines[i]):
                end_idx = i
                break
            if not collected and not is_yamlish(lines[i]):
                # 아직 수집 시작 전인 경우(예: 빈줄/코멘트)라도, YAML 형태 아니면 넘어감
                continue
            collected.append(lines[i])

        # 빈 프론트매터(--- 바로 다음 ---)는 건너뛰고 다음 블록 탐색
        if not collected:
            search_from = (end_idx or start_idx + 1) + 1
            continue

        meta_text = "\n".join(collected)
        body_start = end_idx + 1 if end_idx is not None else (start_idx + 1 + len(collected))
        body = "\n".join(lines[body_start:])

        try:
            meta = yaml.safe_load(meta_text) or {}
        except Exception:
            meta = {}

        if meta:
            return meta, body

        fallback_body = body
        # 메타가 비어 있으면 이후 블록을 계속 탐색
        search_from = body_start + 1

    return {}, fallback_body


def parse_date_from_filename(path: str) -> Optional[date]:
    """
    파일명 앞의 YYYY-MM-DD 를 파싱해 date 로 반환.
    실패하면 None.
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    # 1) ISO 포맷 시도 (앞 10자 또는 첫 토큰)
    candidates = [name[:10], name.split("_")[0]]
    for cand in candidates:
        try:
            return datetime.fromisoformat(cand).date()
        except Exception:
            pass

    # 2) YYYYMMDD 숫자 8자리
    digits_prefix = "".join(ch for ch in name if ch.isdigit())
    if len(digits_prefix) >= 8:
        try:
            d = datetime.strptime(digits_prefix[:8], "%Y%m%d").date()
            # 비현실적인 연도(2100 이후)는 YYMMDD가 붙은 케이스일 수 있으니 무시
            if d.year <= 2100:
                return d
        except Exception:
            pass

    # 3) YYMMDD 숫자 6자리 (예: 250809 -> 2025-08-09)
    if len(digits_prefix) >= 6:
        try:
            return datetime.strptime(digits_prefix[:6], "%y%m%d").date()
        except Exception:
            pass

    return None


def _parse_date_field(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except Exception:
            return None
    return None


def parse_diary_file(rel_path: str, content: str) -> DiaryEntry:
    meta, body = parse_front_matter(content)
    d = None

    # 1순위: meta.date
    if isinstance(meta.get("date"), (datetime, date)):
        d = meta["date"]
    elif isinstance(meta.get("date"), str):
        try:
            d = datetime.fromisoformat(meta["date"]).date()
        except Exception:
            d = None

    # 2순위: 파일명에서 날짜 파싱
    if d is None:
        d = parse_date_from_filename(rel_path)

    def as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x]
        return [str(x)]

    # mood_score 숫자 파싱
    raw_score = meta.get("mood_score")
    mood_score_val: Optional[float] = None
    if isinstance(raw_score, (int, float)):
        mood_score_val = float(raw_score)
    elif isinstance(raw_score, str):
        try:
            mood_score_val = float(raw_score)
        except Exception:
            mood_score_val = None

    # scene_potential bool 파싱
    sp = meta.get("scene_potential")
    if isinstance(sp, bool):
        scene_potential_val = sp
    elif isinstance(sp, str):
        scene_potential_val = sp.lower() in ("true", "1", "yes", "y")
    else:
        scene_potential_val = None

    abs_path = os.path.join(DIARY_ROOT, rel_path)

    return DiaryEntry(
        path=abs_path,
        rel_path=rel_path,
        date=d,
        time=str(meta.get("time")) if meta.get("time") else None,
        title=str(meta.get("title")) if meta.get("title") else None,
        mood=str(meta.get("mood")) if meta.get("mood") else None,
        mood_score=mood_score_val,
        tags=as_list(meta.get("tags")),
        people=as_list(meta.get("people")),
        location=str(meta.get("location")) if meta.get("location") else None,
        type=str(meta.get("type")) if meta.get("type") else None,
        projects=as_list(meta.get("projects")),
        scene_potential=scene_potential_val,
        body=body,
    )


async def load_diary_entries() -> List[DiaryEntry]:
    async with httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=60.0) as client:
        rel_paths = await list_diary_files(client)
        entries: List[DiaryEntry] = []
        for rel_path in rel_paths:
            content = await read_diary_file(rel_path, client)
            entries.append(parse_diary_file(rel_path, content))
        return entries


# =======================
# 멀티 소스 로딩 (V3.1)
# =======================

class GenericEntry(BaseModel):
    type: str
    path: str
    rel_path: str
    title: Optional[str]
    tags: List[str] = []
    topics: List[str] = []
    summary: Optional[str]
    body: str
    meta: Dict[str, Any] = Field(default_factory=dict)


def _root_for_type(file_type: str) -> Optional[str]:
    roots = {
        "diary": DIARY_ROOT,
        "ideas": IDEAS_ROOT,
        "web_research": WEB_RESEARCH_ROOT,
        "works": WORKS_ROOT,
        "bible": BIBLE_ROOT,
    }
    return roots.get(file_type)


def _parse_generic_entry(file_type: str, rel_path: str, content: str) -> GenericEntry:
    meta, body = parse_front_matter(content)
    root = _root_for_type(file_type) or ""
    abs_path = os.path.join(root, rel_path)
    return GenericEntry(
        type=file_type,
        path=abs_path,
        rel_path=rel_path,
        title=meta.get("title") or os.path.splitext(os.path.basename(rel_path))[0],
        tags=[str(t) for t in meta.get("tags", [])] if isinstance(meta.get("tags"), list) else [],
        topics=[str(t) for t in meta.get("topics", [])] if isinstance(meta.get("topics"), list) else [],
        summary=meta.get("summary"),
        body=body,
        meta=meta,
    )


def _resolve_limit_for_type(file_type: str, requested: Optional[int]) -> int:
    override = TYPE_LIMIT_OVERRIDES.get(file_type, DEFAULT_LIMIT_PER_TYPE)
    if requested is None:
        return override
    return min(requested, override) if override else requested


def _entry_matches_filter(meta: Dict[str, Any], body: str,
                          rel_path: str,
                          start_date: Optional[str],
                          end_date: Optional[str],
                          keyword: Optional[str]) -> bool:
    def parse_dt(v: Optional[str]):
        try:
            return datetime.fromisoformat(v).date() if v else None
        except Exception:
            return None

    d = _parse_date_field(meta.get("date")) \
        or _parse_date_field(meta.get("created_at")) \
        or _parse_date_field(meta.get("updated_at")) \
        or parse_date_from_filename(rel_path)

    if start_date:
        s = parse_dt(start_date)
        if s and d and d < s:
            return False
    if end_date:
        e = parse_dt(end_date)
        if e and d and d > e:
            return False

    if keyword:
        def as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i) for i in x]
            return [str(x)]

        in_meta = any(
            keyword in item for item in [
                meta.get("title") or "",
                *as_list(meta.get("tags")),
                *as_list(meta.get("topics")),
                *as_list(meta.get("people")),
                *as_list(meta.get("locations")),
            ]
        )
        if (not in_meta) and (keyword not in body):
            return False
    return True


async def load_entries_by_type(file_type: str, limit: Optional[int] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               keyword: Optional[str] = None) -> List[GenericEntry]:
    """
    type별 md 파일 로딩. V3.1: 모든 타입에 start_date/end_date/keyword 필터 적용.
    limit 인자가 None이면 타입별 환경변수 제한을 사용한다.
    """
    root = _root_for_type(file_type)
    if not root:
        return []

    async with httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=60.0) as client:
        # 새 /files-all 사용
        all_files = await list_all_files(client)
        rel_paths = all_files.get(file_type, [])

        max_items = _resolve_limit_for_type(file_type, limit)
        entries: List[GenericEntry] = []
        for rel_path in rel_paths:
            if len(entries) >= max_items:
                break
            content = await read_file_by_type(file_type, rel_path, client)
            meta, body = parse_front_matter(content)
            if not _entry_matches_filter(meta, body, rel_path, start_date, end_date, keyword):
                continue
            entry = _parse_generic_entry(file_type, rel_path, content)
            entries.append(entry)
        return entries


async def call_llm(prompt: str) -> str:
    """
    공통 LLM 호출 함수
    - Ollama /api/generate (direct)
    """
    payload = {
        "model": OLLAMA_REFINE_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")
    except httpx.ReadTimeout:
        return "LLM 호출이 시간 초과되었습니다. 다시 시도해 주세요."

# ---------- Summarize(기존) ----------

class SummarizeRequest(BaseModel):
    keyword: Optional[str] = None
    start_date: Optional[str] = None  # "YYYY-MM-DD"
    end_date: Optional[str] = None    # "YYYY-MM-DD"
    mode: str = "summary"             # "summary" or "outline"
    topic: Optional[str] = None       # 기획서 초점
    extra_instruction: Optional[str] = None  # 추가 지시


def filter_by_date(entries: List[DiaryEntry],
                   start_date: Optional[str],
                   end_date: Optional[str]) -> List[DiaryEntry]:
    if not start_date and not end_date:
        return entries

    s = date.min
    e = date.max

    def parse_dt(v: str):
        try:
            return datetime.fromisoformat(v).date()
        except Exception:
            return None

    if start_date:
        parsed = parse_dt(start_date)
        if parsed:
            s = parsed
    if end_date:
        parsed = parse_dt(end_date)
        if parsed:
            e = parsed

    result = []
    for entry in entries:
        if entry.date is None:
            continue
        if s <= entry.date <= e:
            result.append(entry)
    return result


def build_structured_summary_prompt(diaries_text: str,
                                    mode: str = "summary",
                                    topic: Optional[str] = None,
                                    extra_instruction: Optional[str] = None) -> str:
    focus = topic.strip() if topic else "전체 일기"
    extra = extra_instruction.strip() if extra_instruction else ""

    topic_block = f"- 기획서 주제 우선순위: {focus}"
    extra_block = f"- 추가 지시 반영: {extra}" if extra else ""

    return f"""
너는 일기 데이터 기반 '소설 기획서용 분석 요약'을 만드는 분석가다.
반드시 아래 섹션 헤더와 순서를 정확히 지켜 출력하며, 앞뒤 설명/인사말을 덧붙이지 마라.
원문에 없는 사실/이름/사건을 창작하지 말고, 근거 없는 일반론(희망/감동/성장 등)도 금지한다.

요약 시 고려:
- 핵심 초점: {focus}
{extra_block}
- 애매한 표현은 피하고 날짜/키워드를 명시적으로 적어라.
- 동일한 섹션 헤더를 정확히 사용하라.

[기간 요약]
- 선택된 기간/필터에 해당하는 일기의 전반 흐름을 2~3문장으로 압축
- {focus}와 연관된 흐름을 우선 서술

[반복 키워드 TOP 10]
- 형식: "- 키워드: N회"
- 빈도 높은 키워드 상위 10개 (적더라도 가능한 만큼)

[주요 장면 후보]
- 최소 5개, 최대 10개
- 형식: "- YYYY-MM-DD: 장면 한 줄 요약 (장면성 있는 사건만)"
- {focus} 또는 추가 지시와 연관된 장면을 우선 배치

[인물/관계]
- 반복 등장 인물/관계와 그 성격을 나열
- 형식 예: "- 인물/관계: 성격/역할 (근거 포함)"

[감정 변화 흐름]
- 초반/중반/후반 감정 흐름을 각 1~2문장
- 근거를 날짜/사건과 함께 언급

[원문 표현(발췌)]
- 최대 5개, 최소 1개
- 형식: "- YYYY-MM-DD: \"원문 표현\""
- 원문에 존재하는 문장/구절만 발췌, 단어 추가/창작 금지

[기획서에 바로 쓸 근거 목록(JSON)]
- 바로 아래 줄부터는 코드블록 없이 순수 JSON 배열만 출력한다.
- 가능하면 10~20개 사이의 근거를 담아라.
- 각 아이템에는 date, title, excerpt, reason 키가 모두 있어야 하며 type 기본값은 "diary"다.
- excerpt는 원문에 실제로 존재하는 표현만 사용하고 창작을 금지한다.
- JSON 배열 외의 불릿/설명/마크다운은 절대 추가하지 마라.

아래는 선택된 일기 데이터다. 반드시 위 형식을 따르라.

[일기 시작]
{diaries_text}
[일기 끝]
""".strip()


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    entries = await load_diary_entries()
    entries = filter_by_date(entries, req.start_date, req.end_date)

    # 키워드 필터: body 또는 title/tags에 포함
    if req.keyword:
        key = req.keyword
        filtered = []
        for e in entries:
            if (e.body and key in e.body) or \
               (e.title and key in e.title) or \
               any(key in t for t in e.tags):
                filtered.append(e)
        entries = filtered

    if not entries:
        return {"result": "선택된 조건에 해당하는 일기가 없습니다."}

    # 일기 텍스트 합치기
    chunks = []
    for e in entries:
        header = f"# {e.date or ''} {e.title or e.rel_path}"
        chunks.append(f"{header}\n{e.body}\n")

    diaries_text = "\n\n".join(chunks)

    prompt = build_structured_summary_prompt(
        diaries_text=diaries_text,
        mode=req.mode,
        topic=req.topic,
        extra_instruction=req.extra_instruction,
    )

    result = await call_llm(prompt)
    return {"result": result}


# ---------- /select-and-summarize : 파일 선택 + 요약 (mcp-filesystem 활용) ----------

class FilteredEntry(BaseModel):
    type: str
    rel_path: str
    title: Optional[str]
    meta: Dict[str, Any] = Field(default_factory=dict)
    excerpt: str = ""


class SelectAndSummarizeRequest(BaseModel):
    include: List[str] = Field(default_factory=lambda: ["diary", "ideas", "web_research", "works", "bible"])
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    keyword: Optional[str] = None
    limit_per_type: int = 10
    preview_chars: int = 1200
    extra_instruction: Optional[str] = None


def _to_filtered_entry(entry: GenericEntry, preview_chars: int) -> FilteredEntry:
    body = entry.body.strip() if entry.body else ""
    preview = body[:preview_chars]
    if len(body) > preview_chars:
        preview += "\n...[본문 일부만 포함]"
    return FilteredEntry(
        type=entry.type,
        rel_path=entry.rel_path,
        title=entry.title,
        meta=entry.meta,
        excerpt=preview,
    )


async def fetch_filtered_entries(req: SelectAndSummarizeRequest) -> Dict[str, List[FilteredEntry]]:
    payload = {
        "include": req.include,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "keyword": req.keyword,
        "limit_per_type": req.limit_per_type,
        "preview_chars": req.preview_chars,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{MCP_FILESYSTEM_URL}/filter", json=payload)
        r.raise_for_status()
        data = r.json().get("data", {})

    normalized: Dict[str, List[FilteredEntry]] = {}
    for k, items in data.items():
        bucket: List[FilteredEntry] = []
        for item in items:
            try:
                bucket.append(FilteredEntry(**item))
            except Exception:
                continue
        normalized[k] = bucket
    return normalized


async def _fallback_load_entries(req: SelectAndSummarizeRequest,
                                 current: Dict[str, List[FilteredEntry]]) -> Dict[str, List[FilteredEntry]]:
    """
    필터 결과가 빈 타입에 대해 keyword 제약을 풀고 load_entries_by_type로 채워넣는다.
    - 날짜 제한은 유지, keyword는 해제하여 자료가 없을 때 다른 소스를라도 포함.
    """
    filled = dict(current)
    for t in req.include:
        existing = filled.get(t, [])
        if existing:
            continue
        fallback = await load_entries_by_type(
            file_type=t,
            limit=req.limit_per_type,
            start_date=req.start_date,
            end_date=req.end_date,
            keyword=None,  # keyword 해제하여 자료 확보
        )
        filled[t] = [_to_filtered_entry(e, req.preview_chars) for e in fallback]
    return filled


def build_filtered_prompt(data: Dict[str, List[FilteredEntry]], extra_instruction: Optional[str]) -> str:
    def render(label: str, entries: List[FilteredEntry]) -> str:
        if not entries:
            return f"[{label}]\n- (데이터 없음)\n"
        lines = [f"[{label}]"]
        for e in entries:
            meta = e.meta or {}
            tags = meta.get("tags") or []
            topics = meta.get("topics") or []
            meta_info = []
            if tags:
                meta_info.append(f"tags={','.join(tags)}")
            if topics:
                meta_info.append(f"topics={','.join(topics)}")
            meta_str = f" ({'; '.join(meta_info)})" if meta_info else ""
            title = e.title or e.rel_path
            lines.append(f"## {title} [{e.rel_path}] {meta_str}".strip())
            summary_line = meta.get("summary")
            if summary_line:
                lines.append(f"- summary: {summary_line}")
            if e.excerpt:
                lines.append(e.excerpt.strip())
            lines.append("")
        return "\n".join(lines)

    extra = (extra_instruction or "").strip()
    parts = [
        "너는 mcp-filesystem이 추려준 다중 소스(md) 데이터를 기반으로 기획서용 분석 요약을 만든다.",
        render("DIARY", data.get("diary", [])),
        render("IDEAS", data.get("ideas", [])),
        render("WEB_RESEARCH", data.get("web_research", [])),
        render("WORKS", data.get("works", [])),
        render("BIBLE", data.get("bible", [])),
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
    ]
    if extra:
        parts.append(f"- 추가 지시: {extra}")
    return "\n".join(parts)


@app.post("/select-and-summarize")
async def select_and_summarize(req: SelectAndSummarizeRequest):
    includes = [t for t in req.include if t in ("diary", "ideas", "web_research", "works", "bible")]
    filtered = await fetch_filtered_entries(req.copy(update={"include": includes}))
    filtered = await _fallback_load_entries(req.copy(update={"include": includes}), filtered)

    prompt = build_filtered_prompt(filtered, req.extra_instruction)
    result = await call_llm(prompt)
    sources = {k: [e.rel_path for e in v] for k, v in filtered.items()}

    return {
        "result": result,
        "sources": sources,
        "entries": {k: [e.dict() for e in v] for k, v in filtered.items()},
    }


# ---------- /search : 조건 검색 ----------

class SearchRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    mood_min: Optional[float] = None
    mood_max: Optional[float] = None
    scene_potential: Optional[bool] = None


@app.post("/search")
async def search_entries(req: SearchRequest):
    entries = await load_diary_entries()
    entries = filter_by_date(entries, req.start_date, req.end_date)

    # 태그 필터 (AND or OR? 여기서는 OR로 구현: 교집합 있으면 통과)
    if req.tags:
        tags_set = set(req.tags)
        entries = [
            e for e in entries
            if tags_set.intersection(set(e.tags))
        ]

    # projects 필터
    if req.projects:
        proj_set = set(req.projects)
        entries = [
            e for e in entries
            if proj_set.intersection(set(e.projects))
        ]

    # mood_score 범위
    if req.mood_min is not None or req.mood_max is not None:
        mn = req.mood_min if req.mood_min is not None else float("-inf")
        mx = req.mood_max if req.mood_max is not None else float("inf")
        entries = [
            e for e in entries
            if e.mood_score is not None and mn <= e.mood_score <= mx
        ]

    # scene_potential 필터
    if req.scene_potential is not None:
        entries = [
            e for e in entries
            if e.scene_potential == req.scene_potential
        ]

    # 응답은 메타만 (body는 선택적)
    result = [
        {
            "rel_path": e.rel_path,
            "date": e.date.isoformat() if e.date else None,
            "title": e.title,
            "mood": e.mood,
            "mood_score": e.mood_score,
            "tags": e.tags,
            "projects": e.projects,
            "scene_potential": e.scene_potential,
        }
        for e in entries
    ]
    return {"count": len(result), "entries": result}


# ---------- /mood-stats : 대시보드용 감정 그래프 데이터 ----------

class MoodStatsRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.post("/mood-stats")
async def mood_stats(req: MoodStatsRequest):
    entries = await load_diary_entries()
    entries = filter_by_date(entries, req.start_date, req.end_date)

    # 날짜별로 mood_score 수집
    bucket: Dict[date, List[float]] = {}
    for e in entries:
        if e.date is None or e.mood_score is None:
            continue
        bucket.setdefault(e.date, []).append(e.mood_score)

    stats = []
    for d in sorted(bucket.keys()):
        scores = bucket[d]
        avg = sum(scores) / len(scores)
        stats.append({
            "date": d.isoformat(),
            "avg_mood_score": avg,
            "min_mood_score": min(scores),
            "max_mood_score": max(scores),
            "count": len(scores),
        })

    return {"stats": stats}


# ---------- /project-timeline : 프로젝트 타임라인 ----------

class ProjectTimelineRequest(BaseModel):
    projects: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@app.post("/project-timeline")
async def project_timeline(req: ProjectTimelineRequest):
    if not req.projects:
        return {"error": "projects 필드는 최소 1개 이상 필요합니다."}

    proj_set = set(req.projects)

    entries = await load_diary_entries()
    entries = filter_by_date(entries, req.start_date, req.end_date)

    # 프로젝트 이름별로 정리
    timeline: Dict[str, List[Dict[str, Any]]] = {p: [] for p in proj_set}

    for e in entries:
        inter = proj_set.intersection(set(e.projects))
        if not inter:
            continue
        for p in inter:
            timeline[p].append({
                "date": e.date.isoformat() if e.date else None,
                "title": e.title,
                "rel_path": e.rel_path,
                "mood": e.mood,
                "mood_score": e.mood_score,
                "tags": e.tags,
            })

    # 날짜 순으로 정렬
    for p in timeline:
        timeline[p].sort(key=lambda x: x["date"] or "")

    return {"projects": timeline}

@app.get("/health")
async def health():
    fs_status = "unknown"
    try:
        async with httpx.AsyncClient(base_url=MCP_FILESYSTEM_URL, timeout=5.0) as client:
            resp = await client.get("/health")
            resp.raise_for_status()
            fs_status = resp.json().get("status", "ok")
    except Exception as exc:
        fs_status = f"error:{exc}"

    ok = fs_status == "ok"
    return {
        "status": "ok" if ok else "mcp_filesystem_error",
        "diary_root": DIARY_ROOT,
        "filesystem_status": fs_status,
    }
