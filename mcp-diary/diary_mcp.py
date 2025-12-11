import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import yaml
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DIARY_ROOT = os.environ.get("DIARY_ROOT", "/data/diary")
MCP_FILESYSTEM_URL = os.environ.get("MCP_FILESYSTEM_URL", "http://mcp-filesystem:7001")

# Ollama용
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2:7b")
# LLM 백엔드 선택
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
# MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.1")

# OpenAI / 호환 서버용 (선택)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


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


def parse_front_matter(text: str) -> (Dict[str, Any], str):
    """
    YAML front matter를 찾아 (meta, body) 튜플로 반환.
    - 앞에 안내 문구가 있어도, 첫 번째 '---' 이후의 YAML 덩어리를 파싱
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

    # 1) '---' 시작 지점 탐색
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            start_idx = i
            break

    if start_idx is None:
        return {}, text

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

    if not collected:
        return {}, text

    meta_text = "\n".join(collected)
    body_start = end_idx + 1 if end_idx is not None else (start_idx + 1 + len(collected))
    body = "\n".join(lines[body_start:])

    try:
        meta = yaml.safe_load(meta_text) or {}
    except Exception:
        meta = {}

    return meta, body


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


async def call_llm(prompt: str) -> str:
    """
    공통 LLM 호출 함수
    - LLM_BACKEND=ollama  : Ollama /api/generate
    - LLM_BACKEND=openai  : OpenAI 또는 호환 서버 /v1/chat/completions
    """
    backend = LLM_BACKEND

    # 1) Ollama
    if backend == "ollama":
        payload = {
            "model": MODEL_NAME,
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

    # 2) OpenAI / 호환 서버 (예: vLLM, LM Studio)
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
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    else:
        raise RuntimeError(f"지원하지 않는 LLM_BACKEND: {backend}")

# ---------- Summarize(기존) ----------

class SummarizeRequest(BaseModel):
    keyword: Optional[str] = None
    start_date: Optional[str] = None  # "YYYY-MM-DD"
    end_date: Optional[str] = None    # "YYYY-MM-DD"
    mode: str = "summary"             # "summary" or "outline"


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

    if req.mode == "outline":
        system_instruction = """
당신은 작가의 보조자입니다.
아래 일기를 바탕으로 소설 기획안을 만드세요.

요구사항:
- 3막 구조 또는 명확한 서사 구조로 정리.
- 등장인물, 갈등, 클라이맥스, 결말을 정리.
- 표가 필요하면 Markdown 표 형식으로 작성.
"""
    else:
        system_instruction = """
당신은 심리/서사 분석가입니다.
아래 일기를 바탕으로 주요 사건, 감정 변화, 반복되는 주제를 요약하세요.
표가 필요하면 Markdown 형식으로 작성.
"""

    prompt = f"""{system_instruction}

[일기 시작]
{diaries_text}
[일기 끝]
"""

    result = await call_llm(prompt)
    return {"result": result}


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
