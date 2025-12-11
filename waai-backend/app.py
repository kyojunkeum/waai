from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.templating import Jinja2Templates

from mcp_client import (
    get_mood_stats,
    get_project_timeline,
    list_diary_files,
    summarize_diary,
)

DIARY_ROOT = os.environ.get("DIARY_ROOT", "/data/diary")
DIARY_OUTPUT_DIR = Path(DIARY_ROOT)
DIARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/data/outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 🔹 공통 LLM 설정
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# Ollama용
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2:7b")

# OpenAI / 호환 서버용 (선택)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


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


class DiaryCreateRequest(BaseModel):
    title: str
    raw_text: str
    date: str | None = None   # 없으면 오늘 날짜
    time: str | None = None   # 없으면 현재 시각


class DiaryCreateResponse(BaseModel):
    result: str   # 생성된 md 전체
    path: str     # 저장된 파일 경로


class DiaryWriteRequest(BaseModel):
    title: str
    raw_text: str


class DiaryWriteResponse(BaseModel):
    filename: str
    path: str
    markdown: str


class PlanGenerateRequest(BaseModel):
    """
    Open WebUI에서 이 API를 호출할 때 넘길 수 있는 옵션들입니다.
    아무것도 안 넘기면 '전체 일기 기반 기획서'를 만든다고 생각하면 됩니다.
    """
    topic: str | None = None              # 기획서 제목/주제 (예: "최근 일기 기반 단편소설 기획서")
    keyword: str | None = None            # 특정 키워드 기반으로 다루고 싶을 때
    start_date: str | None = None         # "2025-01-01" 이런 식
    end_date: str | None = None           # "2025-03-31"
    mode: str = "outline"                 # mcp-diary summarize 모드 (outline/summary 등)
    output_format: str = "md"             # "txt" 또는 "md"
    extra_instruction: str | None = None  # "가족/신앙 비중을 더 강조해줘" 같은 추가 지시


class PlanGenerateResponse(BaseModel):
    title: str        # 기획서 제목
    content: str      # 기획서 본문 (Open WebUI에서 바로 보여줄 내용)
    file_path: str    # /waai/data/outputs/... 저장된 경로


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


def save_output(title: str, content: str, as_markdown: bool) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "md" if as_markdown else "txt"
    safe_title = "".join(c for c in title if c.isalnum() or c in ("_", "-")) or "waai"
    filename = f"{ts}_{safe_title}.{ext}"
    path = os.path.join(OUTPUT_ROOT, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def save_diary_markdown_to_dir(md_text: str) -> str:
    """
    완성된 md 텍스트를 /data/diary 아래에 저장.
    파일명은 현재 시각 + title 기반으로 생성.
    - title 은 md의 YAML에서 한 번 파싱 시도하고, 실패하면 'diary' 사용.
    """
    lines = md_text.splitlines()
    title_value = "diary"
    for line in lines:
        if line.strip().startswith("title:"):
            raw = line.split(":", 1)[1].strip()
            title_value = raw.strip("\"' ") or "diary"
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title_value if c.isalnum() or c in ("_", "-")) or "diary"
    filename = f"{ts}_{safe_title}.md"
    path = DIARY_OUTPUT_DIR / filename
    path.write_text(md_text, encoding="utf-8")
    return str(path)


async def call_ollama_for_diary_format(prompt: str) -> str:
    # 별도 포맷 함수가 필요한 경우를 대비한 래퍼
    return await call_llm(prompt)


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
   - projects: [\"소설아이디어\", \"NGO\", \"IT\", \"가족\"] 중 일기와 가장 관련 높은 1~2개 선택
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
    diary_summary에는 mcp-diary가 만든 요약/통계 결과를 넣습니다.
    """
    extra = extra_instruction.strip() if extra_instruction else ""

    return f"""
당신은 사용자의 일기와 창작 노트를 기반으로
'단편소설 기획 전문 에디터 & 스토리 컨설턴트'입니다.

아래는 사용자가 작성한 일기·메타데이터·요약입니다.
이를 기반으로 한 편의 **단편소설 기획서**를 만들어주세요.

[요청 단편소설 기획 주제]
- {topic}

[일기 기반 창작 데이터 요약]
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
- 이 단편에 어울리는 문장 스타일
- 느린/빠른/서정적/압축적 등 문체 가이드
- 참고하면 좋은 작가 스타일(예: 마르케스, 무라카미, 김애란 등)

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


추가 참고 지시사항(있으면 반영, 없으면 무시 가능):
{extra}
""".strip()


def safe_now_date_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M")


def extract_tags_from_md(md_text: str):
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


def slugify_filename(text: str) -> str:
    text = text.strip().replace(" ", "")
    for ch in "/\\?%*:|\"<>":
        text = text.replace(ch, "-")
    return text or "note"


def make_diary_filename(date: str, md_text: str, title: str) -> str:
    tags = extract_tags_from_md(md_text)
    # 우선 tags 2개, 없으면 title 기반
    if len(tags) >= 2:
        base = f"{slugify_filename(tags[0])}-{slugify_filename(tags[1])}"
    elif len(tags) == 1:
        base = f"{slugify_filename(tags[0])}-diary"
    else:
        base = slugify_filename(title)
    return f"{date}_{base}.md"


@app.post("/api/plan/from-diaries", response_model=PlanGenerateResponse)
async def generate_plan_from_diaries(req: PlanGenerateRequest):
    """
    Open WebUI에서 '기획서 만들어줘' 라고 했을 때 호출할 핵심 API.
    내부적으로는 mcp-diary를 통해 일기 요약/통계를 받고,
    그걸 다시 LLM에 넘겨서 기획서를 만든 뒤 파일로 저장합니다.
    """
    return await generate_plan_internal(req)


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


@app.post("/api/diary/write", response_model=DiaryWriteResponse)
async def api_diary_write(body: DiaryWriteRequest):
    # 1) 날짜/시간 자동 결정
    d, t = safe_now_date_time()

    # 2) LLM에게 포맷팅 요청
    fmt_req = DiaryFormatRequest(
        date=d,
        time=t,
        title=body.title,
        raw_text=body.raw_text,
    )
    prompt = build_diary_format_prompt(fmt_req)
    md_text = await call_ollama_for_diary_format(prompt)

    # 3) 파일명 생성 후 저장 (./data/diary 에 마운트된 DIARY_ROOT 사용)
    filename = make_diary_filename(d, md_text, body.title)
    path = Path(DIARY_ROOT) / filename
    path.write_text(md_text, encoding="utf-8")

    return DiaryWriteResponse(
        filename=filename,
        path=str(path),
        markdown=md_text,
    )


@app.post("/api/diary/format", response_model=DiaryFormatResponse)
async def api_diary_format(body: DiaryFormatRequest):
    prompt = build_diary_format_prompt(body)
    md_text = await call_llm(prompt)
    return DiaryFormatResponse(result=md_text)


@app.post("/api/diary/create", response_model=DiaryCreateResponse)
async def api_diary_create(body: DiaryCreateRequest):
    """
    Open WebUI 등에서 바로 줄글 일기를 보내면:
    - /api/diary/format 과 동일한 LLM 포맷팅 로직을 사용하여 md 생성 후
    - /data/diary 아래에 파일 저장
    """
    now = datetime.now()
    date_str = body.date or now.strftime("%Y-%m-%d")
    time_str = body.time or now.strftime("%H:%M")

    req = DiaryFormatRequest(
        date=date_str,
        time=time_str,
        title=body.title,
        raw_text=body.raw_text,
    )

    prompt = build_diary_format_prompt(req)
    md_text = await call_llm(prompt)

    saved_path = save_diary_markdown_to_dir(md_text)

    return DiaryCreateResponse(result=md_text, path=saved_path)


def build_diary_repair_prompt(original_md: str) -> str:
    """
    기존에 저장된 일기 Markdown을 입력받아:
    - YAML 메타데이터를 점검/보완
    - 누락된 필드를 채우고, 이상한 값은 자연스럽게 수정
    - 본문 섹션 구조는 유지하되, 약간의 다듬기는 허용
    - '원본 텍스트' 코드블럭 내용은 절대 바꾸지 않기
    """
    return f"""
당신은 '일기 메타데이터 검수/보정 전문가'입니다.

입력: 사용자의 기존 일기 Markdown (YAML + 본문 + 원본 텍스트)

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

2. 본문 섹션:
   - # 오늘 요약 (3줄)
   - # 오늘의 사건
   - # 감정 / 생각
   - # 배운 것 / 통찰
   - # 소설 아이디어 메모 (옵션)
   - # TODO / 다음에 이어서 쓸 것 (옵션)
   이 섹션 구조는 유지하되, 내용이 너무 빈약하면 자연스럽게 조금 보완해도 됩니다.

3. "원본 텍스트 (자동 보존)" 섹션이 있다면
   - ```text ... ``` 코드 블럭 안의 내용은 절대 수정하지 마세요.
   - 없다면, 입력 Markdown에서 원본 일기 내용 부분을 찾아
     맨 아래에 "원본 텍스트 (자동 보존)" 섹션을 새로 만들어 넣으세요.

아래는 사용자가 저장한 기존 Markdown입니다.
이를 기반으로 위 규칙에 맞는 완성된 Markdown 전체를 다시 출력하세요.

----- 기존 Markdown 시작 -----
{original_md}
----- 기존 Markdown 끝 -----
""".strip()


@app.post("/api/diary/reformat-md", response_model=DiaryReformatResponse)
async def api_diary_reformat_md(body: DiaryReformatRequest):
    """
    기존에 저장된 md(일기)를 입력받아
    - YAML 메타데이터 보정
    - 누락 필드 채우기
    - summary, projects, tags 등을 재추론
    """
    prompt = build_diary_repair_prompt(body.markdown)
    new_md = await call_llm(prompt)
    return DiaryReformatResponse(result=new_md)


# =========================
# 🌐 Web UI 라우트
# =========================


async def generate_plan_internal(req: PlanGenerateRequest) -> PlanGenerateResponse:
    """
    HTML 폼과 Open WebUI 양쪽에서 재사용하는 기획서 생성 로직.
    """
    topic = req.topic or "최근 일기 기반 단편소설 기획서"

    diary_summary = await summarize_diary(
        keyword=req.keyword or None,
        start_date=req.start_date or None,
        end_date=req.end_date or None,
        mode=req.mode or "outline",
    )

    prompt = build_plan_prompt(
        topic=topic,
        diary_summary=diary_summary,
        extra_instruction=req.extra_instruction,
    )

    plan_text = await call_llm(prompt)
    as_md = (req.output_format == "md")
    file_path = save_output(topic, plan_text, as_markdown=as_md)

    return PlanGenerateResponse(
        title=topic,
        content=plan_text,
        file_path=file_path,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    diary_files = await list_diary_files()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "diary_files": diary_files,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_from_form(
    request: Request,
    title: str = Form(...),
    keyword: str | None = Form(None),
    start_date: str | None = Form(None),
    end_date: str | None = Form(None),
    mode: str = Form("outline"),
    output_format: str = Form("txt"),
    extra_instruction: str | None = Form(None),
):
    req = PlanGenerateRequest(
        topic=title,
        keyword=keyword,
        start_date=start_date or None,
        end_date=end_date or None,
        mode=mode or "outline",
        output_format=output_format or "txt",
        extra_instruction=extra_instruction,
    )

    result = await generate_plan_internal(req)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "title": result.title,
            "result": result.content,
            "file_path": result.file_path,
        },
    )


@app.post("/diary/create-form")
async def diary_create_form(
    title: str = Form(...),
    raw_text: str = Form(...),
    date: str | None = Form(None),
    time: str | None = Form(None),
):
    """
    대시보드/HTML 폼에서 호출하는 일기 생성 라우트.
    """
    req = DiaryCreateRequest(
        title=title,
        raw_text=raw_text,
        date=date or None,
        time=time or None,
    )
    result = await api_diary_create(req)
    # 생성 후 대시보드로 이동
    return RedirectResponse(
        url="/dashboard?created=1&path=" + result.path,
        status_code=303,
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
):
    mood_stats = await get_mood_stats(start_date=start_date, end_date=end_date)
    projects = ["소설아이디어", "NGO", "IT", "가족"]
    project_data = await get_project_timeline(projects, start_date=start_date, end_date=end_date)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": os.environ.get("DASHBOARD_USER", "guest"),
            "mood_stats": mood_stats,
            "projects": projects,
            "project_data": project_data,
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


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
