from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, validator

KST = timezone(timedelta(hours=9))


def _to_aware_datetime(value: Any) -> datetime:
    """
    Convert incoming datetime-like values to timezone-aware datetime in KST.
    - Accepts ISO8601 string, date, or datetime.
    - Naive datetime is assumed to be local KST.
    """
    if value is None:
        return datetime.now(tz=KST)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=KST)
        return value.astimezone(KST)

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=KST)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=KST)
            return parsed.astimezone(KST)
        except Exception:
            return datetime.now(tz=KST)

    return datetime.now(tz=KST)


class TimeRange(BaseModel):
    start: date
    end: date
    model_config = ConfigDict(
        json_schema_extra={"example": {"start": "2025-01-01", "end": "2025-01-31"}}
    )

    @validator("end")
    def validate_range(cls, v: date, values: dict[str, Any]) -> date:
        start = values.get("start")
        if start and v < start:
            raise ValueError("end date must be on or after start date")
        return v


class QueryContext(BaseModel):
    topic: Optional[str] = None
    keywords: List[str] = Field(default_factory=list, description="우선순위가 높은 검색 키워드 목록")
    time_range: Optional[TimeRange] = None
    tone: Optional[str] = Field(default=None, description="원하는 톤/스타일")
    intent: Optional[str] = Field(default=None, description="검색 의도 (예: planning, critique)")
    doc_types: List[str] = Field(default_factory=list, description="검색 대상 문서 타입 리스트")
    people: List[str] = Field(default_factory=list, description="관계/인물 필터")
    must_cite: bool = Field(default=False, description="citations 필수 여부")
    k: int = Field(default=5, ge=1, le=100, description="최대 반환 chunk 수")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "12월 가족 이야기",
                "keywords": ["가족", "연말", "감정"],
                "time_range": {"start": "2025-12-01", "end": "2025-12-31"},
                "tone": "따뜻하고 담백한",
                "intent": "planning",
                "doc_types": ["diary", "ideas", "web_research"],
                "people": ["아내", "아이"],
                "must_cite": True,
                "k": 8,
            }
        }
    )

    @validator("keywords", "doc_types", "people", pre=True, each_item=False)
    def normalize_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(item) for item in v if str(item).strip()]


class ChunkSource(BaseModel):
    url: Optional[str] = None
    publisher: Optional[str] = None


class ContextChunk(BaseModel):
    chunk_id: str = Field(..., description="고유 chunk 식별자")
    doc_type: str = Field(..., description="문서 타입 (diary/ideas/web_research/works/bible 등)")
    doc_id: str = Field(..., description="원본 문서 식별자 또는 경로")
    title: str = Field(..., description="원본 문서 또는 chunk 제목")
    source: Optional[ChunkSource] = Field(default=None, description="원문 출처 정보")
    created_at: datetime = Field(..., description="chunk 생성 시각 (tz-aware)")
    excerpt: str = Field(..., description="원문 발췌 텍스트")
    hash: Optional[str] = Field(default=None, description="원본 chunk 해시")

    @validator("created_at", pre=True)
    def ensure_tzaware(cls, v: Any) -> datetime:
        return _to_aware_datetime(v)


class RetrievalStats(BaseModel):
    total_candidates: int = Field(..., ge=0, description="검토한 전체 후보 chunk 수")
    selected_chunks: int = Field(..., ge=0, description="선택된 chunk 수")
    filters: dict[str, Any] = Field(default_factory=dict, description="적용된 필터 목록")
    latency_ms: float = Field(..., ge=0, description="검색 소요 시간(ms)")
    citations_path: Optional[str] = Field(default=None, description="생성된 citations 파일 경로")


class ContextPack(BaseModel):
    context_pack: List[ContextChunk] = Field(default_factory=list)
    retrieval_stats: RetrievalStats
    trace_id: str


class RagIndexRequest(BaseModel):
    paths: List[str] = Field(..., description="인덱싱할 파일 절대경로 또는 상대경로 목록")
    doc_type: str = Field(..., description="문서 타입")
    force: bool = Field(default=False, description="강제 재인덱싱 여부")


class RagIndexAllRequest(BaseModel):
    doc_types: List[str] = Field(..., description="인덱싱 대상 문서 타입 리스트")
    force: bool = Field(default=False, description="강제 재인덱싱 여부")


class RagSearchRequest(BaseModel):
    query_text: str = Field(..., description="사용자 자연어 질의")
    query_context: QueryContext


class ErrorPayload(BaseModel):
    code: str
    message: str
    detail: Optional[Any] = None


class ErrorResponse(BaseModel):
    error: ErrorPayload
    trace_id: str
