"""Retriever utilities."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

from app.rag.chroma_client import query as chroma_query
from app.rag.embedding import OllamaEmbeddingProvider
from app.rag.citations import write_citations
from app.rag.models import ContextChunk, ContextPack, QueryContext, RetrievalStats
from app.util.trace import ensure_trace_id, log_event

_embedding_provider = OllamaEmbeddingProvider()

INTENT_DOCTYPE_BOOST = {
    "planning": {"ideas": 0.15, "web_research": 0.15},
    "research": {"web_research": 0.2},
    "critique": {"critique": 0.2, "critique_criteria": 0.3},
}


def _to_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    try:
        dt = datetime.fromisoformat(str(val))
        if dt.tzinfo is None:
            return dt
        return dt
    except Exception:
        return None


def _recency_score(created_at: str | None, intent: str | None) -> float:
    dt = _to_datetime(created_at)
    if not dt:
        return 0.0
    days = (datetime.now(dt.tzinfo) - dt).days
    if days < 0:
        days = 0
    # 최근일수록 가점, intent별 가중치 조정 가능
    weight = 1.0 if intent in {"planning", "research"} else 0.5
    score = max(0.0, 1.0 - (days / 365.0))
    return score * weight


def _keyword_hit_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return min(1.0, hits / max(1, len(keywords)))


def _doctype_boost(doc_type: str, intent: str | None) -> float:
    if not intent:
        return 0.0
    return INTENT_DOCTYPE_BOOST.get(intent, {}).get(doc_type, 0.0)


def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 - distance


def _apply_filters(candidates: list[dict[str, Any]], ctx: QueryContext) -> list[dict[str, Any]]:
    doc_types = ctx.doc_types or []
    start = ctx.time_range.start if ctx.time_range else None
    end = ctx.time_range.end if ctx.time_range else None

    def date_ok(meta: dict[str, Any]) -> bool:
        if not (start or end):
            return True
        dt = _to_datetime(meta.get("created_at"))
        if dt is None:
            return True  # 메타 없는 경우 후단계에서 가드
        d = dt.date()
        if start and d < start:
            return False
        if end and d > end:
            return False
        return True

    filtered = []
    for item in candidates:
        meta = item.get("meta") or {}
        if doc_types and meta.get("doc_type") not in doc_types:
            continue
        if not date_ok(meta):
            continue
        filtered.append(item)
    return filtered


def _rerank(candidates: list[dict[str, Any]], ctx: QueryContext) -> list[dict[str, Any]]:
    keywords = ctx.keywords or []
    intent = ctx.intent or ""
    for item in candidates:
        meta = item.get("meta") or {}
        text = (item.get("document") or "") + " " + (meta.get("title") or "")
        sim_score = _distance_to_similarity(item.get("distance"))
        kw_score = _keyword_hit_score(text, keywords)
        recency = _recency_score(meta.get("created_at"), intent)
        boost = _doctype_boost(meta.get("doc_type"), intent)
        item["score"] = sim_score * 0.6 + kw_score * 0.2 + recency * 0.1 + boost
    return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)


def _build_candidates(raw_res: dict[str, Any]) -> list[dict[str, Any]]:
    ids = (raw_res.get("ids") or [[]])[0]
    documents = (raw_res.get("documents") or [[]])[0]
    metadatas = (raw_res.get("metadatas") or [[]])[0]
    distances = (raw_res.get("distances") or [[]])[0]
    candidates: list[dict[str, Any]] = []
    for idx, cid in enumerate(ids):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        doc = documents[idx] if idx < len(documents) else ""
        dist = distances[idx] if idx < len(distances) else None
        candidates.append({"id": cid, "document": doc or "", "meta": meta or {}, "distance": dist})
    return candidates


def _ensure_criteria(chunks: list[ContextChunk], ctx: QueryContext, trace_id: str, base_embedding: list[float] | None) -> Tuple[list[ContextChunk], bool]:
    """
    critique intent 시 critique_criteria doc_type 최소 14개 포함하도록 시도.
    """
    if (ctx.intent or "").lower() != "critique":
        return chunks, False

    criteria_chunks = [c for c in chunks if c.doc_type == "critique_criteria"]
    if len(criteria_chunks) >= 14:
        return chunks, False

    # 추가 fetch
    extra_needed = 14 - len(criteria_chunks)
    embed = base_embedding if base_embedding is not None else [0.0]
    extra_res = chroma_query(
        embedding=embed,
        where_filter={"doc_type": "critique_criteria"},
        n_results=max(20, extra_needed),
    )
    extra_candidates = _build_candidates(extra_res)
    extra_chunks: list[ContextChunk] = []
    for cand in extra_candidates:
        meta = cand.get("meta") or {}
        extra_chunks.append(
            ContextChunk(
                chunk_id=meta.get("chunk_id") or cand.get("id") or "",
                doc_type=meta.get("doc_type") or "critique_criteria",
                doc_id=meta.get("doc_id") or "",
                title=meta.get("title") or "",
                source=None,
                created_at=_to_datetime(meta.get("created_at")) or datetime.now(),
                excerpt=cand.get("document") or "",
            )
        )
        if len(extra_chunks) >= extra_needed:
            break

    merged = chunks + extra_chunks
    return merged, True


def retrieve_context(query_text: str, query_context: QueryContext, trace_id: str | None = None) -> ContextPack:
    trace_id = ensure_trace_id(trace_id)
    start = time.time()
    filters = {"doc_types": query_context.doc_types or [], "time_range": query_context.time_range.dict() if query_context.time_range else None}

    query_vecs = _embedding_provider.embed([query_text])
    query_embedding = query_vecs[0] if query_vecs else []

    raw_res = chroma_query(embedding=query_embedding, where_filter=None, n_results=40)
    candidates = _build_candidates(raw_res)
    filtered = _apply_filters(candidates, query_context)
    reranked = _rerank(filtered, query_context)

    top_k = query_context.k or 5
    top_items = reranked[:top_k]

    chunks: list[ContextChunk] = []
    for item in top_items:
        meta = item.get("meta") or {}
        chunks.append(
            ContextChunk(
                chunk_id=meta.get("chunk_id") or item.get("id") or "",
                doc_type=meta.get("doc_type") or "",
                doc_id=meta.get("doc_id") or "",
                title=meta.get("title") or "",
                source=None,
                created_at=_to_datetime(meta.get("created_at")) or datetime.now(),
                excerpt=item.get("document") or "",
                hash=meta.get("hash"),
            )
        )

    chunks, criteria_missing = _ensure_criteria(chunks, query_context, trace_id, query_embedding)

    latency_ms = int((time.time() - start) * 1000)
    citations_path: str | None = None
    try:
        citations_path = write_citations(trace_id, query_context.intent, query_context, RetrievalStats(
            total_candidates=len(candidates),
            selected_chunks=len(chunks),
            filters={},
            latency_ms=latency_ms,
        ), chunks)
    except Exception as exc:
        log_event(trace_id, "citations_write_failed", {"error": str(exc)})

    stats = RetrievalStats(
        total_candidates=len(candidates),
        selected_chunks=len(chunks),
        filters={"doc_types": query_context.doc_types, "time_range": filters.get("time_range"), "criteria_missing": criteria_missing},
        latency_ms=latency_ms,
        citations_path=citations_path,
    )
    log_event(trace_id, "retrieve_context_done", {"candidates": len(candidates), "selected": len(chunks), "criteria_missing": criteria_missing})
    return ContextPack(context_pack=chunks, retrieval_stats=stats, trace_id=trace_id)
