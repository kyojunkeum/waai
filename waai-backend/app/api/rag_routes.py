from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.rag.models import (
    ContextPack,
    ErrorPayload,
    ErrorResponse,
    RagIndexAllRequest,
    RagIndexRequest,
    RagSearchRequest,
)
from app.rag.retriever import retrieve_context
from app.rag.indexer import index_paths, index_all
from app.rag.chroma_client import (
    count as chroma_count,
    peek as chroma_peek,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
)
from app.util.trace import ensure_trace_id, log_event, new_trace_id

router = APIRouter(prefix="/api/rag", tags=["rag"])


def _build_index_message(res: dict[str, object]) -> str:
    indexed = len(res.get("indexed_files", []))
    skipped = len(res.get("skipped_files", []))
    failed = len(res.get("failed_files", []))
    chunks = res.get("indexed_chunks", 0)
    return f"Indexed {indexed} files ({chunks} chunks), skipped {skipped}, failed {failed}"


@router.post(
    "/search",
    response_model=ContextPack,
    responses={501: {"model": ErrorResponse}},
    operation_id="rag_search",
)
async def rag_search(body: RagSearchRequest):
    trace_id = new_trace_id()
    log_event(trace_id, "rag_search_request", {"body": body.dict()})
    try:
        ctx_pack = retrieve_context(body.query_text, body.query_context, trace_id=trace_id)
        return ctx_pack
    except Exception as exc:
        error = ErrorResponse(
            error=ErrorPayload(
                code="RAG_SEARCH_FAILED",
                message="RAG search failed",
                detail=str(exc),
            ),
            trace_id=trace_id,
        )
        log_event(trace_id, "rag_search_error", {"error": str(exc)})
        return JSONResponse(status_code=500, content=error.dict())


@router.post(
    "/index",
    responses={500: {"model": ErrorResponse}},
    operation_id="rag_index",
)
async def rag_index(body: RagIndexRequest):
    trace_id = ensure_trace_id(None)
    log_event(trace_id, "rag_index_request", {"body": body.dict()})
    try:
        res = index_paths(body.paths, body.doc_type, force=body.force, trace_id=trace_id)
        message = _build_index_message(res)
        res["trace_id"] = trace_id
        return {"success": True, "message": message, "data": res, "trace_id": trace_id}
    except Exception as exc:
        error = ErrorResponse(
            error=ErrorPayload(
                code="RAG_INDEX_FAILED",
                message="RAG indexer failed",
                detail=str(exc),
            ),
            trace_id=trace_id,
        )
        log_event(trace_id, "rag_index_error", {"error": str(exc)})
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "RAG indexer failed", "error": error.dict(), "trace_id": trace_id},
        )


@router.post(
    "/index_all",
    responses={500: {"model": ErrorResponse}},
    operation_id="ragIndexAll",
)
async def rag_index_all(body: RagIndexAllRequest):
    trace_id = ensure_trace_id(None)
    log_event(trace_id, "rag_index_all_request", {"body": body.dict()})
    try:
        res = index_all(body.doc_types, force=body.force, trace_id=trace_id)
        message = _build_index_message(res)
        res["trace_id"] = trace_id
        return {"success": True, "message": message, "data": res, "trace_id": trace_id}
    except Exception as exc:
        error = ErrorResponse(
            error=ErrorPayload(
                code="RAG_INDEX_ALL_FAILED",
                message="RAG bulk indexing failed",
                detail=str(exc),
            ),
            trace_id=trace_id,
        )
        log_event(trace_id, "rag_index_all_error", {"error": str(exc)})
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "RAG bulk indexing failed", "error": error.dict(), "trace_id": trace_id},
        )


@router.get(
    "/stats",
    responses={200: {"description": "RAG stats"}, 500: {"model": ErrorResponse}},
    operation_id="rag_stats",
)
async def rag_stats():
    trace_id = ensure_trace_id(None)
    try:
        total = chroma_count()
        peeked_raw = chroma_peek(5)
        # ensure serializable
        peeked = []
        if isinstance(peeked_raw, list):
            for item in peeked_raw:
                if isinstance(item, dict):
                    peeked.append({k: v for k, v in item.items()})
                else:
                    try:
                        peeked.append(dict(item))
                    except Exception:
                        peeked.append(str(item))
        return {
            "success": True,
            "message": "ok",
            "data": {
                "document_count": total,  # alias
                "chunk_count": total,
                "peek": peeked,
                "collection_name": CHROMA_COLLECTION,
                "persist_dir": CHROMA_PERSIST_DIR,
                "last_indexed_at": None,
            },
            "trace_id": trace_id,
        }
    except Exception as exc:
        error = ErrorResponse(
            error=ErrorPayload(
                code="RAG_STATS_FAILED",
                message="failed to fetch rag stats",
                detail=str(exc),
            ),
            trace_id=trace_id,
        )
        log_event(trace_id, "rag_stats_error", {"error": str(exc)})
        return JSONResponse(status_code=500, content=error.dict())
