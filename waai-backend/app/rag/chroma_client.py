"""Chroma client helpers for RAG."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

from app.util.trace import ensure_trace_id, log_event

logger = logging.getLogger("waai.chroma")

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/waai/chroma")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")


try:
    import chromadb  # type: ignore
    from chromadb.api import ClientAPI  # type: ignore
except Exception:  # pragma: no cover - fallback when chromadb is absent
    chromadb = None
    ClientAPI = None  # type: ignore


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    embedding: List[float]
    created_at: str
    excerpt: str
    hash: str
    token_len: int
    url: Optional[str] = None
    text: Optional[str] = None  # 저장 옵션


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except PermissionError:
        # fallback to home-based directory if root-level is not writable
        fallback = Path.home() / ".waai" / "chroma"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning("fallback chroma dir to %s due to permission", fallback)
        return fallback


def _get_persist_dir() -> Path:
    return _ensure_dir(CHROMA_PERSIST_DIR)


# =========================
# Chroma client (real)
# =========================


def _get_chroma_client() -> ClientAPI | None:
    if chromadb is None:
        return None
    try:
        persist_dir = _get_persist_dir()
        client = chromadb.PersistentClient(path=str(persist_dir))
        return client
    except Exception as exc:  # pragma: no cover - chroma init failure
        logger.error("failed to init chroma client: %s", exc)
        return None


def get_collection():
    """
    Return Chroma collection if available, otherwise None.
    """
    client = _get_chroma_client()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    except Exception as exc:  # pragma: no cover
        logger.error("failed to get/create chroma collection: %s", exc)
        return None


def upsert_chunks(chunks: List[ChunkRecord], store_text: bool = False, trace_id: str | None = None):
    """
    Upsert chunk records into Chroma. Falls back to stub storage when chromadb is unavailable.
    """
    trace_id = ensure_trace_id(trace_id)
    log_event(trace_id, "chroma_upsert_start", {"count": len(chunks)})

    collection = get_collection()
    if collection is None:
        _stub_upsert(chunks, store_text, trace_id)
        return

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    documents: list[str] = []  # excerpt only by default

    for c in chunks:
        ids.append(c.chunk_id)
        embeddings.append(c.embedding)
        meta = {
            "doc_type": c.doc_type,
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "title": c.title,
            "created_at": c.created_at,
            "hash": c.hash,
            "url": c.url,
            "token_len": c.token_len,
        }
        metadatas.append(meta)
        doc_text = c.text if store_text else c.excerpt
        documents.append(doc_text)

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log_event(trace_id, "chroma_upsert_done", {"count": len(ids)})
    except Exception as exc:  # pragma: no cover
        logger.error("chroma upsert failed trace_id=%s err=%s", trace_id, exc)
        _stub_upsert(chunks, store_text, trace_id)


def query(embedding: list[float], where_filter: dict[str, Any] | None = None, n_results: int = 5):
    collection = get_collection()
    if collection is None:
        return _stub_query(embedding, where_filter, n_results)

    where_clause = where_filter if where_filter else None
    try:
        res = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_clause,
        )
        return res
    except Exception as exc:  # pragma: no cover
        logger.error("chroma query failed err=%s", exc)
        return _stub_query(embedding, where_filter, n_results)


def count() -> int:
    collection = get_collection()
    if collection is None:
        return _stub_count()
    try:
        return collection.count()
    except Exception as exc:  # pragma: no cover
        logger.error("chroma count failed err=%s", exc)
        return _stub_count()


def peek(n: int = 5):
    collection = get_collection()
    if collection is None:
        return _stub_peek(n)
    try:
        return collection.peek(n)
    except Exception as exc:  # pragma: no cover
        logger.error("chroma peek failed err=%s", exc)
        return _stub_peek(n)


# =========================
# Stub fallback (no chromadb)
# =========================

STUB_FILE = _get_persist_dir() / f"{CHROMA_COLLECTION}_stub.json"


def _load_stub() -> list[dict[str, Any]]:
    if not STUB_FILE.exists():
        return []
    try:
        return json.loads(STUB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_stub(data: Iterable[dict[str, Any]]) -> None:
    try:
        STUB_FILE.write_text(json.dumps(list(data), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("failed to save stub index: %s", exc)


def _stub_upsert(chunks: List[ChunkRecord], store_text: bool, trace_id: str) -> None:
    data = _load_stub()
    # deduplicate by chunk_id
    existing = {item.get("chunk_id"): item for item in data if "chunk_id" in item}

    for c in chunks:
        item = asdict(c)
        if not store_text:
            item["text"] = None
        existing[c.chunk_id] = item

    _save_stub(existing.values())
    log_event(trace_id, "stub_upsert_done", {"count": len(chunks)})


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    import math

    if not v1 or not v2 or len(v1) != len(v2):
        return -1.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return -1.0
    return dot / (norm1 * norm2)


def _matches_where(meta: dict[str, Any], where_filter: dict[str, Any]) -> bool:
    for k, v in (where_filter or {}).items():
        if meta.get(k) != v:
            return False
    return True


def _stub_query(embedding: list[float], where_filter: dict[str, Any] | None, n_results: int):
    data = _load_stub()
    scored = []
    for item in data:
        meta = {
            "doc_type": item.get("doc_type"),
            "doc_id": item.get("doc_id"),
            "chunk_id": item.get("chunk_id"),
            "title": item.get("title"),
            "created_at": item.get("created_at"),
            "hash": item.get("hash"),
            "url": item.get("url"),
            "token_len": item.get("token_len"),
        }
        if where_filter and not _matches_where(meta, where_filter):
            continue
        score = _cosine_similarity(embedding, item.get("embedding") or [])
        scored.append((score, item, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]
    return {
        "ids": [[t[1].get("chunk_id") for t in top]],
        "documents": [[t[1].get("text") or t[1].get("excerpt") for t in top]],
        "metadatas": [[t[2] for t in top]],
        "distances": [[1 - t[0] for t in top]],
    }


def _stub_count() -> int:
    return len(_load_stub())


def _stub_peek(n: int = 5):
    data = _load_stub()
    return data[:n]
