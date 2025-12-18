"""Indexer utilities."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from app.rag.chunker import chunk_document
from app.rag.chroma_client import ChunkRecord, upsert_chunks
from app.rag.embedding import OllamaEmbeddingProvider
from app.util import (
    read_md_with_front_matter,
    write_md_with_front_matter,
    sha256_file,
    safe_glob,
)
from app.util.trace import ensure_trace_id, log_event

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
if not DEFAULT_DATA_ROOT.exists():
    repo_root = Path(__file__).resolve().parents[3]
    DEFAULT_DATA_ROOT = repo_root / "data"

# doc_type -> 폴더 매핑 (필요 시 조정 가능)
DOC_TYPE_DIRS = {
    "diary": Path(os.environ.get("DIARY_ROOT", DEFAULT_DATA_ROOT / "diary")),
    "ideas": Path(os.environ.get("IDEAS_ROOT", DEFAULT_DATA_ROOT / "ideas")),
    "web_research": Path(os.environ.get("WEB_RESEARCH_ROOT", DEFAULT_DATA_ROOT / "web_research")),
    "works": Path(os.environ.get("WORKS_ROOT", DEFAULT_DATA_ROOT / "works")),
    "bible": Path(os.environ.get("BIBLE_ROOT", DEFAULT_DATA_ROOT / "bible")),
    "critique": Path(os.environ.get("CRITIQUE_RESULTS_ROOT", DEFAULT_DATA_ROOT / "critique/results")).parent,
    "critique_criteria": Path(os.environ.get("CRITIQUE_CRITERIA_PATH", DEFAULT_DATA_ROOT / "critique/criteria/합평기준규칙.md")).parent,
}

_embedding_provider = OllamaEmbeddingProvider()


def _embed_chunks(chunks: List[ChunkRecord]) -> List[ChunkRecord]:
    texts = [c.excerpt or "" for c in chunks]
    vectors = _embedding_provider.embed(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec
    return chunks


def index_paths(paths: List[str], doc_type: str, force: bool = False, trace_id: str | None = None) -> Dict[str, Any]:
    trace_id = ensure_trace_id(trace_id)
    started = time.time()
    indexed_files: list[str] = []
    skipped_files: list[str] = []
    failed_files: list[dict[str, str]] = []
    indexed_chunks = 0

    log_event(trace_id, "index_paths_start", {"paths": paths, "doc_type": doc_type, "force": force})

    for raw_path in paths:
        path = Path(raw_path)
        try:
            meta, body = read_md_with_front_matter(path)
        except Exception as exc:
            failed_files.append({"path": str(path), "reason": str(exc)})
            log_event(trace_id, "index_skip_no_yaml", {"path": str(path), "error": str(exc)})
            continue

        file_hash = sha256_file(path)
        meta_hash = meta.get("hash")
        if (not force) and meta_hash and meta_hash == file_hash:
            skipped_files.append(str(path))
            continue

        # chunking
        chunks = chunk_document(
            doc_type=doc_type,
            doc_id=str(path.relative_to(path.parent)),
            title=meta.get("title") or path.stem,
            created_at=meta.get("created_at") or meta.get("date"),
            body_text=body,
            yaml_meta=meta,
        )

        # embed and upsert
        chunks = _embed_chunks(chunks)
        upsert_chunks(chunks, store_text=False, trace_id=trace_id)

        # update meta hash and write back
        meta["hash"] = file_hash
        write_md_with_front_matter(path, meta, body)

        indexed_files.append(str(path))
        indexed_chunks += len(chunks)

    latency_ms = int((time.time() - started) * 1000)
    result = {
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files,
        "failed_files": failed_files,
        "latency_ms": latency_ms,
        "trace_id": trace_id,
    }
    log_event(trace_id, "index_paths_done", result)
    return result


def _collect_paths(doc_type: str) -> List[str]:
    root = DOC_TYPE_DIRS.get(doc_type)
    if not root:
        return []
    if not root.exists():
        return []
    return [str(p) for p in safe_glob(root, "*.md")]


def index_all(doc_types: List[str], force: bool = False, trace_id: str | None = None) -> Dict[str, Any]:
    trace_id = ensure_trace_id(trace_id)
    started = time.time()
    all_indexed_files: list[str] = []
    all_skipped_files: list[str] = []
    all_failed_files: list[dict[str, str]] = []
    total_chunks = 0

    for dt in doc_types:
        paths = _collect_paths(dt)
        res = index_paths(paths, dt, force=force, trace_id=trace_id)
        all_indexed_files.extend(res["indexed_files"])
        all_skipped_files.extend(res["skipped_files"])
        all_failed_files.extend(res["failed_files"])
        total_chunks += res["indexed_chunks"]

    latency_ms = int((time.time() - started) * 1000)
    result = {
        "indexed_files": all_indexed_files,
        "indexed_chunks": total_chunks,
        "skipped_files": all_skipped_files,
        "failed_files": all_failed_files,
        "latency_ms": latency_ms,
        "trace_id": trace_id,
    }
    log_event(trace_id, "index_all_done", result)
    return result
