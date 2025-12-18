"""Citation handling."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List

from app.rag.models import ContextChunk, QueryContext, RetrievalStats
from app.util.trace import ensure_trace_id, log_event

DEFAULT_CITATIONS_DIR = Path(os.environ.get("CITATIONS_DIR", "/waai/data/outputs/citations"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_citations(
    trace_id: str,
    intent: str | None,
    query_context: QueryContext,
    retrieval_stats: RetrievalStats,
    chunks: List[ContextChunk],
    output_dir: str | Path = DEFAULT_CITATIONS_DIR,
) -> str:
    trace_id = ensure_trace_id(trace_id)
    out_dir = _ensure_dir(Path(output_dir))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{trace_id}_citations.md"
    path = out_dir / filename

    ctx_json = json.dumps(query_context.dict(), ensure_ascii=False, indent=2)
    stats_json = json.dumps(retrieval_stats.dict(), ensure_ascii=False, indent=2)

    lines = [
        f"# Citations ({intent or 'planning'})",
        "",
        f"- trace_id: {trace_id}",
        f"- intent: {intent or ''}",
        "- query_context:",
        "```json",
        ctx_json,
        "```",
        "- retrieval_stats:",
        "```json",
        stats_json,
        "```",
        "",
        "## Chunks",
    ]

    for c in chunks:
        lines.extend(
            [
                f"### {c.chunk_id}",
                f"- doc_type: {c.doc_type}",
                f"- title: {c.title}",
                f"- doc_id: {c.doc_id}",
                f"- created_at: {c.created_at}",
                f"- url: {getattr(c, 'source', None) and getattr(c.source, 'url', None) if hasattr(c, 'source') else None}",
                f"- hash: {getattr(c, 'hash', None) if hasattr(c, 'hash') else None}",
                "",
                "```text",
                c.excerpt,
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    log_event(trace_id, "citations_written", {"path": str(path), "count": len(chunks)})
    return str(path)
