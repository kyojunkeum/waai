import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


class DummyEmbedder:
    def __init__(self, vector=None):
        self.vector = vector or [1.0, 0.0, 0.0]

    def embed(self, texts):
        return [self.vector for _ in texts]


def reload_rag_modules(monkeypatch, chroma_dir, citations_dir):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))
    monkeypatch.setenv("CITATIONS_DIR", str(citations_dir))
    modules = [
        "app.rag.chroma_client",
        "app.rag.embedding",
        "app.rag.chunker",
        "app.rag.indexer",
        "app.rag.retriever",
        "app.rag.citations",
    ]
    for m in modules:
        if m in sys.modules:
            importlib.reload(sys.modules[m])
        else:
            importlib.import_module(m)
    # rewire embedder
    import app.rag.indexer as indexer
    import app.rag.retriever as retriever

    dummy = DummyEmbedder()
    indexer._embedding_provider = dummy
    retriever._embedding_provider = dummy


def test_planning_rag_flow(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma1"
    citations_dir = tmp_path / "citations1"
    reload_rag_modules(monkeypatch, chroma_dir, citations_dir)

    from app.rag.indexer import index_paths
    from app.rag.retriever import retrieve_context
    from app.rag.models import QueryContext

    fixture = Path(__file__).parent / "fixtures" / "web_research_sample.md"
    work_file = tmp_path / "web1.md"
    shutil.copy(fixture, work_file)

    res = index_paths([str(work_file)], doc_type="web_research", force=True, trace_id="test-trace")
    assert res["indexed_files"]

    ctx = QueryContext(
        topic="샘플 기획",
        keywords=["AI"],
        time_range=None,
        tone=None,
        intent="planning",
        doc_types=["web_research"],
        people=[],
        must_cite=True,
        k=5,
    )
    pack = retrieve_context("AI 기획", ctx, trace_id="test-trace")
    assert pack.context_pack
    assert pack.retrieval_stats.citations_path
    assert Path(pack.retrieval_stats.citations_path).exists()


def test_critique_rag_with_criteria(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma2"
    citations_dir = tmp_path / "citations2"
    reload_rag_modules(monkeypatch, chroma_dir, citations_dir)

    from app.rag.indexer import index_paths
    from app.rag.retriever import retrieve_context
    from app.rag.models import QueryContext

    crit_fixture = Path(__file__).parent / "fixtures" / "critique_criteria_sample.md"
    criteria_files = []
    for i in range(14):
        target = tmp_path / f"criteria_{i}.md"
        shutil.copy(crit_fixture, target)
        criteria_files.append(target)

    res = index_paths([str(p) for p in criteria_files], doc_type="critique_criteria", force=True, trace_id="crit-trace")
    assert len(res["indexed_files"]) == 14

    ctx = QueryContext(
        topic="합평 테스트",
        keywords=[],
        time_range=None,
        tone=None,
        intent="critique",
        doc_types=["critique_criteria"],
        people=[],
        must_cite=True,
        k=14,
    )
    pack = retrieve_context("합평 원고", ctx, trace_id="crit-trace")
    criteria_chunks = [c for c in pack.context_pack if c.doc_type in ("critique_criteria", "critique/criteria")]
    assert len(criteria_chunks) >= 14
    assert pack.retrieval_stats.citations_path
    assert Path(pack.retrieval_stats.citations_path).exists()
