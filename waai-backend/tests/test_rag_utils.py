import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.util.fileio import read_md_with_front_matter, sha256_file
from app.rag.chunker import chunk_document
from app.rag.retriever import _rerank
from app.rag.models import QueryContext


def test_read_md_with_front_matter(tmp_path):
    src = Path(__file__).parent / "fixtures" / "web_research_sample.md"
    target = tmp_path / "sample.md"
    shutil.copy(src, target)
    meta, body = read_md_with_front_matter(target)
    assert meta.get("title") == "샘플 웹 리서치"
    assert "샘플 웹 리서치 내용" in body


def test_sha256_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)


@pytest.mark.parametrize("doc_type", ["diary", "ideas", "web_research", "works", "bible", "critique", "critique_criteria"])
def test_chunker_generates_sequential_ids(doc_type):
    chunks = chunk_document(
        doc_type=doc_type,
        doc_id="sample/doc",
        title="제목",
        created_at="2025-01-01T00:00:00+09:00",
        body_text="## 섹션\n첫 문장. 둘째 문장. 셋째 문장.",
        yaml_meta={},
    )
    assert len(chunks) >= 1
    assert chunks[0].chunk_id.endswith("_c1")
    assert chunks[0].doc_type == doc_type


def test_rerank_keyword_and_boost():
    candidates = [
        {"id": "a", "document": "일반 내용", "meta": {"doc_type": "diary"}, "distance": 0.2},
        {"id": "b", "document": "키워드 포함 AI", "meta": {"doc_type": "web_research"}, "distance": 0.2},
    ]
    ctx = QueryContext(
        topic="테스트",
        keywords=["AI"],
        time_range=None,
        tone=None,
        intent="planning",
        doc_types=["diary", "web_research"],
        people=[],
        must_cite=True,
        k=2,
    )
    reranked = _rerank(candidates, ctx)
    assert reranked[0]["id"] == "b"
