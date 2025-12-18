"""Document chunking utilities."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, List

from app.rag.chroma_client import ChunkRecord
from app.util.fileio import compute_sha256

KST = timezone(timedelta(hours=9))

# doc_type별 튜닝 가능 파라미터
CHUNK_CONFIG: dict[str, dict[str, int]] = {
    "diary": {"target_tokens": 256, "overlap": 32},
    "ideas": {"target_tokens": 256, "overlap": 32},
    "web_research": {"target_tokens": 320, "overlap": 40},
    "works": {"target_tokens": 320, "overlap": 40},
    "bible": {"target_tokens": 200, "overlap": 24},
    "critique": {"target_tokens": 256, "overlap": 32},
    "critique_criteria": {"target_tokens": 256, "overlap": 32},
}


def _approx_tokens(text: str) -> int:
    """간이 토큰 추정: char_len / 4."""
    return max(1, int(len(text) / 4))


def _normalize_created_at(value: Any) -> str:
    if value is None:
        return datetime.now(tz=KST).isoformat()
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except Exception:
            dt = datetime.now(tz=KST)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    else:
        dt = dt.astimezone(KST)
    return dt.isoformat()


def _sanitize_id(val: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", val)


def _split_sections(text: str) -> List[str]:
    """
    간단한 섹션 추정: heading(##, ###), 긴 공백을 기준으로 분리.
    """
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            sections.append("\n".join(current).strip())

    for line in lines:
        heading = re.match(r"^#{2,3}\\s", line)
        if heading:
            flush()
            current = [line]
            continue
        if not line.strip():
            # 빈 줄 두 개 이상이면 분리
            if current and current[-1].strip() == "":
                flush()
                current = []
            else:
                current.append("")
            continue
        current.append(line)
    flush()

    # 빈 섹션 제거
    return [s for s in sections if s.strip()]


def _chunk_section(section: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    """
    섹션을 target_tokens 기반으로 분할하며 overlap 적용.
    """
    chunks: list[str] = []
    words = section.split()
    approx_tokens = [_approx_tokens(w) for w in words]

    start = 0
    while start < len(words):
        current_tokens = 0
        end = start
        while end < len(words) and current_tokens + approx_tokens[end] <= target_tokens:
            current_tokens += approx_tokens[end]
            end += 1
        if end == start:  # 토큰이 너무 짧으면 최소 한 단어라도 포함
            end = min(len(words), start + 1)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(words):
            break
        # overlap 계산
        back_tokens = 0
        overlap_words = 0
        idx = end
        while idx > start:
            back_tokens += approx_tokens[idx - 1]
            overlap_words += 1
            if back_tokens >= overlap_tokens:
                break
            idx -= 1
        # overlap_words는 전체 chunk 길이보다 작게 유지하여 진행 보장
        overlap_words = min(overlap_words, max(0, len(chunk_words) - 1))
        start = end - overlap_words
    return chunks


def chunk_document(
    doc_type: str,
    doc_id: str,
    title: str,
    created_at: Any,
    body_text: str,
    yaml_meta: dict[str, Any] | None = None,
) -> List[ChunkRecord]:
    """
    문서를 doc_type별 파라미터에 따라 chunking하여 ChunkRecord 리스트를 반환한다.
    """
    cfg = CHUNK_CONFIG.get(doc_type, {"target_tokens": 256, "overlap": 32})
    target_tokens = max(32, cfg.get("target_tokens", 256))
    overlap_tokens = max(0, cfg.get("overlap", 32))
    created_at_iso = _normalize_created_at(created_at)
    clean_doc_id = _sanitize_id(doc_id or "doc")

    sections = _split_sections(body_text or "")
    if not sections:
        sections = [body_text or ""]

    all_chunks: list[str] = []
    for sec in sections:
        sec_chunks = _chunk_section(sec, target_tokens, overlap_tokens)
        all_chunks.extend(sec_chunks)

    chunk_records: list[ChunkRecord] = []
    for idx, chunk_text in enumerate(all_chunks, start=1):
        chunk_id = f"{clean_doc_id}_c{idx}"
        token_len = _approx_tokens(chunk_text)
        excerpt = chunk_text[:500]
        hash_val = compute_sha256(chunk_text.encode("utf-8"))
        chunk_records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_type=doc_type,
                title=title,
                embedding=[],
                created_at=created_at_iso,
                excerpt=excerpt,
                text=None,  # 기본은 excerpt만 저장, 전체 텍스트는 store_text 옵션으로 결정
                hash=hash_val,
                token_len=token_len,
                url=yaml_meta.get("source") if yaml_meta else None,
            )
        )

    return chunk_records
