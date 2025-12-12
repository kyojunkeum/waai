from __future__ import annotations

import ast
import json
import re
from typing import Any

# 헤더 우선 추출용 패턴
HEADER_PATTERN = re.compile(r"\[(기획서에 바로 쓸 근거 목록(?:\(JSON\))?|근거 목록)\]", re.I)
# JSON 배열처럼 보이는 블록을 찾기 위한 패턴 (중첩 최소화)
JSON_ARRAY_PATTERN = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]", re.M)

SMART_QUOTE_TABLE = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‛": "'",
    }
)


def _strip_code_fences(text: str) -> str:
    # ```json ... ``` 같은 블록 제거
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _clean_jsonish(text: str) -> str:
    cleaned = _strip_code_fences(text).translate(SMART_QUOTE_TABLE).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    # 중괄호/대괄호 앞의 trailing comma 정리
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _normalize_entry(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)  # 원본 필드를 보존
    normalized.setdefault("type", "diary")
    for key in ("date", "title", "excerpt", "reason"):
        value = normalized.get(key, "")
        normalized[key] = "" if value is None else str(value).strip()
    return normalized


def _parse_jsonish_array(blob: str) -> list[dict[str, Any]] | None:
    cleaned = _clean_jsonish(blob)
    parsers = (
        lambda s: json.loads(s),
        lambda s: json.loads(s.replace("'", '"')),
        lambda s: ast.literal_eval(s),
    )

    for parser in parsers:
        try:
            data = parser(cleaned)
            if isinstance(data, list):
                items = [_normalize_entry(i) for i in data if isinstance(i, dict)]
                if items:
                    return items
        except Exception:
            continue
    return None


def _collect_candidate_blocks(summary: str) -> list[str]:
    candidates: list[str] = []

    for match in HEADER_PATTERN.finditer(summary):
        start = match.end()
        rest = summary[start:]
        next_header = re.search(r"\n\[[^\n\]]+\]", rest)
        end = next_header.start() + start if next_header else len(summary)
        block = summary[start:end].strip()
        if block:
            candidates.append(block)

    if not candidates:
        for match in JSON_ARRAY_PATTERN.finditer(summary):
            candidates.append(match.group(0))

    return candidates


def extract_sources_from_summary(diary_summary: str) -> list[dict[str, Any]]:
    """
    요약 문자열에서 근거 JSON 배열을 최대한 복구해 추출.
    - 헤더가 있으면 우선 사용
    - JSON-ish 배열을 발견하면 클린업 후 파싱
    - 실패 시 항상 [] 반환 (예외 금지)
    """
    try:
        candidates = _collect_candidate_blocks(diary_summary or "")
        for cand in candidates:
            parsed = _parse_jsonish_array(cand)
            if parsed is not None:
                return parsed
        return []
    except Exception:
        return []
