from __future__ import annotations

import os
import re
from pathlib import Path

BRAINSTORM_IDEAS_DIR = Path(os.environ.get("BRAINSTORM_IDEAS_DIR", "/home/witness/memory/ideas")).resolve()


def slugify_ko(text: str) -> str:
    cleaned = re.sub(r"\s+", "-", (text or "").strip())
    cleaned = re.sub(r"[^A-Za-z0-9가-힣_-]", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    cleaned = cleaned[:40]
    return cleaned or "idea"


def build_idea_filename(date_yyyymmdd: str, uuid_str: str, slug: str) -> str:
    safe_slug = slugify_ko(slug)
    return f"{date_yyyymmdd}__{uuid_str}__{safe_slug}.txt"


def char_limit_1000(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= 1000:
        return cleaned

    cut = None
    for match in re.finditer(r"[.!?。！？]", cleaned):
        idx = match.end()
        if idx <= 1000:
            cut = idx
        else:
            break

    if cut is None:
        cut = 997

    trimmed = cleaned[:cut].rstrip()
    if len(trimmed) > 999:
        trimmed = trimmed[:999].rstrip()
    return f"{trimmed}…"
