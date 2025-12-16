import re
from datetime import datetime
from pathlib import Path


def normalize_query(query: str) -> str:
    """
    Trim whitespace, drop wrapping quotes, and collapse internal spacing.
    """
    if query is None:
        return ""

    text = query.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(value: str | None, default: str = "article") -> str:
    """
    Produce a filesystem-friendly filename fragment (no path).
    """
    candidate = (value or default).strip()
    if not candidate:
        candidate = default

    candidate = re.sub(r"\s+", "_", candidate)
    candidate = re.sub(r"[^A-Za-z0-9._-]", "_", candidate)
    candidate = candidate.strip("._-") or default

    return candidate[:128]


def save_txt(out_dir: str | Path, title: str | None, url: str | None, body: str | None) -> str:
    base_dir = ensure_dir(out_dir)
    safe_title = safe_filename(title or url or "article")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filepath = base_dir / f"{timestamp}_{safe_title}.txt"

    content = f"URL: {url or ''}\nTITLE: {title or ''}\n\n{body or ''}\n"
    filepath.write_text(content, encoding="utf-8")

    return str(filepath)
