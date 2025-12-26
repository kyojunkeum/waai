from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Tuple

import yaml

from .monitor_log import monitor_log

logger = logging.getLogger(__name__)


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    """
    Compute SHA256 hash of a file content.
    """
    data = Path(path).read_bytes()
    return compute_sha256(data)


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding=encoding)
    return str(target)


def parse_front_matter(md_text: str) -> Tuple[dict[str, Any], str]:
    """
    Parse YAML front matter from Markdown text.
    Returns (meta, body). If no front matter exists or parse fails, meta is {}.
    """
    if not md_text.lstrip().startswith("---"):
        return {}, md_text

    parts = md_text.split("---", 2)
    if len(parts) < 3:
        return {}, md_text

    meta_raw = parts[1]
    body = parts[2]
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except Exception:
        meta = {}
    return meta, body


def read_md_with_front_matter(path: str | Path) -> tuple[dict[str, Any], str]:
    """
    Read a Markdown file and split YAML front matter and body.
    Raises ValueError if front matter is missing or invalid so callers can skip indexing.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    if not meta or not isinstance(meta, dict):
        logger.error("missing or invalid YAML front matter: %s", p)
        raise ValueError(f"missing or invalid YAML front matter: {p}")
    return meta, body.lstrip("\n")


def write_md_with_front_matter(path: str | Path, yaml_dict: dict[str, Any], body_text: str) -> str:
    """
    Write a Markdown file with YAML front matter and body.
    """
    meta_text = yaml.safe_dump(yaml_dict or {}, allow_unicode=True, sort_keys=False)
    content = f"---\n{meta_text}---\n\n{body_text}"
    return write_text(path, content, encoding="utf-8")


def safe_glob_md(root: str | Path = "/waai/data") -> list[Path]:
    """
    Safely glob markdown files under root. Skips errors instead of failing the whole process.
    """
    base = Path(root)
    if not base.exists():
        logger.warning("safe_glob_md root does not exist: %s", base)
        return []

    results: list[Path] = []
    for path in base.rglob("*.md"):
        try:
            if path.is_file():
                results.append(path)
        except OSError as exc:
            logger.error("failed to inspect path=%s err=%s", path, exc)
            continue

    results = sorted(results)
    monitor_log("file_list", "safe_glob_md", {"root": str(base), "count": len(results)})
    return results


def safe_glob(root: str | Path, pattern: str = "*") -> list[Path]:
    """
    Safe glob helper for arbitrary patterns. Returns sorted list and swallows filesystem errors.
    """
    base = Path(root)
    if not base.exists():
        logger.warning("safe_glob root does not exist: %s", base)
        return []

    results: list[Path] = []
    try:
        for path in base.rglob(pattern):
            try:
                if path.is_file():
                    results.append(path)
            except OSError as exc:
                logger.error("failed to inspect path=%s err=%s", path, exc)
                continue
    except OSError as exc:
        logger.error("failed to glob root=%s pattern=%s err=%s", base, pattern, exc)
        return []

    results = sorted(results)
    monitor_log(
        "file_list",
        "safe_glob",
        {"root": str(base), "pattern": pattern, "count": len(results)},
    )
    return results
