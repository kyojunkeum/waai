"""Shared utility functions for WAAI backend."""

from .trace import ensure_trace_id, generate_trace_id, new_trace_id, log_event
from .fileio import (
    compute_sha256,
    parse_front_matter,
    read_md_with_front_matter,
    read_text,
    safe_glob_md,
    safe_glob,
    sha256_file,
    write_md_with_front_matter,
    write_text,
)

__all__ = [
    "ensure_trace_id",
    "generate_trace_id",
    "new_trace_id",
    "log_event",
    "compute_sha256",
    "sha256_file",
    "parse_front_matter",
    "read_md_with_front_matter",
    "write_md_with_front_matter",
    "read_text",
    "write_text",
    "safe_glob_md",
    "safe_glob",
]
