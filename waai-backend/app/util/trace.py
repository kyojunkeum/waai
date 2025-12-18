from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("waai.trace")


def generate_trace_id() -> str:
    """
    Generate a short, URL-safe trace id (legacy).
    """
    ts = int(time.time() * 1000)
    rand = uuid.uuid4().hex[:8]
    return f"trace-{ts}-{rand}"


def new_trace_id() -> str:
    """
    Generate a uuid4-based trace id.
    """
    return f"trace-{uuid.uuid4()}"


def ensure_trace_id(trace_id: str | None) -> str:
    """
    Return existing trace_id or generate a new one.
    """
    return trace_id if trace_id else new_trace_id()


def log_event(trace_id: str, event_name: str, payload: dict[str, Any] | None = None) -> None:
    """
    Structured JSON logger for tracing.
    """
    try:
        record = {
            "trace_id": trace_id,
            "event": event_name,
            "payload": payload or {},
        }
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:
        # 로깅 실패 시에도 애플리케이션 흐름에 영향 주지 않음
        logger.exception("failed to log event: %s", event_name)
