from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(os.environ.get("MONITOR_LOG_DIR", "/data/_logs"))
LOG_FILE = LOG_DIR / "waai-backend.jsonl"


def monitor_log(category: str, message: str, payload: dict[str, Any] | None = None) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "module": "waai-backend",
        "category": category,
        "message": message,
        "payload": payload or {},
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 모니터 로그 실패는 서비스 흐름에 영향 주지 않도록 무시
        pass
