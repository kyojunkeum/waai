"""Embedding providers for RAG."""

from __future__ import annotations

import logging
import os
from typing import Any, List

import httpx

from app.rag.models import ErrorPayload, ErrorResponse
from app.util.trace import ensure_trace_id, log_event

logger = logging.getLogger("waai.embedding")


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or "http://ollama:11434").rstrip(
            "/"
        )
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL") or "nomic-embed-text"
        self.timeout = timeout

    def _embed_one(self, text: str, trace_id: str) -> list[float] | None:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise ValueError("missing embedding")
                return [float(x) for x in embedding]
        except Exception as exc:
            error = ErrorResponse(
                error=ErrorPayload(
                    code="EMBEDDING_FAILED",
                    message="failed to get embedding from ollama",
                    detail=str(exc),
                ),
                trace_id=trace_id,
            )
            logger.error("embedding failed trace_id=%s err=%s", trace_id, exc)
            log_event(trace_id, "embedding_failed", error.dict())
            return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        trace_id = ensure_trace_id(None)
        results: List[List[float]] = []
        for t in texts:
            vec = self._embed_one(t, trace_id)
            if vec is not None:
                results.append(vec)
            else:
                # 개별 실패는 빈 벡터로 대체하여 전체 파이프라인 중단 방지
                results.append([])
        log_event(trace_id, "embedding_batch_complete", {"count": len(texts)})
        return results
