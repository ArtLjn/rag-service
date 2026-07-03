"""Embedding 客户端：在线 HTTP 调用。

默认对接 Google gemini-embedding-001（与主系统 ai-agent-learning/config.yaml 一致），
通过 `EMBEDDING_PROVIDER` 切换协议：
- `google`（默认）：原生 Generative Language API（`/models/{model}:embedContent`，单条/请求）
- `openai`：OpenAI 兼容协议（`/v1/embeddings`，批量）

策略：
- 服务启动不阻塞
- 首次调用 embed() 时同步探测 API Key 与端点可用
- 探测成功后标记 ready；失败记 warning，调用方应捕获 EmbedderUnavailable 并降级
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import EmbedderUnavailable
from app.core.logging import logger

_LOCK = threading.Lock()
_INSTANCE: _Embedder | None = None


class _Embedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        batch_size: int,
        provider: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.provider = provider.lower()
        self._state = "idle"  # idle / probing / ready / failed
        self._error: str | None = None
        self._lock = threading.Lock()
        self._client: httpx.AsyncClient | None = None

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return self._state == "ready"

    def is_failed(self) -> bool:
        return self._state == "failed"

    def embed_query(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim
        loop = asyncio.new_event_loop()
        try:
            vectors = loop.run_until_complete(self.embed([text]))
        finally:
            loop.close()
        return vectors[0] if vectors else [0.0] * self.dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        await self._ensure_ready()
        assert self._client is not None
        if self.provider == "google":
            return await self._embed_google(texts)
        return await self._embed_openai(texts)

    async def _ensure_ready(self) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise EmbedderUnavailable(f"embedder unavailable: {self._error}")
        await self._probe()

    async def _probe(self) -> None:
        with self._lock:
            if self._state in {"ready", "failed"}:
                return
            self._state = "probing"
        try:
            if not self.api_key:
                raise ValueError("EMBEDDING_API_KEY not set")
            headers = self._auth_headers()
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=settings.http_timeout,
                headers=headers,
            )
            # 直接调底层 embed 函数，避免经过 _ensure_ready 再次进入 probe
            if self.provider == "google":
                probe_vec = await self._embed_google(["probe"])
            else:
                probe_vec = await self._embed_openai(["probe"])
            if not probe_vec or len(probe_vec[0]) != self.dim:
                got = len(probe_vec[0]) if probe_vec else 0
                raise ValueError(f"embedding dim mismatch: expected {self.dim}, got {got}")
            self._state = "ready"
            logger.info(f"embedder ready: provider={self.provider} model={self.model} dim={self.dim}")
        except Exception as exc:
            self._state = "failed"
            self._error = repr(exc)
            logger.warning(f"embedder probe failed: {exc!r}")
            raise EmbedderUnavailable(f"embedder probe failed: {exc}") from exc

    def _auth_headers(self) -> dict[str, str]:
        if self.provider == "google":
            return {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _embed_google(self, texts: list[str]) -> list[list[float]]:
        assert self._client is not None
        results: list[list[float]] = []
        url = f"/models/{self.model}:embedContent"
        for text in texts:
            payload = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
            }
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            except httpx.HTTPStatusError as exc:
                raise EmbedderUnavailable(
                    f"google embedding API status {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except Exception as exc:
                raise EmbedderUnavailable(f"google embedding API call failed: {exc}") from exc
            values = data.get("embedding", {}).get("values", [])
            results.append(list(map(float, values)))
        return results

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        assert self._client is not None
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch}
            try:
                response = await self._client.post("/v1/embeddings", json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            except httpx.HTTPStatusError as exc:
                raise EmbedderUnavailable(
                    f"openai embedding API status {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except Exception as exc:
                raise EmbedderUnavailable(f"openai embedding API call failed: {exc}") from exc
            items = data.get("data") or []
            items.sort(key=lambda item: item.get("index", 0))
            for item in items:
                results.append(list(map(float, item.get("embedding", []))))
        return results


def get_embedder() -> _Embedder:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = _Embedder(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
                provider=settings.embedding_provider,
            )
    return _INSTANCE


def reset_embedder() -> None:
    """测试用：重置单例。"""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None


def estimate_vector_dim(model_name: str) -> int:
    """根据模型名推断向量维度。"""
    table = {
        "gemini-embedding-001": 3072,
        "text-embedding-004": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
    }
    return table.get(model_name, settings.embedding_dim)


__all__ = [
    "estimate_vector_dim",
    "get_embedder",
    "reset_embedder",
]
