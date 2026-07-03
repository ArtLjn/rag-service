"""重排器：四种 provider 可选。

| provider | 触发条件 | 实现 | 是否需要下载 |
| --- | --- | --- | --- |
| `disabled` | `RERANKER_ENABLED=false` | 不重排，按原 score 排序，warning=reranker_disabled | 否 |
| `jina` | `RERANKER_ENABLED=true + RERANKER_PROVIDER=jina` | Jina Rerank API（推荐，免费 1M token/月） | 否 |
| `llm` | `RERANKER_ENABLED=true + RERANKER_PROVIDER=llm` | 主系统 LLM 网关给候选打分 | 否 |
| `local` | `RERANKER_ENABLED=true + RERANKER_PROVIDER=local` | 本地 CrossEncoder (BAAI/bge-reranker-v2-m3, ~568MB) | 是（HF 下载） |

调用方接口（rerank 函数）四者一致；get_reranker() 按 settings 返回对应实现。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.core.config import settings
from app.core.exceptions import RerankerUnavailable
from app.core.logging import logger
from app.models.query import RerankResult

_LOCK = threading.Lock()
_INSTANCE: _Reranker | _DisabledReranker | _LLMReranker | _JinaReranker | None = None


class _Reranker:
    """本地 sentence-transformers CrossEncoder。"""

    provider = "local"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._state = "idle"
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return self._state == "ready"

    def is_failed(self) -> bool:
        return self._state == "failed"

    def ensure_loaded(self, *, wait_seconds: float = 60.0) -> None:
        if self._state == "ready":
            return
        if self._state == "idle":
            self._start_loading()
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise RerankerUnavailable(f"reranker failed to load: {self._error}")
            time.sleep(0.5)
        raise RerankerUnavailable(f"reranker not ready after {wait_seconds}s (state={self._state})")

    def compute_scores_sync(self, query: str, documents: list[str]) -> list[float]:
        self.ensure_loaded()
        try:
            pairs = [(query, doc) for doc in documents]
            scores = self._model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            raise RerankerUnavailable(f"rerank inference failed: {exc}") from exc
        return list(np.asarray(scores, dtype=np.float32).tolist())

    def _start_loading(self) -> None:
        with self._lock:
            if self._state != "idle":
                return
            self._state = "loading"
            thread = threading.Thread(target=self._load_model, name="reranker-load", daemon=True)
            thread.start()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"loading reranker model {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            self._state = "ready"
            logger.info("reranker model ready")
        except Exception as exc:
            self._state = "failed"
            self._error = repr(exc)
            logger.warning(f"reranker load failed: {exc!r}")


class _DisabledReranker:
    """关闭态：所有调用直接抛 RerankerUnavailable，调用方走原 score 排序。"""

    provider = "disabled"

    def __init__(self) -> None:
        self._state = "disabled"

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return False

    def is_failed(self) -> bool:
        return False

    def ensure_loaded(self, *, wait_seconds: float = 0.0) -> None:
        raise RerankerUnavailable("reranker disabled by RERANKER_ENABLED=false")

    def compute_scores_sync(self, query: str, documents: list[str]) -> list[float]:
        raise RerankerUnavailable("reranker disabled")


class _LLMReranker:
    """LLM 重排：让 LLM 给每个 (query, doc) 打 0-10 的分。

    优点：不下载模型；可解释（LLM 输出理由可选）
    缺点：慢（每 batch 一次 API 调用）、贵（按 token 计费）
    """

    provider = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._state = "ready"

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return True

    def is_failed(self) -> bool:
        return False

    def ensure_loaded(self, *, wait_seconds: float = 0.0) -> None:
        return

    async def compute_scores_async(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        prompt = _build_llm_rerank_prompt(query, documents)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a relevance scoring assistant. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise RerankerUnavailable(f"llm rerank api failed: {exc}") from exc

        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return _parse_llm_rerank_scores(text, expected=len(documents))


def _build_llm_rerank_prompt(query: str, documents: list[str]) -> str:
    lines = [f"Query: {query}", "", "Documents:"]
    for i, doc in enumerate(documents):
        # 截断避免 prompt 过长
        snippet = doc[:300].replace("\n", " ")
        lines.append(f"[{i}] {snippet}")
    lines.append("")
    lines.append('Score each document 0-10 by relevance to the query.')
    lines.append('Output ONLY JSON: {"scores": [7.5, 3.0, ...]}')
    return "\n".join(lines)


def _parse_llm_rerank_scores(text: str, expected: int) -> list[float]:
    import re

    match = re.search(r'\{[^{}]*"scores"[^{}]*\}', text)
    if not match:
        return [0.0] * expected
    try:
        payload = json.loads(match.group(0))
        raw_scores = payload.get("scores", [])
    except Exception:
        return [0.0] * expected
    scores = [float(s) for s in raw_scores[:expected]]
    while len(scores) < expected:
        scores.append(0.0)
    return scores


class _JinaReranker:
    """Jina Rerank API（推荐免费方案，注册 https://jina.ai）。

    文档：https://jina.ai/reranker
    免费额度：1M token/月（个人账号）
    模型：jina-reranker-v2-base-multilingual（默认，支持中英）
    """

    provider = "jina"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.jina.ai/v1/rerank",
        model: str = "jina-reranker-v2-base-multilingual",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._state = "ready"

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return True

    def is_failed(self) -> bool:
        return False

    def ensure_loaded(self, *, wait_seconds: float = 0.0) -> None:
        return

    async def compute_scores_async(self, query: str, documents: list[str]) -> list[float]:
        """调 Jina /v1/rerank，返回与 documents 同序的 score 列表（按 relevance 归一化到 0-1）。"""
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RerankerUnavailable(
                f"jina rerank HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise RerankerUnavailable(f"jina rerank api failed: {exc}") from exc

        # Jina 返回 {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        results = data.get("results") or []
        scores = [0.0] * len(documents)
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score") or item.get("score") or 0.0
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = float(score)
        return scores


class _FlashRankReranker:
    """FlashRank ONNX 重排器（推荐 2核4G 部署）。

    特点：
    - ONNX Runtime CPU 推理，无需 GPU
    - 模型小：18-120MB（vs BAAI 568MB）
    - 内存峰值 ~500MB（vs BAAI ~2GB）
    - 首次启动需下载模型，缓存到 RERANKER_FLASHRANK_CACHE_DIR
    """

    provider = "flashrank"

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None
        self._state = "idle"
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def is_ready(self) -> bool:
        return self._state == "ready"

    def is_failed(self) -> bool:
        return self._state == "failed"

    def ensure_loaded(self, *, wait_seconds: float = 60.0) -> None:
        if self._state == "ready":
            return
        if self._state == "idle":
            self._start_loading()
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise RerankerUnavailable(f"flashrank load failed: {self._error}")
            time.sleep(0.3)
        raise RerankerUnavailable(f"flashrank not ready after {wait_seconds}s (state={self._state})")

    def compute_scores_sync(self, query: str, documents: list[str]) -> list[float]:
        self.ensure_loaded()
        try:
            from flashrank import RerankRequest  # type: ignore[import-not-found]

            # FlashRank 期望 passages = [{"text": "..."}] 格式
            passages = [{"text": doc} for doc in documents]
            request = RerankRequest(query=query, passages=passages)
            raw = self._model.rerank(request)
        except Exception as exc:
            raise RerankerUnavailable(f"flashrank inference failed: {exc}") from exc

        # FlashRank 返回 list of {"text": str, "score": float}（按 score 降序）
        # 没有 index 字段，需要按 text 反查原 documents 顺序
        items = raw.results if hasattr(raw, "results") else raw
        # 文本→索引映射（重复文本取首个；score 取最大）
        text_to_idx: dict[str, int] = {}
        for i, doc in enumerate(documents):
            text_to_idx.setdefault(doc, i)
        scores = [0.0] * len(documents)
        for item in items:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
            if text in text_to_idx:
                idx = text_to_idx[text]
                scores[idx] = max(scores[idx], float(score or 0.0))
        return scores

    def _start_loading(self) -> None:
        with self._lock:
            if self._state != "idle":
                return
            self._state = "loading"
            thread = threading.Thread(target=self._load_model, name="flashrank-load", daemon=True)
            thread.start()

    def _load_model(self) -> None:
        try:
            from flashrank import Ranker as _FlashRanker  # type: ignore[import-not-found]

            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"loading flashrank model {self.model_name} (cache={self.cache_dir})")
            self._model = _FlashRanker(model_name=self.model_name, cache_dir=self.cache_dir)
            self._state = "ready"
            logger.info(f"flashrank model ready: {self.model_name}")
        except Exception as exc:
            self._state = "failed"
            self._error = repr(exc)
            logger.warning(f"flashrank load failed: {exc!r}")


def get_reranker() -> _Reranker | _DisabledReranker | _LLMReranker | _JinaReranker | _FlashRankReranker:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = _build_reranker()
    return _INSTANCE


def _build_reranker() -> _Reranker | _DisabledReranker | _LLMReranker | _JinaReranker | _FlashRankReranker:
    if not settings.reranker_enabled:
        logger.info("reranker disabled (RERANKER_ENABLED=false); /rerank will fallback to original order")
        return _DisabledReranker()
    provider = (settings.reranker_provider or "local").lower()
    if provider == "flashrank":
        logger.info(
            f"using flashrank reranker: model={settings.reranker_flashrank_model} "
            f"cache={settings.reranker_flashrank_cache_dir}"
        )
        return _FlashRankReranker(
            model_name=settings.reranker_flashrank_model,
            cache_dir=settings.reranker_flashrank_cache_dir,
        )
    if provider == "jina":
        api_key = settings.reranker_jina_api_key
        if not api_key:
            logger.warning("jina reranker api key missing; fallback to disabled")
            return _DisabledReranker()
        logger.info(
            f"using jina reranker: model={settings.reranker_jina_model} base={settings.reranker_jina_base_url}"
        )
        return _JinaReranker(
            api_key=api_key,
            base_url=settings.reranker_jina_base_url,
            model=settings.reranker_jina_model,
            timeout=float(settings.http_timeout),
        )
    if provider == "llm":
        base_url = settings.reranker_llm_base_url or settings.hyde_llm_base_url
        api_key = settings.reranker_llm_api_key or settings.hyde_llm_api_key
        model = settings.reranker_llm_model or settings.hyde_llm_model
        if not base_url or not api_key or not model:
            logger.warning("llm reranker mis-configured (need base_url/api_key/model); fallback to disabled")
            return _DisabledReranker()
        logger.info(f"using llm reranker: model={model} base={base_url}")
        return _LLMReranker(base_url=base_url, api_key=api_key, model=model, timeout=float(settings.http_timeout))
    # 默认 local
    return _Reranker(settings.reranker_model)


def reset_reranker() -> None:
    """测试用：重置单例。"""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None


async def rerank(
    query: str,
    documents: list[str] | list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[RerankResult]:
    if not documents:
        return []
    reranker = get_reranker()
    doc_strings: list[str] = []
    doc_meta: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        if isinstance(doc, str):
            doc_strings.append(doc)
            doc_meta.append({"original_index": idx})
        elif isinstance(doc, dict):
            text = doc.get("content") or doc.get("text") or ""
            doc_strings.append(text)
            meta = {k: v for k, v in doc.items() if k not in {"content", "text"}}
            meta.setdefault("original_index", idx)
            doc_meta.append(meta)
        else:
            doc_strings.append(str(doc))
            doc_meta.append({"original_index": idx})

    if isinstance(reranker, (_LLMReranker, _JinaReranker)):
        scores = await reranker.compute_scores_async(query, doc_strings)
    else:
        try:
            scores = await _run_in_thread(reranker.compute_scores_sync, query, doc_strings)
        except RerankerUnavailable as exc:
            raise exc

    paired = list(zip(scores, doc_strings, doc_meta, strict=True))
    paired.sort(key=lambda item: item[0], reverse=True)
    paired = paired[:top_k]

    return [
        RerankResult(content=text, score=float(score), original_index=meta.get("original_index", 0), metadata=meta)
        for score, text, meta in paired
    ]


async def _run_in_thread(fn: Any, *args: Any) -> Any:
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        return await loop.run_in_executor(pool, fn, *args)


__all__ = ["get_reranker", "reset_reranker", "rerank"]
