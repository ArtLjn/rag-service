"""重排服务：调用 reranker 模块，模型不可用时降级。"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import RerankerUnavailable
from app.core.logging import logger
from app.models.query import RerankResult
from app.retrieval.reranker import rerank as _rerank


async def rerank_documents(
    *,
    query: str,
    documents: list[str] | list[dict[str, Any]],
    top_k: int = 5,
) -> tuple[list[RerankResult], str | None]:
    try:
        results = await _rerank(query, documents, top_k=top_k)
        return results, None
    except RerankerUnavailable as exc:
        logger.warning(f"reranker unavailable, degrading by original score: {exc.message}")
        fallback = _fallback_by_order(query, documents, top_k)
        return fallback, "reranker_degraded"


def _fallback_by_order(
    query: str,
    documents: list[str] | list[dict[str, Any]],
    top_k: int,
) -> list[RerankResult]:
    """无模型时按原顺序返回（score=1.0 - rank*0.001 单调递减）。"""
    results: list[RerankResult] = []
    for idx, doc in enumerate(documents[:top_k]):
        if isinstance(doc, str):
            content = doc
            meta: dict[str, Any] = {"original_index": idx}
        else:
            content = doc.get("content") or doc.get("text") or ""
            meta = {k: v for k, v in doc.items() if k not in {"content", "text"}}
            meta.setdefault("original_index", idx)
        results.append(
            RerankResult(
                content=content,
                score=max(0.0, 1.0 - idx * 0.001),
                original_index=meta.get("original_index", idx),
                metadata=meta,
            )
        )
    return results


__all__ = ["rerank_documents"]
