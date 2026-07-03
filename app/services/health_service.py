"""健康检查服务：聚合 Qdrant / Embedder / Reranker 三组件状态。"""

from __future__ import annotations

from app.core.logging import logger
from app.core.response import HealthResponse


async def check_health() -> HealthResponse:
    components: dict[str, str] = {}

    components["qdrant"] = await _check_qdrant()
    components["embedder"] = await _check_embedder()
    components["reranker"] = await _check_reranker()

    # reranker=disabled 不计入 degraded（用户主动关闭）
    degraded = [
        name
        for name, status in components.items()
        if status not in {"ok", "disabled"}
    ]
    status = "degraded" if degraded else "ok"
    warning = f"degraded components: {', '.join(degraded)}" if degraded else None
    if warning:
        logger.warning(f"rag-service health degraded: {components}")
    return HealthResponse(status=status, components=components, warning=warning)


async def _check_qdrant() -> str:
    try:
        from app.storage.qdrant_client import get_client

        client = get_client()
        client.get_collections()
        return "ok"
    except Exception as exc:
        logger.debug(f"qdrant health check failed: {exc!r}")
        return "unavailable"


async def _check_embedder() -> str:
    try:
        from app.retrieval.embedder import get_embedder

        embedder = get_embedder()
        if embedder.is_ready():
            return "ok"
        return "loading"
    except Exception as exc:
        logger.debug(f"embedder health check failed: {exc!r}")
        return "unavailable"


async def _check_reranker() -> str:
    """reranker 状态：disabled（用户关）/ loading / ok / unavailable。"""
    try:
        from app.retrieval.reranker import get_reranker

        reranker = get_reranker()
        if getattr(reranker, "provider", "") == "disabled":
            return "disabled"
        if reranker.is_ready():
            return "ok"
        return "loading"
    except Exception as exc:
        logger.debug(f"reranker health check failed: {exc!r}")
        return "unavailable"
