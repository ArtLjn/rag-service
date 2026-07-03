"""GET /health — Qdrant / Embedder / Reranker 三组件状态检查。

任一关键组件不可用则 status=degraded（HTTP 200，由主系统决定是否走降级分支）。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app.services.health_service import check_health

    return await check_health()
