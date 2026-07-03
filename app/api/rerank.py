"""POST /rerank — Cross-Encoder 精排。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.core.metrics import metrics
from app.core.response import ApiResponse
from app.services.rerank_service import rerank_documents

router = APIRouter(prefix="/rerank", tags=["rerank"])


class RerankBody(BaseModel):
    query: str
    documents: list[Any]
    top_k: int = Field(default=5, ge=1, le=100)
    model: str | None = None


@router.post("")
async def rerank_endpoint(body: RerankBody) -> ApiResponse[dict[str, Any]]:
    with metrics.time("rerank_latency_seconds", "rerank endpoint latency"):
        results, warning = await rerank_documents(
            query=body.query,
            documents=body.documents,
            top_k=body.top_k,
        )
    metrics.counter("rerank_total").inc()
    logger.info(f"/rerank query='{body.query[:20]}...' in={len(body.documents)} out={len(results)} warning={warning}")

    return ApiResponse.ok(
        {
            "results": [r.model_dump(mode="json") for r in results],
        },
        warning=warning,
    )
