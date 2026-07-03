"""POST /retrieve — vector / bm25 / hybrid 三模式检索。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.core.metrics import metrics
from app.core.response import ApiResponse
from app.models.query import RetrieveMode
from app.services.retrieve_service import retrieve

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


class RetrieveBody(BaseModel):
    query: str
    collection: str
    mode: RetrieveMode = RetrieveMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    use_hyde: bool = False


@router.post("")
async def retrieve_endpoint(body: RetrieveBody) -> ApiResponse[dict[str, Any]]:
    with metrics.time("retrieve_latency_seconds", "retrieve endpoint latency"):
        results, warning, actual_mode = await retrieve(
            query=body.query,
            collection=body.collection,
            mode=body.mode,
            top_k=body.top_k,
            filters=body.filters or None,
            use_hyde=body.use_hyde,
        )
    metrics.counter("retrieve_total").inc()
    metrics.counter(f"retrieve_mode_{actual_mode.value}").inc()
    logger.info(
        f"/retrieve mode={body.mode.value}→{actual_mode.value} hits={len(results)} warning={warning}"
    )

    return ApiResponse.ok(
        {
            "results": [r.model_dump(mode="json") for r in results],
            "actual_mode": actual_mode.value,
            "query_vector_dim": _safe_query_vector_dim(),
        },
        warning=warning,
    )


def _safe_query_vector_dim() -> int | None:
    try:
        from app.retrieval.embedder import get_embedder

        embedder = get_embedder()
        if embedder.is_ready():
            return embedder._infer_dim()  # noqa: SLF001
    except Exception:
        return None
    return None
