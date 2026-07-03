"""Dense 向量检索（Qdrant）。"""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.exceptions import QdrantUnavailable
from app.core.logging import logger
from app.models.chunk import ChunkMetadata
from app.models.query import RetrieveResult
from app.storage.collection_manager import DENSE_VECTOR_NAME
from app.storage.qdrant_client import get_client


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise QdrantUnavailable(f"dense search failed: {exc}") from exc


def search(
    *,
    collection: str,
    query_vector: list[float],
    top_k: int,
    score_threshold: float | None = None,
    filters: dict[str, Any] | None = None,
) -> list[RetrieveResult]:
    client = get_client()
    threshold = settings.score_threshold if score_threshold is None else score_threshold
    qdrant_filter = _build_filter(filters)

    try:
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            score_threshold=threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points
    except Exception as exc:
        raise QdrantUnavailable(f"dense search failed: {exc}") from exc

    return _hits_to_results(hits)


def _build_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions: list[qmodels.FieldCondition] = []
    must_dict = filters.get("must", {}) if "must" in filters else filters
    if isinstance(must_dict, dict):
        for key, value in must_dict.items():
            conditions.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def _hits_to_results(hits: list[Any]) -> list[RetrieveResult]:
    results: list[RetrieveResult] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        content = payload.pop("content", "") if isinstance(payload, dict) else ""
        metadata = ChunkMetadata.from_qdrant_payload(payload if isinstance(payload, dict) else {})
        results.append(
            RetrieveResult(
                content=content,
                score=float(getattr(hit, "score", 0.0)),
                doc_id=metadata.doc_id,
                chunk_index=metadata.chunk_index,
                metadata=metadata,
            )
        )
    logger.debug(f"dense search returned {len(results)} hits")
    return results


__all__ = ["search"]
