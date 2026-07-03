"""Hybrid 检索：RRF 融合 dense + sparse 召回。

借鉴 airQA retrieval/hybrid_retriever._merge_results 升级点：
1. MinMax 归一化（可选）：融合前对 dense / sparse 各自做 MinMax，避免量纲差异
2. diversity_penalty：同 doc_id 多结果衰减（下限 diversity_floor）

公式：
- 默认（airQA v1 风格）：score = Σ w_i / (k + rank_i)
- 升级（归一化 + diversity）：
    fused = Σ w_i * normalized_score_i
    if same_doc_count > 1: fused *= max(floor, 1 - penalty * (same_doc_count - 1))
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.core.config import settings
from app.models.query import RetrieveResult
from app.retrieval import dense_searcher, sparse_searcher


def fuse(
    dense_results: list[RetrieveResult],
    sparse_results: list[RetrieveResult],
    *,
    top_k: int,
    weights: tuple[float, float] | None = None,
    k: int | None = None,
    use_minmax: bool | None = None,
    diversity_penalty: float | None = None,
    diversity_floor: float | None = None,
) -> list[RetrieveResult]:
    """融合 dense + sparse 召回结果。"""
    vector_weight, sparse_weight = weights or (settings.rrf_vector_weight, settings.rrf_sparse_weight)
    rrf_k = k if k is not None else settings.rrf_k
    do_minmax = settings.normalize_scores_before_fusion if use_minmax is None else use_minmax
    penalty = settings.diversity_penalty if diversity_penalty is None else diversity_penalty
    floor = settings.diversity_floor if diversity_floor is None else diversity_floor

    dense_norm = _minmax([r.score for r in dense_results]) if do_minmax else None
    sparse_norm = _minmax([r.score for r in sparse_results]) if do_minmax else None

    scores: dict[tuple[str, int], dict[str, Any]] = {}

    for rank, result in enumerate(dense_results):
        key = (result.doc_id or "", result.chunk_index)
        bucket = scores.setdefault(key, {"result": result, "score": 0.0})
        if dense_norm is not None:
            bucket["score"] += vector_weight * dense_norm[rank]
        else:
            bucket["score"] += vector_weight / (rrf_k + rank + 1)

    for rank, result in enumerate(sparse_results):
        key = (result.doc_id or "", result.chunk_index)
        bucket = scores.setdefault(key, {"result": result, "score": 0.0})
        if sparse_norm is not None:
            bucket["score"] += sparse_weight * sparse_norm[rank]
        else:
            bucket["score"] += sparse_weight / (rrf_k + rank + 1)

    fused = sorted(scores.values(), key=lambda b: b["score"], reverse=True)

    if penalty > 0:
        fused = _apply_diversity_penalty(fused, penalty=penalty, floor=floor)

    fused = fused[:top_k]
    return [
        RetrieveResult(
            content=bucket["result"].content,
            score=bucket["score"],
            doc_id=bucket["result"].doc_id,
            chunk_index=bucket["result"].chunk_index,
            metadata=bucket["result"].metadata,
        )
        for bucket in fused
    ]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _apply_diversity_penalty(
    fused: list[dict[str, Any]],
    *,
    penalty: float,
    floor: float,
) -> list[dict[str, Any]]:
    """同 doc_id 多结果衰减：第 n 次出现 * max(floor, 1 - penalty * (n-1))。"""
    doc_count: dict[str, int] = defaultdict(int)
    for bucket in fused:
        doc_id = bucket["result"].doc_id or ""
        doc_count[doc_id] += 1
        if doc_id and doc_count[doc_id] > 1:
            factor = max(floor, 1.0 - penalty * (doc_count[doc_id] - 1))
            bucket["score"] *= factor
    return sorted(fused, key=lambda b: b["score"], reverse=True)


async def search(
    *,
    collection: str,
    query: str,
    query_vector: list[float] | None,
    top_k: int,
    filters: dict[str, Any] | None = None,
    recall_k: int | None = None,
) -> list[RetrieveResult]:
    recall_top = recall_k or max(top_k * 2, 20)
    dense_results: list[RetrieveResult] = []
    if query_vector:
        dense_results = dense_searcher.search(
            collection=collection,
            query_vector=query_vector,
            top_k=recall_top,
            filters=filters,
        )
    sparse_results = sparse_searcher.search(
        collection=collection,
        query=query,
        top_k=recall_top,
        filters=filters,
    )
    return fuse(dense_results, sparse_results, top_k=top_k)


def normalize_weights(w_vec: float, w_sparse: float) -> tuple[float, float]:
    total = w_vec + w_sparse
    if total <= 0:
        return settings.rrf_vector_weight, settings.rrf_sparse_weight
    return w_vec / total, w_sparse / total


__all__ = ["fuse", "normalize_weights", "search"]
