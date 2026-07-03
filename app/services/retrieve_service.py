"""检索服务编排：dense / sparse / hybrid 三模式 + HyDE 改写 + 降级 + 去重。"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import EmbedderUnavailable, InvalidMode, QdrantUnavailable
from app.core.logging import logger
from app.models.query import RetrieveMode, RetrieveResult
from app.retrieval import dense_searcher, hybrid_searcher, sparse_searcher
from app.retrieval.dedup import dedup as dedup_results
from app.retrieval.embedder import get_embedder
from app.retrieval.hyde import maybe_rewrite


async def retrieve(
    *,
    query: str,
    collection: str,
    mode: RetrieveMode | str,
    top_k: int,
    filters: dict[str, Any] | None = None,
    use_hyde: bool = False,
) -> tuple[list[RetrieveResult], str | None, RetrieveMode]:
    if isinstance(mode, str):
        mode = RetrieveMode(mode)

    effective_query, hyde_warning = await maybe_rewrite(query, use_hyde=use_hyde)

    try:
        if mode == RetrieveMode.VECTOR:
            results, warning, actual_mode = await _retrieve_vector(collection, effective_query, top_k, filters, hyde_warning, mode)
        elif mode == RetrieveMode.BM25:
            results, warning, actual_mode = _retrieve_bm25(collection, effective_query, top_k, filters, hyde_warning, mode)
        elif mode == RetrieveMode.HYBRID:
            results, warning, actual_mode = await _retrieve_hybrid(collection, effective_query, top_k, filters, hyde_warning, mode)
        else:
            raise InvalidMode(f"unknown mode: {mode}")
    except QdrantUnavailable:
        raise
    except EmbedderUnavailable as exc:
        if mode == RetrieveMode.VECTOR:
            logger.warning(f"vector mode embedder unavailable, fallback to bm25: {exc.message}")
            results, warning = _retrieve_bm25_raw(collection, effective_query, top_k, filters)
            results = _maybe_dedup(results)
            return results, _combine_warnings(hyde_warning, warning, "vector_to_bm25_fallback"), RetrieveMode.BM25
        if mode == RetrieveMode.HYBRID:
            logger.warning(f"hybrid mode embedder unavailable, fallback to bm25: {exc.message}")
            results, warning = _retrieve_bm25_raw(collection, effective_query, top_k, filters)
            results = _maybe_dedup(results)
            return results, _combine_warnings(hyde_warning, warning, "hybrid_to_bm25_fallback"), RetrieveMode.BM25
        raise

    results = _maybe_dedup(results)
    return results, warning, actual_mode


def _maybe_dedup(results: list[RetrieveResult]) -> list[RetrieveResult]:
    """Jaccard 去重；若实际去掉了一些结果，添加 warning 标记。"""
    deduped = dedup_results(results)
    if len(deduped) < len(results):
        logger.debug(f"dedup removed {len(results) - len(deduped)} duplicate results")
    return deduped


async def _retrieve_vector(
    collection: str,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    hyde_warning: str | None,
    original_mode: RetrieveMode,
) -> tuple[list[RetrieveResult], str | None, RetrieveMode]:
    embedder = get_embedder()
    query_vectors = await embedder.embed([query])
    query_vector = query_vectors[0] if query_vectors else []
    results = dense_searcher.search(
        collection=collection,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
    )
    return results, hyde_warning, original_mode


def _retrieve_bm25(
    collection: str,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    hyde_warning: str | None,
    original_mode: RetrieveMode,
) -> tuple[list[RetrieveResult], str | None, RetrieveMode]:
    results, warning = _retrieve_bm25_raw(collection, query, top_k, filters)
    return results, _combine_warnings(hyde_warning, warning), original_mode


def _retrieve_bm25_raw(
    collection: str,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> tuple[list[RetrieveResult], str | None]:
    results = sparse_searcher.search(
        collection=collection,
        query=query,
        top_k=top_k,
        filters=filters,
    )
    return results, None


async def _retrieve_hybrid(
    collection: str,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    hyde_warning: str | None,
    original_mode: RetrieveMode,
) -> tuple[list[RetrieveResult], str | None, RetrieveMode]:
    try:
        embedder = get_embedder()
        query_vectors = await embedder.embed([query])
        query_vector = query_vectors[0] if query_vectors else []
    except EmbedderUnavailable as exc:
        logger.warning(f"hybrid embedder unavailable, fallback to bm25: {exc.message}")
        results, warning = _retrieve_bm25_raw(collection, query, top_k, filters)
        return results, _combine_warnings(hyde_warning, warning, "hybrid_to_bm25_fallback"), RetrieveMode.BM25

    results = await hybrid_searcher.search(
        collection=collection,
        query=query,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
    )
    return results, hyde_warning, original_mode


def _combine_warnings(*parts: str | None) -> str | None:
    cleaned = [p for p in parts if p]
    return ",".join(cleaned) if cleaned else None


__all__ = ["retrieve"]
