"""collection 管理：创建 / 删除 / 列表 / 存在性检查。

约定：
- 每个 collection 同时配置 dense 向量（Cosine）与 sparse 向量（BM25）
- sparse 字段名固定 `text-sparse`，dense 字段名 `dense`
- 创建时如已存在，幂等返回（不报错）
- 默认向量维度对齐 settings.embedding_dim（与主系统 Embedding 模型一致）
"""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.exceptions import CollectionNotFound, QdrantUnavailable
from app.core.logging import logger
from app.storage.qdrant_client import get_client

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "text-sparse"


def _default_vector_dim() -> int:
    return settings.embedding_dim


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise QdrantUnavailable(f"qdrant operation failed: {exc}") from exc


def _resolve_distance(distance: str) -> qmodels.Distance:
    normalized = distance.upper()
    if normalized == "COS":
        normalized = "COSINE"
    return qmodels.Distance[normalized]


def create_collection(
    name: str,
    vector_dim: int | None = None,
    distance: str = "Cosine",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    client = get_client()
    dim = vector_dim or _default_vector_dim()

    exists = _safe_call(client.collection_exists, name)
    if exists:
        if overwrite:
            logger.info(f"collection {name} exists, recreating")
            _safe_call(client.delete_collection, name)
        else:
            logger.info(f"collection {name} already exists, idempotent")
            return {"collection": name, "action": "noop", "vector_dim": dim}

    vector_dim = dim
    distance_enum = _resolve_distance(distance)
    vectors_config = {
        DENSE_VECTOR_NAME: qmodels.VectorParams(size=vector_dim, distance=distance_enum),
    }
    sparse_vectors_config = {
        SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
            index=qmodels.SparseIndexParams(on_disk=False),
        ),
    }

    _safe_call(
        client.create_collection,
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    logger.info(f"collection {name} created vector_dim={vector_dim} distance={distance}")
    return {
        "collection": name,
        "action": "created",
        "vector_dim": vector_dim,
        "distance": distance,
    }


def delete_collection(name: str) -> dict[str, Any]:
    client = get_client()
    exists = _safe_call(client.collection_exists, name)
    if not exists:
        raise CollectionNotFound(f"collection {name} not found")
    _safe_call(client.delete_collection, name)
    logger.info(f"collection {name} deleted")
    return {"collection": name, "action": "deleted"}


def list_collections() -> list[dict[str, Any]]:
    client = get_client()
    response = _safe_call(client.get_collections)
    result: list[dict[str, Any]] = []
    for entry in response.collections:
        info = _safe_call(client.get_collection, entry.name)
        result.append(
            {
                "name": entry.name,
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
                "status": str(info.status) if info.status else None,
            }
        )
    return result


def collection_exists(name: str) -> bool:
    client = get_client()
    return bool(_safe_call(client.collection_exists, name))


def ensure_collection_or_raise(name: str) -> None:
    if not collection_exists(name):
        raise CollectionNotFound(f"collection {name} not found")


def delete_document_points(collection: str, doc_id: str) -> int:
    """删除指定 doc_id 的所有 point。返回删除数量估计（points 数）。"""
    client = get_client()
    ensure_collection_or_raise(collection)
    info = _safe_call(
        client.scroll,
        collection_name=collection,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )
    points, _ = info
    point_ids = [p.id for p in points]
    if point_ids:
        _safe_call(client.delete, collection_name=collection, points_selector=qmodels.PointIdsList(points=point_ids))
    return len(point_ids)


__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "collection_exists",
    "create_collection",
    "delete_collection",
    "delete_document_points",
    "ensure_collection_or_raise",
    "list_collections",
]
