"""集成测试 fixtures：rag_service_client 提供指向测试实例的 TestClient。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def rag_service_client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture()
def mocked_qdrant() -> MagicMock:
    """模拟 Qdrant 客户端可用：collection_exists 返回 True，upsert 成功。"""
    fake = MagicMock()
    fake.collection_exists.return_value = True
    fake.get_collections.return_value = MagicMock(collections=[])
    fake.upsert.return_value = None
    fake.delete_collection.return_value = None
    fake.scroll.return_value = ([], None)
    import app.storage.collection_manager as cm
    import app.storage.qdrant_client as qc

    with patch.object(qc, "get_client", return_value=fake), \
         patch.object(cm, "get_client", return_value=fake):
        yield fake


@pytest.fixture()
def mocked_embedder() -> Any:
    """模拟 embedder 已加载、embed 返回固定向量。"""
    import app.retrieval.embedder as emb_mod
    import app.services.retrieve_service as rs

    class _Fake:
        state = "ready"

        def is_ready(self) -> bool:
            return True

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 8

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 8 for _ in texts]

    fake = _Fake()
    with patch.object(emb_mod, "get_embedder", return_value=fake), \
         patch.object(rs, "get_embedder", return_value=fake):
        yield fake


@pytest.fixture()
def mocked_reranker() -> Any:
    """模拟 reranker 模型加载好。"""
    import app.retrieval.reranker as rk

    class _Fake:
        state = "ready"

        def is_ready(self) -> bool:
            return True

        def compute_scores_sync(self, query: str, docs: list[str]) -> list[float]:
            return [0.5 + 0.01 * i for i in range(len(docs))]

    fake = _Fake()
    with patch.object(rk, "get_reranker", return_value=fake):
        yield fake
