"""/rerank API 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_rerank_returns_sorted_documents(client: TestClient) -> None:
    from app.models.query import RerankResult

    async def fake_rerank(**kwargs):
        return [
            RerankResult(content="best", score=0.95, original_index=1),
            RerankResult(content="ok", score=0.4, original_index=0),
        ], None

    with patch("app.api.rerank.rerank_documents", new=AsyncMock(side_effect=fake_rerank)):
        response = client.post(
            "/rerank",
            json={"query": "q", "documents": ["ok", "best"], "top_k": 2},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["results"][0]["content"] == "best"


def test_rerank_returns_warning_when_degraded(client: TestClient) -> None:
    from app.models.query import RerankResult

    async def fake_rerank(**kwargs):
        return [RerankResult(content="ok", score=1.0, original_index=0)], "reranker_degraded"

    with patch("app.api.rerank.rerank_documents", new=AsyncMock(side_effect=fake_rerank)):
        response = client.post(
            "/rerank",
            json={"query": "q", "documents": ["ok"], "top_k": 1},
        )
    body = response.json()
    assert body["warning"] == "reranker_degraded"


def test_rerank_empty_documents_returns_empty(client: TestClient) -> None:
    response = client.post("/rerank", json={"query": "q", "documents": [], "top_k": 5})
    assert response.status_code == 200
    assert response.json()["data"]["results"] == []
