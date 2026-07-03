"""/retrieve API 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.chunk import ChunkMetadata
from app.models.query import RetrieveMode, RetrieveResult


@pytest.fixture()
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def _result(doc_id: str, idx: int, score: float) -> RetrieveResult:
    return RetrieveResult(
        content=f"content-{doc_id}-{idx}",
        score=score,
        doc_id=doc_id,
        chunk_index=idx,
        metadata=ChunkMetadata(doc_id=doc_id, chunk_index=idx),
    )


def test_retrieve_returns_hybrid_results(client: TestClient) -> None:
    async def fake_retrieve(**kwargs):
        return [_result("doc", 0, 0.91), _result("doc", 1, 0.45)], None, RetrieveMode(kwargs["mode"])

    with patch("app.api.retrieve.retrieve", new=AsyncMock(side_effect=fake_retrieve)):
        response = client.post(
            "/retrieve",
            json={"query": "hello", "collection": "c1", "mode": "hybrid"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["actual_mode"] == "hybrid"
    assert len(body["data"]["results"]) == 2


def test_retrieve_invalid_mode_returns_400(client: TestClient) -> None:
    response = client.post(
        "/retrieve",
        json={"query": "q", "collection": "c1", "mode": "fancy"},
    )
    assert response.status_code == 422


def test_retrieve_vector_fallback_to_bm25(client: TestClient) -> None:
    async def fake_retrieve(**kwargs):
        return [_result("doc", 0, 0.5)], "vector_to_bm25_fallback", RetrieveMode.BM25

    with patch("app.api.retrieve.retrieve", new=AsyncMock(side_effect=fake_retrieve)):
        response = client.post(
            "/retrieve",
            json={"query": "q", "collection": "c1", "mode": "vector"},
        )
    body = response.json()
    assert body["data"]["actual_mode"] == "bm25"
    assert "vector_to_bm25_fallback" in body["warning"]


def test_retrieve_filters_passed_through(client: TestClient) -> None:
    captured: dict = {}

    async def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [], None, RetrieveMode(kwargs["mode"])

    with patch("app.api.retrieve.retrieve", new=AsyncMock(side_effect=fake_retrieve)):
        response = client.post(
            "/retrieve",
            json={
                "query": "q",
                "collection": "c1",
                "mode": "bm25",
                "filters": {"category": "technical"},
                "use_hyde": True,
            },
        )
    assert response.status_code == 200
    assert captured["filters"] == {"category": "technical"}
    assert captured["use_hyde"] is True
