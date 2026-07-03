"""/ingest API 测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_ingest_returns_404_when_collection_missing(client: TestClient, sample_text: str) -> None:
    with patch("app.services.ingest_service.ensure_collection_or_raise") as mock:
        from app.core.exceptions import CollectionNotFound

        mock.side_effect = CollectionNotFound("not found")
        response = client.post(
            "/ingest",
            data={"collection": "missing", "text": sample_text},
        )
    assert response.status_code == 404
    assert response.json()["error_code"] == "COLLECTION_NOT_FOUND"


def test_ingest_writes_through_pipeline(client: TestClient, sample_text: str) -> None:
    async def fake_ingest(**kwargs):
        return {"doc_id": "fake", "chunk_count": 3, "collection": kwargs["collection"], "action": "created"}

    with patch("app.api.ingest.ingest_content", new=AsyncMock(side_effect=fake_ingest)):
        response = client.post(
            "/ingest",
            data={"collection": "c1", "text": sample_text, "strategy": "fixed"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["chunk_count"] == 3
    assert body["data"]["doc_id"] == "fake"


def test_ingest_rejects_missing_collection(client: TestClient, sample_text: str) -> None:
    response = client.post("/ingest", data={"text": sample_text})
    assert response.status_code == 422


def test_ingest_503_when_qdrant_unavailable(client: TestClient, sample_text: str) -> None:
    from app.core.exceptions import QdrantUnavailable

    with patch("app.services.ingest_service.ensure_collection_or_raise") as mock:
        mock.side_effect = QdrantUnavailable("down")
        response = client.post(
            "/ingest",
            data={"collection": "c1", "text": sample_text},
        )
    assert response.status_code == 503
    assert response.json()["error_code"] == "QDRANT_UNAVAILABLE"
