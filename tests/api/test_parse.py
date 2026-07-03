"""/parse 与 /health API 测试。"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_health_returns_degraded_without_qdrant(client: TestClient) -> None:
    with patch("app.storage.qdrant_client.get_client") as mock_qdrant, \
         patch("app.retrieval.embedder.get_embedder") as mock_embedder, \
         patch("app.retrieval.reranker.get_reranker") as mock_reranker:
        mock_qdrant.side_effect = RuntimeError("offline")
        mock_embedder.side_effect = RuntimeError("offline")
        mock_reranker.side_effect = RuntimeError("offline")
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["qdrant"] == "unavailable"


def test_parse_text_returns_chunks_and_summary(client: TestClient, sample_text: str) -> None:
    response = client.post(
        "/parse",
        data={"text": sample_text, "strategy": "fixed", "chunk_size": "100", "chunk_overlap": "10"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "OK"
    assert body["data"]["layout_summary"]["total"] >= 1
    assert all("content" in c for c in body["data"]["chunks"])


def test_parse_markdown_extracts_titles(client: TestClient, sample_markdown: str) -> None:
    response = client.post(
        "/parse",
        data={"text": sample_markdown, "file_type": "md", "strategy": "structure_aware"},
    )
    assert response.status_code == 200
    body = response.json()
    categories = {c["metadata"]["category"] for c in body["data"]["chunks"]}
    assert "title" in categories


def test_parse_rejects_missing_input(client: TestClient) -> None:
    response = client.post("/parse", data={})
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "UNSUPPORTED_FORMAT"
