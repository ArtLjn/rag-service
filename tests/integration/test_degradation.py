"""降级场景测试。"""

from __future__ import annotations

from unittest.mock import patch

from app.core.exceptions import EmbedderUnavailable, QdrantUnavailable


def test_health_returns_degraded_when_all_components_unavailable(rag_service_client) -> None:
    with patch("app.storage.qdrant_client.get_client") as qd, \
         patch("app.retrieval.embedder.get_embedder") as ed, \
         patch("app.retrieval.reranker.get_reranker") as rr:
        qd.side_effect = RuntimeError("offline")
        ed.side_effect = RuntimeError("offline")
        rr.side_effect = RuntimeError("offline")
        response = rag_service_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["qdrant"] == "unavailable"
    assert body["components"]["embedder"] == "unavailable"
    assert body["components"]["reranker"] == "unavailable"


def test_retrieve_returns_503_when_qdrant_down(rag_service_client, mocked_embedder) -> None:
    from app.retrieval import dense_searcher, sparse_searcher

    def boom(**_):
        raise QdrantUnavailable("down")

    with patch.object(sparse_searcher, "search", boom), \
         patch.object(dense_searcher, "search", boom):
        response = rag_service_client.post(
            "/retrieve",
            json={"query": "q", "collection": "c1", "mode": "hybrid"},
        )
    assert response.status_code == 503
    assert response.json()["error_code"] == "QDRANT_UNAVAILABLE"


def test_retrieve_vector_fallbacks_to_bm25_when_embedder_unavailable(rag_service_client) -> None:
    from app.retrieval import sparse_searcher
    from app.services import retrieve_service

    async def boom_embed(self, texts):
        raise EmbedderUnavailable("no model")

    fake_bad = type(
        "Bad",
        (),
        {
            "embed": boom_embed,
            "embed_query": lambda self, t: (_ for _ in ()).throw(EmbedderUnavailable("no model")),
        },
    )()

    with patch.object(retrieve_service, "get_embedder", return_value=fake_bad), \
         patch.object(sparse_searcher, "search", lambda **kw: []):
        response = rag_service_client.post(
            "/retrieve",
            json={"query": "q", "collection": "c1", "mode": "vector"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["actual_mode"] == "bm25"
    assert "vector_to_bm25_fallback" in body["warning"]


def test_rerank_returns_warning_when_model_unavailable(rag_service_client) -> None:
    from app.core.exceptions import RerankerUnavailable

    def boom(*_, **__):
        raise RerankerUnavailable("no model")

    fake_bad = type("Bad", (), {"compute_scores_sync": boom})()
    with patch("app.retrieval.reranker.get_reranker", return_value=fake_bad):
        response = rag_service_client.post(
            "/rerank",
            json={"query": "q", "documents": ["a", "b"], "top_k": 2},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["warning"] == "reranker_degraded"
    assert len(body["data"]["results"]) == 2
