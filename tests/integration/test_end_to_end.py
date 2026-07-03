"""端到端集成测试：/parse → /ingest → /retrieve → /rerank 全流程。"""

from __future__ import annotations

from unittest.mock import patch


def test_full_pipeline_with_mocked_dependencies(
    rag_service_client,
    mocked_qdrant,
    mocked_embedder,
    mocked_reranker,
    sample_markdown,
    tmp_path,
) -> None:
    from app.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=str(tmp_path / "e2e.db"))
    store.init_schema()

    with patch("app.services.ingest_service.MetadataStore", return_value=store), \
         patch("app.services.collection_service.MetadataStore", return_value=store), \
         patch("app.services.ingest_service.ensure_collection_or_raise", return_value=None), \
         patch("app.services.ingest_service.delete_document_points", return_value=0), \
         patch("app.services.ingest_service.get_client") as mock_get_client, \
         patch("app.services.retrieve_service.sparse_searcher.search", lambda **kw: []), \
         patch("app.services.retrieve_service.dense_searcher.search", lambda **kw: []):
        mock_get_client.return_value.upsert.return_value = None

        parse_resp = rag_service_client.post(
            "/parse",
            data={"text": sample_markdown, "file_type": "md"},
        )
        assert parse_resp.status_code == 200
        parsed = parse_resp.json()["data"]
        assert parsed["layout_summary"]["total"] >= 1

        ingest_resp = rag_service_client.post(
            "/ingest",
            data={
                "collection": "ticket_knowledge",
                "text": sample_markdown,
                "file_type": "md",
                "strategy": "structure_aware",
            },
        )
        assert ingest_resp.status_code == 200
        ingested = ingest_resp.json()["data"]
        assert ingested["chunk_count"] >= 1
        assert ingested["collection"] == "ticket_knowledge"

        retrieve_resp = rag_service_client.post(
            "/retrieve",
            json={
                "query": "什么是项目说明",
                "collection": "ticket_knowledge",
                "mode": "hybrid",
                "top_k": 5,
            },
        )
        assert retrieve_resp.status_code == 200
        body = retrieve_resp.json()["data"]
        assert "results" in body
        assert body["actual_mode"] in {"hybrid", "bm25", "vector"}

        rerank_resp = rag_service_client.post(
            "/rerank",
            json={
                "query": "什么是项目说明",
                "documents": ["项目说明", "安装步骤"],
                "top_k": 2,
            },
        )
        assert rerank_resp.status_code == 200
        reranked = rerank_resp.json()["data"]["results"]
        assert len(reranked) == 2
        assert reranked[0]["score"] >= reranked[1]["score"]


def test_collections_crud_round_trip(rag_service_client, mocked_qdrant, tmp_path) -> None:
    from app.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=str(tmp_path / "crud.db"))
    store.init_schema()

    with patch("app.services.collection_service.MetadataStore", return_value=store):
        create_resp = rag_service_client.post(
            "/collections", json={"name": "c1", "vector_dim": 1024, "distance": "Cosine"}
        )
        assert create_resp.status_code == 200

        list_resp = rag_service_client.get("/collections")
        assert list_resp.status_code == 200

        delete_resp = rag_service_client.delete("/collections/c1")
        assert delete_resp.status_code == 200
