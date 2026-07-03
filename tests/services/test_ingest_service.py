"""ingest_service 单测：mock Qdrant + embedder + store，验证 happy path、增量、删除。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services import ingest_service
from app.storage.metadata_store import MetadataStore


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture()
def store(tmp_path):
    s = MetadataStore(db_path=str(tmp_path / "ingest.db"))
    s.init_schema()
    return s


def test_ingest_writes_chunks_and_metadata(store: MetadataStore) -> None:
    upsert_calls: list = []

    fake_client = SimpleNamespace(
        upsert=lambda **kwargs: upsert_calls.append(kwargs) or None,
    )

    with patch("app.services.ingest_service.ensure_collection_or_raise", return_value=None), \
         patch("app.services.ingest_service.get_client", return_value=fake_client), \
         patch("app.services.ingest_service.delete_document_points", return_value=0):
        result = asyncio.get_event_loop().run_until_complete(
            ingest_service.ingest_content(
                content="# 标题\n\n段落内容一。\n\n段落内容二。",
                collection="c1",
                file_type="md",
                strategy="structure_aware",
                metadata_store=store,
                embedder=FakeEmbedder(),
            )
        )

    assert result["chunk_count"] >= 1
    assert result["collection"] == "c1"
    assert upsert_calls
    record = store.get_document(result["doc_id"], "c1")
    assert record is not None
    assert record.chunk_count == result["chunk_count"]


def test_ingest_skips_when_content_hash_unchanged(store: MetadataStore) -> None:
    from datetime import datetime

    from app.models.document import DocumentRecord
    from app.services.parse_service import compute_doc_id

    content = "same"
    doc_id = compute_doc_id(content)
    record = DocumentRecord(
        doc_id=doc_id,
        collection="c1",
        chunk_count=3,
        content_hash=hashlib_md5(content),
        ingested_at=datetime.utcnow(),
    )
    store.upsert_document(record)

    with patch("app.services.ingest_service.ensure_collection_or_raise", return_value=None):
        result = asyncio.get_event_loop().run_until_complete(
            ingest_service.ingest_content(
                content=content,
                collection="c1",
                file_type="txt",
                metadata_store=store,
                embedder=FakeEmbedder(),
            )
        )

    assert result["action"] == "noop"
    assert result["doc_id"] == doc_id


def test_ingest_503_when_qdrant_unavailable(store: MetadataStore) -> None:
    from app.core.exceptions import QdrantUnavailable

    with patch("app.services.ingest_service.ensure_collection_or_raise") as mock:
        mock.side_effect = QdrantUnavailable("down")
        try:
            asyncio.get_event_loop().run_until_complete(
                ingest_service.ingest_content(
                    content="text",
                    collection="c1",
                    file_type="txt",
                    metadata_store=store,
                    embedder=FakeEmbedder(),
                )
            )
        except QdrantUnavailable as exc:
            assert exc.error_code == "QDRANT_UNAVAILABLE"
        else:
            raise AssertionError("expected QdrantUnavailable")


def hashlib_md5(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()
