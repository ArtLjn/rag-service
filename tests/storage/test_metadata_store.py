"""metadata_store 与 version_manager 单测（基于临时 sqlite 文件）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.models.document import DocumentRecord
from app.storage.metadata_store import MetadataStore
from app.storage.version_manager import VersionManager


@pytest.fixture()
def store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(db_path=str(tmp_path / "meta.db"))
    s.init_schema()
    return s


def _record(doc_id: str = "doc1", collection: str = "c1", content_hash: str = "abc") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        collection=collection,
        source="test.pdf",
        category="technical",
        chunk_count=10,
        content_hash=content_hash,
        extra={"file_type": "pdf"},
        ingested_at=datetime.utcnow(),
    )


def test_init_schema_is_idempotent(store: MetadataStore) -> None:
    store.init_schema()
    store.init_schema()


def test_upsert_and_get_document(store: MetadataStore) -> None:
    record = _record()
    store.upsert_document(record)
    fetched = store.get_document("doc1", "c1")
    assert fetched is not None
    assert fetched.chunk_count == 10
    assert fetched.extra == {"file_type": "pdf"}


def test_upsert_replaces_on_conflict(store: MetadataStore) -> None:
    store.upsert_document(_record(content_hash="h1"))
    updated = _record(content_hash="h2")
    updated.chunk_count = 20
    store.upsert_document(updated)
    fetched = store.get_document("doc1", "c1")
    assert fetched is not None
    assert fetched.content_hash == "h2"
    assert fetched.chunk_count == 20


def test_list_documents_paginates(store: MetadataStore) -> None:
    for i in range(15):
        store.upsert_document(_record(doc_id=f"doc{i:02d}"))
    total, page = store.list_documents("c1", page=2, page_size=10)
    assert total == 15
    assert len(page) == 5


def test_delete_document(store: MetadataStore) -> None:
    store.upsert_document(_record())
    assert store.delete_document("doc1", "c1") is True
    assert store.get_document("doc1", "c1") is None
    assert store.delete_document("doc1", "c1") is False


def test_delete_collection_documents(store: MetadataStore) -> None:
    store.upsert_document(_record(doc_id="a"))
    store.upsert_document(_record(doc_id="b"))
    assert store.delete_collection_documents("c1") == 2


def test_version_manager_records_history(store: MetadataStore) -> None:
    vm = VersionManager(store)
    record = _record()
    store.upsert_document(record)
    vm.record_ingest(record)
    record.content_hash = "h2"
    store.upsert_document(record)
    vm.record_ingest(record, previous_hash="abc")

    versions = vm.list_versions("doc1", "c1")
    assert len(versions) == 2
    summary = vm.diff_summary("doc1", "c1")
    assert summary["count"] == 2
