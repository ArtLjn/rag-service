"""版本管理器：每次 ingest 写入一条 document_version 记录。

毕设范围内不做可视化回滚 UI，仅保留历史记录。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from app.models.document import DocumentRecord, DocumentVersion
from app.storage.metadata_store import MetadataStore


class VersionManager:
    def __init__(self, store: MetadataStore | None = None) -> None:
        self.store = store or MetadataStore()

    def record_ingest(
        self,
        record: DocumentRecord,
        *,
        note: str | None = None,
        previous_hash: str | None = None,
    ) -> DocumentVersion:
        version = DocumentVersion(
            version_id=uuid.uuid4().hex[:12],
            doc_id=record.doc_id,
            collection=record.collection,
            content_hash=record.content_hash,
            chunk_count=record.chunk_count,
            created_at=datetime.utcnow(),
            note=note or (f"updated from {previous_hash[:8]}" if previous_hash else "initial"),
        )
        self.store.add_version(version)
        return version

    def list_versions(self, doc_id: str, collection: str) -> list[DocumentVersion]:
        return self.store.list_versions(doc_id, collection)

    def latest_version(self, doc_id: str, collection: str) -> DocumentVersion | None:
        versions = self.list_versions(doc_id, collection)
        return versions[0] if versions else None

    def diff_summary(self, doc_id: str, collection: str) -> dict[str, Any]:
        versions = self.list_versions(doc_id, collection)
        if not versions:
            return {"count": 0}
        return {
            "count": len(versions),
            "current_hash": versions[0].content_hash,
            "first_hash": versions[-1].content_hash,
            "last_updated_at": versions[0].created_at.isoformat(),
        }
