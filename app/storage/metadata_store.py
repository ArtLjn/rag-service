"""SQLite 元数据存储。

表：
- documents: doc_id, collection, source, category, chunk_count, content_hash, extra(JSON), ingested_at
- document_versions: version_id, doc_id, collection, content_hash, chunk_count, created_at, note
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.models.document import DocumentRecord, DocumentVersion

_LOCK = threading.Lock()


class MetadataStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.metadata_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_schema(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        source TEXT,
                        category TEXT,
                        chunk_count INTEGER DEFAULT 0,
                        content_hash TEXT NOT NULL,
                        extra TEXT,
                        ingested_at TEXT NOT NULL,
                        PRIMARY KEY (doc_id, collection)
                    );
                    CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);

                    CREATE TABLE IF NOT EXISTS document_versions (
                        version_id TEXT NOT NULL PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        chunk_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        note TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_versions_doc ON document_versions(doc_id, collection);
                    """
                )
        logger.debug(f"metadata store ready at {self.db_path}")

    def upsert_document(self, record: DocumentRecord) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO documents(doc_id, collection, source, category, chunk_count, content_hash, extra, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id, collection) DO UPDATE SET
                        source=excluded.source,
                        category=excluded.category,
                        chunk_count=excluded.chunk_count,
                        content_hash=excluded.content_hash,
                        extra=excluded.extra,
                        ingested_at=excluded.ingested_at
                    """,
                    (
                        record.doc_id,
                        record.collection,
                        record.source,
                        record.category,
                        record.chunk_count,
                        record.content_hash,
                        json.dumps(record.extra, ensure_ascii=False),
                        record.ingested_at.isoformat(),
                    ),
                )

    def add_version(self, version: DocumentVersion) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO document_versions(version_id, doc_id, collection, content_hash, chunk_count, created_at, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version.version_id,
                        version.doc_id,
                        version.collection,
                        version.content_hash,
                        version.chunk_count,
                        version.created_at.isoformat(),
                        version.note,
                    ),
                )

    def get_document(self, doc_id: str, collection: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id=? AND collection=?",
                (doc_id, collection),
            ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def list_documents(
        self, collection: str, page: int = 1, page_size: int = 20
    ) -> tuple[int, list[DocumentRecord]]:
        offset = max(0, (page - 1) * page_size)
        with self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS c FROM documents WHERE collection=?",
                (collection,),
            ).fetchone()
            total = int(total_row["c"]) if total_row else 0
            rows = conn.execute(
                "SELECT * FROM documents WHERE collection=? ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (collection, page_size, offset),
            ).fetchall()
        return total, [self._row_to_record(r) for r in rows]

    def delete_document(self, doc_id: str, collection: str) -> bool:
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM documents WHERE doc_id=? AND collection=?",
                    (doc_id, collection),
                )
                return cur.rowcount > 0

    def delete_collection_documents(self, collection: str) -> int:
        with _LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM documents WHERE collection=?",
                    (collection,),
                )
                return cur.rowcount

    def list_versions(self, doc_id: str, collection: str) -> list[DocumentVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM document_versions WHERE doc_id=? AND collection=? ORDER BY created_at DESC",
                (doc_id, collection),
            ).fetchall()
        return [
            DocumentVersion(
                version_id=r["version_id"],
                doc_id=r["doc_id"],
                collection=r["collection"],
                content_hash=r["content_hash"],
                chunk_count=r["chunk_count"],
                created_at=datetime.fromisoformat(r["created_at"]),
                note=r["note"],
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        extra_raw = row["extra"]
        try:
            extra = json.loads(extra_raw) if extra_raw else {}
        except Exception:
            extra = {}
        return DocumentRecord(
            doc_id=row["doc_id"],
            collection=row["collection"],
            source=row["source"],
            category=row["category"],
            chunk_count=row["chunk_count"],
            content_hash=row["content_hash"],
            extra=extra,
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )


def file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
