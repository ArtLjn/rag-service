"""文档元数据与版本记录模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: str
    collection: str
    source: str | None = None
    category: str | None = None
    chunk_count: int = 0
    content_hash: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentVersion(BaseModel):
    version_id: str
    doc_id: str
    collection: str
    content_hash: str
    chunk_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    note: str | None = None
