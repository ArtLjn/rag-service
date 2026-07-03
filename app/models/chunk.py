"""文档分块（Chunk）及其元数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source: str | None = None
    page: int | None = None
    category: str = "paragraph"
    heading_path: list[str] = Field(default_factory=list)
    doc_id: str | None = None
    chunk_index: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "page": self.page,
            "category": self.category,
            "heading_path": self.heading_path,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> ChunkMetadata:
        known = {"source", "page", "category", "heading_path", "doc_id", "chunk_index"}
        extra = {k: v for k, v in payload.items() if k not in known and not k.startswith("_")}
        return cls(
            source=payload.get("source"),
            page=payload.get("page"),
            category=payload.get("category", "paragraph"),
            heading_path=payload.get("heading_path", []) or [],
            doc_id=payload.get("doc_id"),
            chunk_index=payload.get("chunk_index", 0),
            extra=extra,
        )


class Chunk(BaseModel):
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

    def to_payload(self) -> dict[str, Any]:
        payload = self.metadata.to_qdrant_payload()
        payload["content"] = self.content
        return payload

    @classmethod
    def from_payload(cls, content: str, payload: dict[str, Any]) -> Chunk:
        meta = ChunkMetadata.from_qdrant_payload(payload)
        return cls(content=content, metadata=meta)
