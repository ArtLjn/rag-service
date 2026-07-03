"""纯文本解析器：按段落（双换行）分割。"""

from __future__ import annotations

from typing import Any

from app.models.chunk import Chunk, ChunkMetadata


class TextParser:
    file_type = "txt"

    async def parse(self, content: bytes | str, metadata: dict[str, Any]) -> list[Chunk]:
        text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]

        source = metadata.get("source")
        doc_id = metadata.get("doc_id")
        return [
            Chunk(
                content=p,
                metadata=ChunkMetadata(
                    source=source,
                    page=1,
                    category="paragraph",
                    heading_path=[],
                    doc_id=doc_id,
                    chunk_index=i,
                ),
            )
            for i, p in enumerate(paragraphs)
        ]
