"""固定窗口分块 + 重叠。

复用主系统 personal_knowledge_base/document_processor.py 的算法思路：
- 按 chunk_size 切分
- 相邻块之间有 overlap 字符重叠，避免句子被切断后丢失上下文
"""

from __future__ import annotations

from app.core.config import settings
from app.models.chunk import Chunk, ChunkMetadata


def chunk(
    chunks: list[Chunk],
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    if not chunks:
        return []
    size = chunk_size or settings.default_chunk_size
    ov = overlap if overlap is not None else settings.default_chunk_overlap
    size = max(50, size)
    ov = max(0, min(ov, size // 2))

    raw_text = "\n\n".join(c.content for c in chunks)
    if not raw_text.strip():
        return chunks

    base = chunks[0].metadata
    pieces: list[str] = []
    cursor = 0
    while cursor < len(raw_text):
        end = min(cursor + size, len(raw_text))
        pieces.append(raw_text[cursor:end])
        if end >= len(raw_text):
            break
        cursor = end - ov

    result: list[Chunk] = []
    for i, text in enumerate(pieces):
        text = text.strip()
        if not text:
            continue
        result.append(
            Chunk(
                content=text,
                metadata=ChunkMetadata(
                    source=base.source,
                    page=base.page,
                    category="paragraph",
                    heading_path=list(base.heading_path),
                    doc_id=base.doc_id,
                    chunk_index=i,
                ),
            )
        )
    return result


__all__ = ["chunk"]
