"""按标题层级切分（structure_aware）。

规则：
- 同一二级标题下，相邻 paragraph 元素聚为一块
- 单块最大不超过 800 字，超出按句子边界分裂
- 标题独立成块（保留层级语义）
"""

from __future__ import annotations

from app.models.chunk import Chunk, ChunkMetadata

MAX_STRUCTURE_CHUNK_CHARS = 800


def chunk(chunks: list[Chunk], *, max_chars: int = MAX_STRUCTURE_CHUNK_CHARS) -> list[Chunk]:
    """对解析后的 chunks 做 structure_aware 合并。

    输入：parser 产出的"碎片 chunks"（每个 chunk 对应一个 layout element）
    输出：合并后的 chunks（同二级标题下段落聚合）
    """
    if not chunks:
        return []

    result: list[Chunk] = []
    buffer: list[Chunk] = []
    buffer_chars = 0
    current_heading_path: list[str] = []
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer_chars, chunk_index
        if not buffer:
            return
        merged_text = "\n\n".join(c.content for c in buffer).strip()
        if merged_text:
            base = buffer[0].metadata
            result.append(
                Chunk(
                    content=merged_text,
                    metadata=ChunkMetadata(
                        source=base.source,
                        page=base.page,
                        category=base.category,
                        heading_path=list(current_heading_path),
                        doc_id=base.doc_id,
                        chunk_index=chunk_index,
                    ),
                )
            )
            chunk_index += 1
        buffer.clear()
        buffer_chars = 0

    for chunk in chunks:
        category = chunk.metadata.category
        heading_path = list(chunk.metadata.heading_path)

        if heading_path != current_heading_path:
            flush()
            current_heading_path = heading_path

        if category in {"title", "table", "figure", "formula"}:
            flush()
            result.append(
                Chunk(
                    content=chunk.content,
                    metadata=ChunkMetadata(
                        source=chunk.metadata.source,
                        page=chunk.metadata.page,
                        category=category,
                        heading_path=list(current_heading_path),
                        doc_id=chunk.metadata.doc_id,
                        chunk_index=chunk_index,
                    ),
                )
            )
            chunk_index += 1
            continue

        if buffer_chars + len(chunk.content) > max_chars and buffer:
            flush()

        buffer.append(chunk)
        buffer_chars += len(chunk.content) + 2

    flush()
    return result


__all__ = ["MAX_STRUCTURE_CHUNK_CHARS", "chunk"]
