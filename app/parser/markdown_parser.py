"""Markdown 解析器：识别 ATX 标题层级、保留代码块、按段落聚合。"""

from __future__ import annotations

import re
from typing import Any

from app.models.chunk import Chunk, ChunkMetadata

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


class MarkdownParser:
    file_type = "md"

    async def parse(self, content: bytes | str, metadata: dict[str, Any]) -> list[Chunk]:
        text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        source = metadata.get("source")
        doc_id = metadata.get("doc_id")
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        buffer: list[str] = []
        current_category = "paragraph"
        in_code_fence = False

        def flush(idx_ref: list[int]) -> None:
            if not buffer:
                return
            content_str = "\n".join(buffer).strip()
            if content_str:
                chunks.append(
                    Chunk(
                        content=content_str,
                        metadata=ChunkMetadata(
                            source=source,
                            page=1,
                            category=current_category,
                            heading_path=list(heading_stack),
                            doc_id=doc_id,
                            chunk_index=idx_ref[0],
                        ),
                    )
                )
                idx_ref[0] += 1
            buffer.clear()

        idx_ref = [0]
        for raw_line in text.split("\n"):
            line = raw_line.rstrip()
            if _FENCE_RE.match(line):
                in_code_fence = not in_code_fence
                buffer.append(line)
                continue
            if in_code_fence:
                buffer.append(line)
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush(idx_ref)
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append("")
                heading_stack.append(title)
                chunks.append(
                    Chunk(
                        content=f"{'#' * level} {title}",
                        metadata=ChunkMetadata(
                            source=source,
                            page=1,
                            category="title",
                            heading_path=list(heading_stack),
                            doc_id=doc_id,
                            chunk_index=idx_ref[0],
                        ),
                    )
                )
                idx_ref[0] += 1
                continue

            if _LIST_RE.match(line):
                buffer.append(line)
                current_category = "list_item"
                continue

            if not line.strip():
                flush(idx_ref)
                current_category = "paragraph"
                continue

            buffer.append(line)
            current_category = "paragraph"

        flush(idx_ref)
        return chunks
