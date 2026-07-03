"""解析服务编排：parse → chunk → clean → 生成 layout_summary。"""

from __future__ import annotations

import hashlib
from typing import Any

from app.core.exceptions import ParseFailed
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.query import ChunkingStrategy
from app.parser.base import detect_file_type, get_parser
from app.parser.chunker import chunk_with_strategy, select_default_strategy
from app.parser.cleaner import clean as clean_chunks


def compute_doc_id(content: bytes | str) -> str:
    raw = content.encode("utf-8", errors="ignore") if isinstance(content, str) else content
    return hashlib.md5(raw).hexdigest()[:12]


def build_layout_summary(chunks: list[Chunk]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for chunk in chunks:
        category = chunk.metadata.category or "paragraph"
        summary[category] = summary.get(category, 0) + 1
    summary["total"] = len(chunks)
    return summary


async def parse_content(
    *,
    content: bytes | str,
    file_type: str | None = None,
    strategy: ChunkingStrategy | str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[Chunk], str, dict[str, int], str | None]:
    metadata = metadata or {}
    if not file_type:
        file_type = detect_file_type(metadata.get("source"), metadata.get("content_type"))
    metadata.setdefault("doc_id", compute_doc_id(content))
    metadata.setdefault("source", "upload")

    parser = get_parser(file_type)
    try:
        raw_chunks = await parser.parse(content, metadata)
    except Exception as exc:
        logger.warning(f"parser {file_type} failed, degrading to text: {exc!r}")
        from app.parser.text_parser import TextParser

        raw_chunks = await TextParser().parse(
            content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else content,
            metadata,
        )

    if strategy is None:
        strategy = select_default_strategy(file_type)

    options: dict[str, Any] = {}
    if chunk_size is not None:
        options["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        options["chunk_overlap"] = chunk_overlap

    try:
        chunked = chunk_with_strategy(raw_chunks, strategy, file_type=file_type, options=options)
    except Exception as exc:
        logger.warning(f"chunker {strategy} failed, degrading to fixed: {exc!r}")
        chunked = chunk_with_strategy(raw_chunks, ChunkingStrategy.FIXED, file_type=file_type, options=options)

    cleaned = clean_chunks(chunked)
    summary = build_layout_summary(cleaned)
    logger.info(
        f"parsed file_type={file_type} strategy={strategy} raw={len(raw_chunks)} cleaned={len(cleaned)} summary={summary}"
    )
    return cleaned, metadata["doc_id"], summary, None


__all__ = [
    "build_layout_summary",
    "compute_doc_id",
    "parse_content",
]


def raise_parse_failed(message: str) -> None:
    raise ParseFailed(message)
