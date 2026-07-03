"""解析器抽象基类与工厂。

约定：
- 每个解析器接收 bytes/str + metadata，返回 list[Chunk]
- 解析器不写库，只产出结构化 Chunk
- chunk metadata 在解析阶段填 source/category/page/heading_path
- chunk_index 由调用方（pipeline）统一赋值
"""

from __future__ import annotations

import abc
from typing import Any

from app.core.exceptions import UnsupportedFormat
from app.core.logging import logger
from app.models.chunk import Chunk

SUPPORTED_FILE_TYPES = ("pdf", "markdown", "md", "txt", "text")


class BaseParser(abc.ABC):
    """解析器抽象基类。"""

    file_type: str = "unknown"

    @abc.abstractmethod
    async def parse(self, content: bytes | str, metadata: dict[str, Any]) -> list[Chunk]:
        """解析 content，返回 Chunk 列表。"""
        raise NotImplementedError

    def _normalize_content(self, content: bytes | str) -> bytes | str:
        return content


def get_parser(file_type: str) -> BaseParser:
    """解析器工厂：按文件后缀返回对应解析器。"""
    normalized = (file_type or "").lower().lstrip(".")
    if normalized == "pdf":
        from app.parser.pdf_parser import PdfParser

        return PdfParser()
    if normalized in {"md", "markdown"}:
        from app.parser.markdown_parser import MarkdownParser

        return MarkdownParser()
    if normalized in {"txt", "text"}:
        from app.parser.text_parser import TextParser

        return TextParser()
    raise UnsupportedFormat(f"unsupported file type: {file_type}")


def detect_file_type(filename: str | None, content_type: str | None = None) -> str:
    """根据文件名 / content-type 推断文件类型。"""
    if content_type:
        ct = content_type.lower()
        if ct == "application/pdf":
            return "pdf"
        if ct in {"text/markdown", "text/x-markdown"}:
            return "md"
        if ct.startswith("text/"):
            return "txt"
    if filename:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix == "pdf":
            return "pdf"
        if suffix in {"md", "markdown"}:
            return "md"
        if suffix in {"txt", "text"}:
            return "txt"
    raise UnsupportedFormat(f"cannot detect file type from filename={filename} content_type={content_type}")


async def parse_with_degradation(
    file_type: str,
    content: bytes | str,
    metadata: dict[str, Any],
) -> tuple[list[Chunk], str | None]:
    """解析入口，捕获解析异常转为 None + warning。"""
    parser = get_parser(file_type)
    try:
        chunks = await parser.parse(content, metadata)
        logger.debug(f"parsed {file_type}: chunks={len(chunks)}")
        return chunks, None
    except Exception as exc:
        logger.warning(f"parser {file_type} failed: {exc!r}, degrading to text parser")
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8", errors="ignore")
            except Exception:
                content = ""
        from app.parser.text_parser import TextParser

        fallback = TextParser()
        chunks = await fallback.parse(content, metadata)
        return chunks, f"parser_degraded: {type(exc).__name__}"
