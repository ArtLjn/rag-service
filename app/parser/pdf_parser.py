"""PDF 复杂解析器：双轨架构。

策略：
1. 若配置了 MINERU_API_TOKEN，优先调用 MinerU 云端 API（论文核心创新点，借鉴 airQA）
2. 否则降级为 PyMuPDF + 启发式版面分析（离线 fallback）
"""

from __future__ import annotations

import io
from typing import Any

from app.core.config import settings
from app.core.exceptions import ParseFailed
from app.core.logging import logger
from app.models.chunk import Chunk, ChunkMetadata
from app.parser.layout.analyzer import (
    CATEGORY_FIGURE,
    CATEGORY_FORMULA,
    CATEGORY_PARAGRAPH,
    CATEGORY_TABLE,
    CATEGORY_TITLE,
    LayoutElement,
    PageLayout,
    analyze_page,
)
from app.parser.table.extractor import ExtractedTable, extract_tables, merge_cross_page_tables

try:
    import fitz  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - PyMuPDF should be installed
    fitz = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class PdfParser:
    file_type = "pdf"

    async def parse(self, content: bytes | str, metadata: dict[str, Any]) -> list[Chunk]:
        if isinstance(content, str):
            try:
                content_bytes = _decode_to_bytes(content)
            except Exception as exc:
                raise ParseFailed(f"cannot decode pdf content: {exc}") from exc
        else:
            content_bytes = content

        if settings.mineru_api_token:
            try:
                return await self._parse_with_mineru(content_bytes, metadata)
            except ParseFailed as exc:
                logger.warning(f"mineru parse failed, fallback to pymupdf: {exc.message}")
        return await self._parse_with_pymupdf(content_bytes, metadata)

    async def _parse_with_mineru(self, content: bytes, metadata: dict[str, Any]) -> list[Chunk]:
        from app.parser.mineru.client import MinerUClient
        from app.parser.mineru.parser import parse_mineru_result

        client = MinerUClient(
            api_token=settings.mineru_api_token or "",
            base_url=settings.mineru_base_url,
            model_version=settings.mineru_model_version,
            timeout=settings.mineru_timeout,
            poll_interval=settings.mineru_poll_interval,
        )
        filename = (metadata.get("source") or "upload.pdf").rsplit("/", 1)[-1]
        raw = await client.parse_pdf_bytes(content, filename=filename)
        return parse_mineru_result(
            raw,
            source=metadata.get("source"),
            doc_id=metadata.get("doc_id"),
        )

    async def _parse_with_pymupdf(self, content: bytes, metadata: dict[str, Any]) -> list[Chunk]:
        if fitz is None:
            raise ParseFailed(f"PyMuPDF not available: {_IMPORT_ERROR}")

        source = metadata.get("source")
        doc_id = metadata.get("doc_id")

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ParseFailed(f"open pdf failed: {exc}") from exc

        layouts: list[PageLayout] = []
        tables_by_page: dict[int, list[ExtractedTable]] = {}
        avg_font_sizes: list[float] = []

        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                layout = analyze_page(page_index, page)
                layouts.append(layout)
                avg_size = _page_avg_font_size(layout)
                if avg_size > 0:
                    avg_font_sizes.append(avg_size)
                tables = extract_tables(page_index, page)
                if tables:
                    tables_by_page[page_index] = tables
        finally:
            doc.close()

        merged_tables = merge_cross_page_tables(tables_by_page)
        global_avg_size = sum(avg_font_sizes) / len(avg_font_sizes) if avg_font_sizes else 0.0

        chunks: list[Chunk] = []
        chunk_index = 0
        current_heading_path: list[str] = []

        for layout in layouts:
            page_table_bboxes = {tuple(t.bbox): t for t in tables_by_page.get(layout.page, [])}

            for el in layout.elements:
                if el.category == CATEGORY_TABLE and tuple(el.bbox) in page_table_bboxes:
                    table = page_table_bboxes[tuple(el.bbox)]
                    content_str = table.markdown or _bbox_caption(el)
                    chunks.append(_mk(content_str, source, layout.page + 1, CATEGORY_TABLE, current_heading_path, doc_id, chunk_index))
                    chunk_index += 1
                    continue

                if el.category == CATEGORY_FIGURE:
                    caption = _find_figure_caption(layout, el)
                    if not caption:
                        continue
                    chunks.append(_mk(caption, source, layout.page + 1, CATEGORY_FIGURE, current_heading_path, doc_id, chunk_index))
                    chunk_index += 1
                    continue

                if el.category == CATEGORY_FORMULA:
                    chunks.append(_mk(f"[公式: page={layout.page + 1}, bbox={el.bbox}]", source, layout.page + 1, CATEGORY_FORMULA, current_heading_path, doc_id, chunk_index))
                    chunk_index += 1
                    continue

                if el.category == CATEGORY_TITLE:
                    current_heading_path = _push_heading(current_heading_path, el, global_avg_size)
                    chunks.append(_mk(el.text, source, layout.page + 1, CATEGORY_TITLE, current_heading_path, doc_id, chunk_index))
                    chunk_index += 1
                    continue

                if el.category == CATEGORY_PARAGRAPH and el.text:
                    chunks.append(_mk(el.text, source, layout.page + 1, CATEGORY_PARAGRAPH, current_heading_path, doc_id, chunk_index))
                    chunk_index += 1

        logger.info(f"pdf parsed (pymupdf): pages={len(layouts)}, chunks={len(chunks)}, tables={len(merged_tables)}")
        return chunks


def _mk(content: str, source: str | None, page: int, category: str, heading_path: list[str], doc_id: str | None, chunk_index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source=source,
            page=page,
            category=category,
            heading_path=list(heading_path),
            doc_id=doc_id,
            chunk_index=chunk_index,
        ),
    )


def _decode_to_bytes(content: str) -> bytes:
    """接受 base64 或 utf-8（PDF 应当 base64 传输，但保留兜底）。"""
    import base64

    try:
        return base64.b64decode(content, validate=True)
    except Exception:
        pass
    return content.encode("utf-8", errors="ignore")


def _page_avg_font_size(layout: PageLayout) -> float:
    sizes = [el.font_size for el in layout.elements if el.font_size > 0]
    if not sizes:
        return 0.0
    return sum(sizes) / len(sizes)


def _push_heading(current: list[str], el: LayoutElement, avg_size: float) -> list[str]:
    from app.parser.layout.analyzer import infer_heading_level

    level = infer_heading_level(el.text, el.font_size, avg_size)
    new_path = current[: level - 1]
    while len(new_path) < level - 1:
        new_path.append("")
    new_path.append(el.text)
    return [p for p in new_path if p]


def _bbox_caption(el: LayoutElement) -> str:
    return f"[table: page={el.page + 1}, bbox={el.bbox}]"


def _find_figure_caption(layout: PageLayout, figure: LayoutElement) -> str | None:
    """找图周围距离最近的 paragraph，作为 caption。"""
    fy = (figure.bbox[1] + figure.bbox[3]) / 2
    fx = (figure.bbox[0] + figure.bbox[2]) / 2
    candidates: list[tuple[float, LayoutElement]] = []
    for el in layout.elements:
        if el.category != CATEGORY_PARAGRAPH or not el.text:
            continue
        if _looks_like_caption(el.text):
            ey = (el.bbox[1] + el.bbox[3]) / 2
            ex = (el.bbox[0] + el.bbox[2]) / 2
            distance = ((ey - fy) ** 2 + (ex - fx) ** 2) ** 0.5
            if distance < max(layout.width, layout.height) * 0.3:
                candidates.append((distance, el))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1].text


def _looks_like_caption(text: str) -> bool:
    head = text.lstrip()[:8].lower()
    return head.startswith("图") or head.startswith("fig")


def layout_summary(chunks: list[Chunk]) -> dict[str, Any]:
    summary: dict[str, int] = {}
    for chunk in chunks:
        category = chunk.metadata.category
        summary[category] = summary.get(category, 0) + 1
    summary["total"] = len(chunks)
    return summary


__all__ = [
    "PdfParser",
    "layout_summary",
]


def open_pdf(content: bytes | str) -> Any:
    if fitz is None:
        raise ParseFailed(f"PyMuPDF not available: {_IMPORT_ERROR}")
    if isinstance(content, str):
        content = _decode_to_bytes(content)
    return fitz.open(stream=content, filetype="pdf")


def normalize_stream(content: bytes | str) -> bytes:
    if isinstance(content, str):
        return _decode_to_bytes(content)
    return content


def to_bytes_stream(content: bytes | str) -> io.BytesIO:
    return io.BytesIO(normalize_stream(content))
