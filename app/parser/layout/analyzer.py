"""版面分析（PDF）。

NSQA 项目代码不可访问，按主流方案实现：
- 基于 PyMuPDF 的 block 结构 + 启发式规则
- 输出元素 bbox 与类别：title/paragraph/table/figure/formula/header/footer/list_item
- 页眉页脚剔除：位置阈值 5% + 文本重复模式
- 多栏布局：检测列分隔，按列还原阅读顺序
- 标题层级：字号、缩进、是否带编号推断

模型方案（如 LayoutLMv3 / PP-Structure）属于扩展点，留口子但不实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import logger

CATEGORY_TITLE = "title"
CATEGORY_PARAGRAPH = "paragraph"
CATEGORY_TABLE = "table"
CATEGORY_FIGURE = "figure"
CATEGORY_FORMULA = "formula"
CATEGORY_HEADER = "header"
CATEGORY_FOOTER = "footer"
CATEGORY_LIST_ITEM = "list_item"

HEADER_FOOTER_RATIO = 0.05
MIN_TITLE_FONT_RATIO = 1.1
LIST_ITEM_REGEX_PREFIXES = ("•", "·", "- ", "* ", "○", "■")


@dataclass
class LayoutElement:
    page: int
    bbox: tuple[float, float, float, float]
    category: str
    text: str = ""
    font_size: float = 0.0
    font_name: str | None = None
    block_no: int = 0
    column: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageLayout:
    page: int
    width: float
    height: float
    elements: list[LayoutElement] = field(default_factory=list)


def analyze_page(page_index: int, page: Any) -> PageLayout:
    """分析一页，输出 PageLayout。page 是 fitz.Page。"""
    page_width = page.rect.width
    page_height = page.rect.height
    layout = PageLayout(page=page_index, width=page_width, height=page_height)

    header_y_max = page_height * HEADER_FOOTER_RATIO
    footer_y_min = page_height * (1 - HEADER_FOOTER_RATIO)

    text_dict = page.get_text("dict", flags=0)
    blocks = text_dict.get("blocks", [])

    avg_font_size = _compute_average_font_size(blocks)
    page_text_lines: list[str] = []

    for block_no, block in enumerate(blocks):
        if block.get("type", 0) != 0:
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            category = CATEGORY_FIGURE if block.get("type", 0) == 1 else CATEGORY_TABLE
            layout.elements.append(
                LayoutElement(
                    page=page_index,
                    bbox=bbox,
                    category=category,
                    block_no=block_no,
                )
            )
            continue

        lines = block.get("lines", [])
        if not lines:
            continue

        block_text_parts: list[str] = []
        block_max_size = 0.0
        block_font_name: str | None = None
        block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        for line in lines:
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                size = float(span.get("size", 0.0))
                font = span.get("font", "")
                if text:
                    block_text_parts.append(text)
                    block_max_size = max(block_max_size, size)
                    if not block_font_name:
                        block_font_name = font
        block_text = " ".join(block_text_parts).strip()
        if not block_text:
            continue

        page_text_lines.append(block_text)

        category = _classify_block(
            block_text=block_text,
            font_size=block_max_size,
            avg_font_size=avg_font_size,
            bbox=block_bbox,
            header_y_max=header_y_max,
            footer_y_min=footer_y_min,
        )

        layout.elements.append(
            LayoutElement(
                page=page_index,
                bbox=block_bbox,
                category=category,
                text=block_text,
                font_size=block_max_size,
                font_name=block_font_name,
                block_no=block_no,
            )
        )

    if page_text_lines:
        _strip_repeated_headers_footers(layout, page_text_lines)

    _assign_columns(layout)
    _restore_reading_order(layout)
    return layout


def _compute_average_font_size(blocks: list[dict[str, Any]]) -> float:
    sizes: list[float] = []
    for block in blocks:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if text:
                    sizes.append(float(span.get("size", 0.0)))
    if not sizes:
        return 0.0
    return sum(sizes) / len(sizes)


def _classify_block(
    block_text: str,
    font_size: float,
    avg_font_size: float,
    bbox: tuple[float, float, float, float],
    header_y_max: float,
    footer_y_min: float,
) -> str:
    _, y0, _, y1 = bbox
    if y1 <= header_y_max:
        return CATEGORY_HEADER
    if y0 >= footer_y_min:
        return CATEGORY_FOOTER

    if any(block_text.startswith(p) for p in LIST_ITEM_REGEX_PREFIXES):
        return CATEGORY_LIST_ITEM

    looks_like_heading = bool(block_text) and (
        _looks_like_numbered_heading(block_text)
        or (avg_font_size > 0 and font_size >= avg_font_size * MIN_TITLE_FONT_RATIO and len(block_text) < 60)
    )
    if looks_like_heading:
        return CATEGORY_TITLE
    return CATEGORY_PARAGRAPH


def _looks_like_numbered_heading(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t[0].isdigit():
        head = t.split(" ", 1)[0] if " " in t else t[:8]
        return all(ch.isdigit() or ch == "." for ch in head) and any(ch.isdigit() for ch in head)
    return False


def _strip_repeated_headers_footers(layout: PageLayout, page_lines: list[str]) -> None:
    """简单启发式：若某行文本与首行完全相同且短，且处于 header/footer 区，则剔除。"""
    if not page_lines:
        return
    first = page_lines[0]
    last = page_lines[-1]
    if len(first) < 30 and page_lines.count(first) >= 1:
        for el in layout.elements:
            if el.text == first and el.category == CATEGORY_HEADER:
                layout.elements.remove(el)
                break
    if len(last) < 30 and page_lines.count(last) >= 1 and last != first:
        for el in layout.elements:
            if el.text == last and el.category == CATEGORY_FOOTER:
                layout.elements.remove(el)
                break


def _assign_columns(layout: PageLayout) -> None:
    """双栏检测：x0 坐标聚类找最大 gap（借鉴 airQA _detect_page_layouts）。

    若最大 gap 显著大于平均 gap（> 2x），判定为双栏，gap 中点为分界线；
    否则所有元素归为单栏。
    """
    if not layout.elements:
        return
    bboxes = [el.bbox for el in layout.elements]
    x0_sorted = sorted(b[0] for b in bboxes)
    if len(x0_sorted) < 4:
        for el in layout.elements:
            el.column = 0
        return

    gaps = [x0_sorted[i] - x0_sorted[i - 1] for i in range(1, len(x0_sorted))]
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)

    if max_gap <= avg_gap * 2:
        for el in layout.elements:
            el.column = 0
        return

    gap_idx = gaps.index(max_gap)
    threshold = (x0_sorted[gap_idx] + x0_sorted[gap_idx + 1]) / 2
    for el in layout.elements:
        el.column = 0 if (el.bbox[0] + el.bbox[2]) / 2 < threshold else 1


def _restore_reading_order(layout: PageLayout) -> None:
    """按 column → y 坐标重新排序。"""
    layout.elements.sort(key=lambda el: (el.column, el.bbox[1], el.bbox[0]))


def infer_heading_level(text: str, font_size: float, avg_font_size: float) -> int:
    """根据字号与编号推断标题层级。返回 1-6。"""
    if _looks_like_numbered_heading(text):
        dot_count = text.split(" ", 1)[0].count(".")
        return min(6, max(1, dot_count + 1))
    if avg_font_size <= 0:
        return 2
    ratio = font_size / avg_font_size if avg_font_size else 1.0
    if ratio >= 1.8:
        return 1
    if ratio >= 1.5:
        return 2
    if ratio >= 1.3:
        return 3
    return 4


def build_heading_path(elements: list[LayoutElement], avg_font_size: float) -> list[str]:
    """对当前段落元素，从前往后构建标题路径。"""
    path: list[str] = []
    for el in elements:
        if el.category == CATEGORY_TITLE:
            level = infer_heading_level(el.text, el.font_size, avg_font_size)
            path = path[: level - 1]
            while len(path) < level - 1:
                path.append("")
            path.append(el.text)
    return [p for p in path if p]


__all__ = [
    "CATEGORY_FIGURE",
    "CATEGORY_FOOTER",
    "CATEGORY_FORMULA",
    "CATEGORY_HEADER",
    "CATEGORY_LIST_ITEM",
    "CATEGORY_PARAGRAPH",
    "CATEGORY_TABLE",
    "CATEGORY_TITLE",
    "LayoutElement",
    "PageLayout",
    "analyze_page",
    "build_heading_path",
    "infer_heading_level",
]


def debug_layout(layout: PageLayout) -> dict[str, Any]:
    return {
        "page": layout.page,
        "width": layout.width,
        "height": layout.height,
        "element_count": len(layout.elements),
        "categories": {cat: sum(1 for e in layout.elements if e.category == cat) for cat in {e.category for e in layout.elements}},
    }


def log_layout_summary(layout: PageLayout) -> None:
    summary = debug_layout(layout)
    logger.debug(f"page {summary['page']} layout: {summary['categories']}")
