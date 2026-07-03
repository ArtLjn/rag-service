"""表格识别（PDF）。

NSQA 代码不可访问，采用 PyMuPDF 内置的 find_tables()（1.23+ 支持）作为表格识别入口；
对单元格文本做回填，并支持跨页表格的拼接（表头一致即视为延续）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import logger


@dataclass
class TableCell:
    row: int
    col: int
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ExtractedTable:
    page: int
    bbox: tuple[float, float, float, float]
    cells: list[TableCell] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    markdown: str = ""
    html: str = ""


def extract_tables(page_index: int, page: Any) -> list[ExtractedTable]:
    """识别单页表格。"""
    tables: list[ExtractedTable] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        logger.debug(f"page {page_index} find_tables unsupported: {exc!r}")
        return tables

    try:
        extracted = finder.extract()
    except Exception:
        extracted = []

    bboxes = list(getattr(finder, "tables", []) or [])
    for idx, df in enumerate(extracted):
        bbox = tuple(getattr(bboxes[idx], "bbox", (0, 0, 0, 0))) if idx < len(bboxes) else (0, 0, 0, 0)
        table = _dataframe_to_table(page_index, bbox, df)
        tables.append(table)

    for table in tables:
        table.markdown = _to_markdown(table)
        table.html = _to_html(table)
    return tables


def _dataframe_to_table(page_index: int, bbox: tuple, df: Any) -> ExtractedTable:
    rows_raw: list[list[str]] = []
    for _, row in df.iterrows():
        rows_raw.append([_stringify(v) for v in row.tolist()])
    if not rows_raw:
        return ExtractedTable(page=page_index, bbox=bbox)

    header = rows_raw[0]
    body = rows_raw[1:] if len(rows_raw) > 1 else []

    cells: list[TableCell] = []
    for r, row_values in enumerate([header, *body]):
        for c, value in enumerate(row_values):
            cells.append(TableCell(row=r, col=c, text=value))
    return ExtractedTable(
        page=page_index,
        bbox=bbox,
        cells=cells,
        header=header,
        rows=body,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    return text.replace("\n", " ").strip()


def _to_markdown(table: ExtractedTable) -> str:
    if not table.header:
        return ""
    lines: list[str] = []
    lines.append("| " + " | ".join(table.header) + " |")
    lines.append("| " + " | ".join("---" for _ in table.header) + " |")
    for row in table.rows:
        padded = list(row) + [""] * (len(table.header) - len(row))
        lines.append("| " + " | ".join(padded[: len(table.header)]) + " |")
    return "\n".join(lines)


def _to_html(table: ExtractedTable) -> str:
    if not table.header:
        return ""
    lines: list[str] = ["<table>"]
    lines.append("<thead><tr>")
    for cell in table.header:
        lines.append(f"<th>{_escape_html(cell)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in table.rows:
        lines.append("<tr>")
        padded = list(row) + [""] * (len(table.header) - len(row))
        for cell in padded[: len(table.header)]:
            lines.append(f"<td>{_escape_html(cell)}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "".join(lines)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def merge_cross_page_tables(tables_by_page: dict[int, list[ExtractedTable]]) -> list[ExtractedTable]:
    """跨页表格拼接：表头一致即视为延续，行合并到第一个表格。"""
    merged: list[ExtractedTable] = []
    for page in sorted(tables_by_page):
        for table in tables_by_page[page]:
            if not merged or not _header_matches(merged[-1].header, table.header):
                merged.append(table)
                continue
            target = merged[-1]
            target.rows.extend(table.rows)
            for cell in table.cells:
                target.cells.append(
                    TableCell(
                        row=cell.row + len(target.header) + len(target.rows) - len(table.rows),
                        col=cell.col,
                        text=cell.text,
                    )
                )
            target.markdown = _to_markdown(target)
            target.html = _to_html(target)
    return merged


def _header_matches(a: list[str], b: list[str]) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    norm_a = [x.lower().strip() for x in a]
    norm_b = [x.lower().strip() for x in b]
    return norm_a == norm_b


__all__ = [
    "ExtractedTable",
    "TableCell",
    "extract_tables",
    "merge_cross_page_tables",
]
