"""表格内容深度规范化（借鉴 airQA src/utils/content_normalization.py）。

输入：HTML / Markdown / TSV / dict（MinerU table_caption 结构） / 普通文本
输出：TableNormalizationResult（rows / header / data_rows / records / text / caption）

亮点：
- _HTMLTableParser（stdlib HTMLParser）— 不引入 BeautifulSoup 等重依赖
- normalize_table_cell — 数字单元格归一（去千分位 / 去尾零）
- _infer_header — 首行非数字 + 唯一性判定
- records 序列化为 key=value — 让 BM25 能按字段查表
"""

from __future__ import annotations

import csv
import html
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any


@dataclass
class TableNormalizationResult:
    table_html: str = ""
    text: str = ""
    caption: str = ""
    rows: list[list[str]] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    data_rows: list[list[str]] = field(default_factory=list)
    records: list[dict[str, str]] = field(default_factory=list)
    is_table_like: bool = False

    @property
    def table_text(self) -> str:
        return self.text

    @property
    def table_caption(self) -> str:
        return self.caption

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_html": self.table_html,
            "table_text": self.table_text,
            "table_caption": self.table_caption,
            "rows": self.rows,
            "header": self.header,
            "data_rows": self.data_rows,
            "records": self.records,
            "caption": self.caption,
            "text": self.text,
            "is_table_like": self.is_table_like,
        }


def normalize_text_fragment(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()


def normalize_table_cell(value: Any) -> str:
    text = normalize_text_fragment(value)
    text = text.strip("|")
    text = re.sub(r"\s*([,%:/=<>])\s*", r"\1", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    numeric_candidate = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric_candidate):
        try:
            numeric_value = float(numeric_candidate)
            if numeric_value.is_integer():
                return str(int(numeric_value))
            normalized = f"{numeric_value:.12f}".rstrip("0").rstrip(".")
            return normalized or "0"
        except ValueError:
            return text
    return text


def looks_like_table_text(text: Any) -> bool:
    value = normalize_text_fragment(text)
    if not value:
        return False
    lowered = value.lower()
    if "<table" in lowered and "</table>" in lowered:
        return True
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len([line for line in lines if line.count("|") >= 2]) >= 2:
        return True
    if len([line for line in lines if "\t" in line]) >= 2:
        return True
    return False


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.in_cell = False
        self.in_caption = False
        self.caption_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []
        elif tag == "tr":
            self.current_row = []
        elif tag == "caption":
            self.in_caption = True
            self.caption_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self.in_cell = False
            cell = normalize_table_cell(" ".join(self.current_cell))
            self.current_row.append(cell)
            self.current_cell = []
        elif tag == "tr":
            row = [c for c in self.current_row if c]
            if row:
                self.rows.append(row)
            self.current_row = []
        elif tag == "caption":
            self.in_caption = False

    def handle_data(self, data: str) -> None:
        if self.in_caption:
            self.caption_parts.append(data)
        if self.in_cell:
            self.current_cell.append(data)

    @property
    def caption(self) -> str:
        return normalize_text_fragment(" ".join(self.caption_parts))


def _is_markdown_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped))


def _parse_markdown_table(text: str) -> list[list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    rows: list[list[str]] = []
    for line in lines:
        if _is_markdown_separator(line):
            continue
        parts = [normalize_table_cell(p) for p in line.strip("|").split("|")]
        parts = [p for p in parts if p]
        if parts:
            rows.append(parts)
    return rows


def _parse_delimited_table(text: str) -> list[list[str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    scores: Counter[str] = Counter()
    for line in lines[:5]:
        for delim in ("\t", ",", ";"):
            if delim in line:
                scores[delim] += 1
    if not scores:
        return []
    delim, _ = scores.most_common(1)[0]
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)
    rows: list[list[str]] = []
    for row in reader:
        normalized = [normalize_table_cell(c) for c in row]
        normalized = [c for c in normalized if c]
        if normalized:
            rows.append(normalized)
    return rows


def _is_numeric_like(value: str) -> bool:
    candidate = value.replace(",", "").replace("%", "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate))


def _infer_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if len(rows) < 2:
        return [], rows
    first_row = rows[0]
    second_row = rows[1]
    if len(first_row) != len(second_row):
        return [], rows
    non_numeric_headers = sum(0 if _is_numeric_like(c) else 1 for c in first_row)
    unique_headers = len({c.lower() for c in first_row if c}) == len(first_row)
    if non_numeric_headers >= max(1, len(first_row) - 1) and unique_headers:
        return first_row, rows[1:]
    return [], rows


def _rows_to_records(header: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    if not header:
        return []
    records: list[dict[str, str]] = []
    for row in rows:
        if len(row) != len(header):
            continue
        records.append({k: v for k, v in zip(header, row, strict=False)})
    return records


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict, set)) and not value:
            continue
        return value
    return ""


def _normalize_structured_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text_fragment(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                item_content = _normalize_structured_text(item.get("content") or item.get("text") or "")
                if not item_content:
                    continue
                parts.append(f"${item_content}$" if item_type == "equation_inline" else item_content)
            else:
                normalized = _normalize_structured_text(item)
                if normalized:
                    parts.append(normalized)
        return normalize_text_fragment(" ".join(parts))
    if isinstance(value, dict):
        for key in ("content", "text", "caption", "table_caption"):
            nested = value.get(key)
            if nested not in (None, "", [], {}):
                return _normalize_structured_text(nested)
    return normalize_text_fragment(value)


def normalize_table_content_detailed(content: Any, caption: Any = "") -> TableNormalizationResult:
    """HTML / Markdown / TSV / dict → 统一的 TableNormalizationResult。

    输出 text 字段拼接 caption + header + rows + records（key=value），
    让 BM25 / sparse 检索能按字段查表（如"错误码=E1001"直接命中）。
    """
    table_html = ""
    raw_text_source: Any = content
    raw_caption_source: Any = caption

    if isinstance(content, dict):
        table_html = str(content.get("html", "") or "")
        raw_text_source = _first_non_empty(
            table_html,
            content.get("text"),
            content.get("table_text"),
            content,
        )
        raw_caption_source = _first_non_empty(
            caption,
            content.get("caption"),
            content.get("table_caption"),
            "",
        )
    elif isinstance(content, str):
        table_html = content

    raw_text = raw_text_source if isinstance(raw_text_source, str) else _normalize_structured_text(raw_text_source)
    normalized_caption = _normalize_structured_text(raw_caption_source)

    rows: list[list[str]] = []
    if "<table" in raw_text.lower():
        parser = _HTMLTableParser()
        parser.feed(raw_text)
        rows = parser.rows
        if parser.caption and not normalized_caption:
            normalized_caption = parser.caption
    elif looks_like_table_text(raw_text):
        rows = _parse_markdown_table(raw_text)
        # markdown 解析结果若每行只有一个 cell（说明不是 markdown 表格），尝试 delimited
        if not rows or all(len(r) <= 1 for r in rows):
            delim_rows = _parse_delimited_table(raw_text)
            if delim_rows and any(len(r) >= 2 for r in delim_rows):
                rows = delim_rows

    if not rows and raw_text:
        normalized = normalize_text_fragment(raw_text)
        if "|" in normalized and normalized.count("|") >= 2:
            pieces = [normalize_table_cell(p) for p in normalized.split("|")]
            pieces = [p for p in pieces if p]
            if pieces:
                rows = [pieces]

    header, data_rows = _infer_header(rows)
    records = _rows_to_records(header, data_rows)

    text_lines: list[str] = []
    if normalized_caption:
        text_lines.append(f"caption: {normalized_caption}")
    if header:
        text_lines.append(f"header: {' | '.join(header)}")
    for idx, row in enumerate(rows, 1):
        text_lines.append(f"row{idx}: {' | '.join(row)}")
    for idx, record in enumerate(records, 1):
        serialized = "; ".join(f"{k}={v}" for k, v in record.items())
        text_lines.append(f"record{idx}: {serialized}")

    if not text_lines:
        fallback = _normalize_structured_text(raw_text_source)
        if fallback:
            text_lines.append(fallback)

    return TableNormalizationResult(
        table_html=table_html,
        text="\n".join(text_lines).strip(),
        caption=normalized_caption,
        rows=rows,
        header=header,
        data_rows=data_rows,
        records=records,
        is_table_like=bool(rows) or looks_like_table_text(raw_text),
    )


__all__ = [
    "TableNormalizationResult",
    "looks_like_table_text",
    "normalize_table_cell",
    "normalize_table_content_detailed",
    "normalize_text_fragment",
]
