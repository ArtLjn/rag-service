"""layout 与 table 子模块单测：基于 fake page 对象。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.parser.layout import analyzer
from app.parser.table import extractor


def _span(text: str, size: float = 12.0, font: str = "Helvetica") -> dict:
    return {"text": text, "size": size, "font": font, "bbox": (0, 0, 10, 10)}


def _line(spans: list[dict]) -> dict:
    return {"spans": spans, "bbox": (0, 0, 100, 12)}


def _block(lines: list[dict], bbox: tuple = (0, 10, 200, 50), btype: int = 0) -> dict:
    return {"type": btype, "lines": lines, "bbox": bbox}


def _page(blocks: list[dict], width: float = 600, height: float = 800) -> SimpleNamespace:
    return SimpleNamespace(
        rect=SimpleNamespace(width=width, height=height),
        get_text=lambda *_, **__: {"blocks": blocks},
    )


def test_analyze_page_classifies_paragraph_and_header() -> None:
    long_header = "本文件为内部资料，仅供测试使用，未经授权不得传播。" * 2
    page = _page(
        [
            _block([_line([_span(long_header, size=10)])], bbox=(0, 5, 600, 15)),
            _block([_line([_span("正文段落，描述主体内容。", size=12)])], bbox=(0, 100, 600, 130)),
        ]
    )
    layout = analyzer.analyze_page(0, page)
    categories = {el.category for el in layout.elements}
    assert "header" in categories
    assert "paragraph" in categories


def test_analyze_page_detects_numbered_title() -> None:
    page = _page(
        [
            _block([_line([_span("1.2 部署说明", size=18)])], bbox=(0, 60, 600, 80)),
            _block([_line([_span("正文段落内容。", size=12)])], bbox=(0, 100, 600, 120)),
        ]
    )
    layout = analyzer.analyze_page(0, page)
    titles = [el for el in layout.elements if el.category == analyzer.CATEGORY_TITLE]
    assert titles, "expected a title element"
    assert "部署说明" in titles[0].text


def test_analyze_page_assigns_two_columns() -> None:
    page = _page(
        [
            _block([_line([_span("左栏文本内容。")])], bbox=(50, 100, 280, 130)),
            _block([_line([_span("右栏文本内容。")])], bbox=(320, 100, 550, 130)),
            _block([_line([_span("左栏下一段。")])], bbox=(50, 150, 280, 180)),
            _block([_line([_span("右栏下一段。")])], bbox=(320, 150, 550, 180)),
        ],
        width=600,
        height=800,
    )
    layout = analyzer.analyze_page(0, page)
    columns = {el.column for el in layout.elements}
    assert columns == {0, 1}


def test_infer_heading_level_by_dots() -> None:
    assert analyzer.infer_heading_level("2.3.1 子标题", font_size=12, avg_font_size=12) == 3
    assert analyzer.infer_heading_level("1 标题", font_size=20, avg_font_size=12) == 1


def test_build_heading_path_filters_empty() -> None:
    page = _page(
        [
            _block([_line([_span("1 概述", size=18)])], bbox=(0, 100, 600, 120)),
            _block([_line([_span("正文段落", size=12)])], bbox=(0, 130, 600, 150)),
        ]
    )
    layout = analyzer.analyze_page(0, page)
    titles = [el for el in layout.elements if el.category == analyzer.CATEGORY_TITLE]
    path = analyzer.build_heading_path(titles, avg_font_size=12)
    assert path == ["1 概述"]


def test_extract_tables_handles_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    page = SimpleNamespace(find_tables=lambda: SimpleNamespace(tables=[], extract=lambda: []))
    tables = extractor.extract_tables(0, page)
    assert tables == []


def test_extract_tables_converts_dataframe_to_markdown() -> None:
    class FakeRow:
        def __init__(self, values: list) -> None:
            self._values = values

        def tolist(self) -> list:
            return self._values

    class FakeDF:
        def __init__(self) -> None:
            self.rows = [FakeRow(["name", "age"]), FakeRow(["张三", 20]), FakeRow(["李四", 25])]

        def iterrows(self):
            yield from enumerate(self.rows)

    class FakeBBox:
        bbox = (0, 0, 100, 50)

    fake_finder = SimpleNamespace(
        tables=[FakeBBox()],
        extract=lambda: [FakeDF()],
    )
    page = SimpleNamespace(find_tables=lambda: fake_finder)
    tables = extractor.extract_tables(0, page)
    assert len(tables) == 1
    assert "张三" in tables[0].markdown
    assert tables[0].header == ["name", "age"]


def test_merge_cross_page_tables_combines_when_header_matches() -> None:
    from app.parser.table.extractor import ExtractedTable, TableCell, merge_cross_page_tables

    t1 = ExtractedTable(
        page=0,
        bbox=(0, 0, 100, 50),
        header=["a", "b"],
        rows=[["1", "2"]],
        cells=[TableCell(row=0, col=0, text="a"), TableCell(row=0, col=1, text="b"), TableCell(row=1, col=0, text="1")],
    )
    t2 = ExtractedTable(
        page=1,
        bbox=(0, 0, 100, 50),
        header=["a", "b"],
        rows=[["3", "4"]],
        cells=[TableCell(row=1, col=0, text="3")],
    )
    merged = merge_cross_page_tables({0: [t1], 1: [t2]})
    assert len(merged) == 1
    assert len(merged[0].rows) == 2


def test_merge_cross_page_tables_keeps_separate_when_headers_differ() -> None:
    from app.parser.table.extractor import ExtractedTable, merge_cross_page_tables

    t1 = ExtractedTable(page=0, bbox=(0, 0, 100, 50), header=["a"], rows=[["1"]])
    t2 = ExtractedTable(page=1, bbox=(0, 0, 100, 50), header=["x"], rows=[["1"]])
    merged = merge_cross_page_tables({0: [t1], 1: [t2]})
    assert len(merged) == 2
