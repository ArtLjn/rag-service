"""table_normalizer 单测。"""

from __future__ import annotations

from app.parser.mineru.table_normalizer import (
    looks_like_table_text,
    normalize_table_cell,
    normalize_table_content_detailed,
)


def test_normalize_table_cell_strips_thousands_separator_and_trailing_zeros() -> None:
    assert normalize_table_cell("1,000.50") == "1000.5"
    assert normalize_table_cell("3.14000") == "3.14"
    assert normalize_table_cell("100") == "100"


def test_normalize_table_cell_normalizes_punctuation_spacing() -> None:
    assert normalize_table_cell("a , b") == "a,b"
    assert normalize_table_cell("100 %") == "100%"


def test_looks_like_table_text_detects_pipe_and_html() -> None:
    assert looks_like_table_text("| A | B |\n|---|---|\n| 1 | 2 |") is True
    assert looks_like_table_text("<table><tr><td>x</td></tr></table>") is True
    assert looks_like_table_text("纯文本一行") is False


def test_normalize_html_extracts_header_and_records() -> None:
    html = "<table><tr><td>错误码</td><td>说明</td></tr><tr><td>E1001</td><td>交换机宕机</td></tr></table>"
    result = normalize_table_content_detailed(html)
    assert result.header == ["错误码", "说明"]
    assert result.records == [{"错误码": "E1001", "说明": "交换机宕机"}]
    assert "错误码=E1001" in result.text


def test_normalize_markdown_table_records() -> None:
    md = "| 错误码 | 说明 |\n|---|---|\n| E1001 | 宕机 |"
    result = normalize_table_content_detailed(md)
    assert result.header == ["错误码", "说明"]
    assert result.records[0]["错误码"] == "E1001"


def test_normalize_tsv_records() -> None:
    tsv = "name\tage\nAlice\t30\nBob\t25"
    result = normalize_table_content_detailed(tsv)
    assert result.header == ["name", "age"]
    assert len(result.records) == 2


def test_normalize_dict_input_with_caption() -> None:
    payload = {"html": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>", "caption": "示例"}
    result = normalize_table_content_detailed(payload)
    assert result.caption == "示例"
    assert "caption: 示例" in result.text
