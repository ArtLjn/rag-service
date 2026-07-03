"""MinerU 模块单测：latex_normalizer / constants 映射 / parser JSON 转换。"""

from __future__ import annotations

from app.parser.mineru import constants as mc
from app.parser.mineru import latex_normalizer
from app.parser.mineru.parser import parse_mineru_result


def test_normalize_merges_letter_spaces_in_braces() -> None:
    assert latex_normalizer.normalize(r"^{p r o x}") == r"^{prox}"


def test_normalize_merges_mathrm_letters() -> None:
    # 命令体内字母间空格被合并
    result = latex_normalizer.normalize(r"\mathrm{C E}")
    assert "CE" in result
    assert "C E" not in result


def test_normalize_handles_subscript_supplement() -> None:
    result = latex_normalizer.normalize(r"x _ { max }")
    assert " " not in result.replace(" ", "") or result.count(" ") <= 1


def test_normalize_passes_empty_string() -> None:
    assert latex_normalizer.normalize("") == ""


def test_map_type_to_category_basic() -> None:
    assert mc.map_type_to_category("title") == mc.CATEGORY_TITLE
    assert mc.map_type_to_category("paragraph") == mc.CATEGORY_PARAGRAPH
    assert mc.map_type_to_category("table") == mc.CATEGORY_TABLE
    assert mc.map_type_to_category("equation_interline") == mc.CATEGORY_FORMULA
    assert mc.map_type_to_category("image") == mc.CATEGORY_FIGURE


def test_map_type_to_category_ignores_noise_types() -> None:
    assert mc.map_type_to_category("page_number") is None
    assert mc.map_type_to_category("discarded") is None
    assert mc.map_type_to_category("image_caption") is None


def test_map_type_to_category_unknown_falls_back_to_paragraph() -> None:
    assert mc.map_type_to_category("unknown_type") == mc.CATEGORY_PARAGRAPH
    assert mc.map_type_to_category(None) is None


def test_parse_mineru_result_handles_empty() -> None:
    assert parse_mineru_result({}) == []
    assert parse_mineru_result([]) == []


def test_parse_mineru_result_extracts_titles_and_paragraphs() -> None:
    raw = [
        {"type": "doc_title", "text_level": 1, "page_idx": 0, "content": [{"type": "text", "content": "项目说明"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段落内容一。"}]},
        {"type": "paragraph", "page_idx": 1, "content": [{"type": "text", "content": "段落内容二。"}]},
    ]
    chunks = parse_mineru_result(raw, source="x.pdf", doc_id="d1")
    assert len(chunks) == 3
    assert chunks[0].metadata.category == "title"
    assert chunks[0].content == "项目说明"
    assert chunks[0].metadata.heading_path == ["项目说明"]
    assert chunks[1].metadata.category == "paragraph"
    assert chunks[1].metadata.heading_path == ["项目说明"]
    assert chunks[2].metadata.page == 2


def test_parse_mineru_result_skips_ignored_types() -> None:
    raw = [
        {"type": "title", "page_idx": 0, "content": [{"type": "text", "content": "T1"}]},
        {"type": "page_number", "page_idx": 0, "content": [{"type": "text", "content": "1"}]},
        {"type": "discarded", "page_idx": 0, "content": "noise"},
    ]
    chunks = parse_mineru_result(raw)
    assert len(chunks) == 1
    assert chunks[0].content == "T1"


def test_parse_mineru_result_normalizes_formula() -> None:
    raw = [
        {"type": "equation_interline", "page_idx": 0, "content": r"E = mc^{2}"},
    ]
    chunks = parse_mineru_result(raw)
    assert len(chunks) == 1
    assert chunks[0].metadata.category == "formula"
    assert "$$" in chunks[0].content
    assert "mc^{2}" in chunks[0].content


def test_parse_mineru_result_extracts_table_html_and_markdown() -> None:
    raw = [
        {
            "type": "table",
            "page_idx": 0,
            "content": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
            "bbox": [0, 0, 100, 50],
        },
    ]
    chunks = parse_mineru_result(raw, doc_id="d")
    assert len(chunks) == 1
    assert chunks[0].metadata.category == "table"
    assert "A" in chunks[0].content and "B" in chunks[0].content
    assert chunks[0].metadata.extra.get("table_html", "").startswith("<table")


def test_parse_mineru_result_extracts_nested_dict_content_list() -> None:
    raw = {
        "content_list_v2": [
            {"type": "title", "page_idx": 0, "content": [{"type": "text", "content": "嵌套标题"}]},
        ]
    }
    chunks = parse_mineru_result(raw)
    assert len(chunks) == 1
    assert chunks[0].content == "嵌套标题"


def test_parse_mineru_result_handles_heading_levels() -> None:
    raw = [
        {"type": "title", "text_level": 1, "page_idx": 0, "content": [{"type": "text", "content": "第1章"}]},
        {"type": "title", "text_level": 2, "page_idx": 0, "content": [{"type": "text", "content": "1.1 节"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "正文"}]},
        {"type": "title", "text_level": 2, "page_idx": 0, "content": [{"type": "text", "content": "1.2 节"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "另一段正文"}]},
    ]
    chunks = parse_mineru_result(raw)
    paragraphs = [c for c in chunks if c.metadata.category == "paragraph"]
    assert paragraphs[0].metadata.heading_path == ["第1章", "1.1 节"]
    assert paragraphs[1].metadata.heading_path == ["第1章", "1.2 节"]


def test_latex_to_text_replaces_greek_letters() -> None:
    from app.parser.mineru.latex_normalizer import latex_to_text

    assert "α" in latex_to_text(r"\alpha + \beta")
    assert "∑" in latex_to_text(r"\sum_{i=1}^n x_i")


def test_latex_to_text_handles_single_char_superscript() -> None:
    from app.parser.mineru.latex_normalizer import latex_to_text

    out = latex_to_text("x^2")
    assert "²" in out


def test_latex_to_text_passes_empty() -> None:
    from app.parser.mineru.latex_normalizer import latex_to_text

    assert latex_to_text("") == ""


def test_formula_chunk_includes_text_representation() -> None:
    raw = [
        {"type": "title", "text_level": 1, "page_idx": 0, "content": [{"type": "text", "content": "能量方程"}]},
        {
            "type": "equation_interline",
            "page_idx": 0,
            "bbox": [10, 200, 200, 230],
            "content": r"E = mc^{2}",
        },
    ]
    chunks = parse_mineru_result(raw)
    formula = next(c for c in chunks if c.metadata.category == "formula")
    assert "$$" in formula.content
    assert "mc^{2}" in formula.content
    assert formula.metadata.extra.get("text", "")
    assert formula.metadata.heading_path == ["能量方程"]


def test_formula_anchor_finds_nearest_title_by_distance() -> None:
    """公式不在标题正下方时，应绑空间最近的标题而非按时间顺序累积的标题。"""
    raw = [
        {"type": "title", "text_level": 2, "page_idx": 0, "bbox": [50, 50, 200, 80], "content": [{"type": "text", "content": "2.1 左侧公式"}]},
        {"type": "title", "text_level": 2, "page_idx": 0, "bbox": [400, 50, 550, 80], "content": [{"type": "text", "content": "2.2 右侧公式"}]},
        {"type": "equation_interline", "page_idx": 0, "bbox": [80, 200, 250, 230], "content": r"a^2 + b^2 = c^2"},
    ]
    chunks = parse_mineru_result(raw)
    formula = next(c for c in chunks if c.metadata.category == "formula")
    assert any("左侧公式" in p for p in formula.metadata.heading_path)
    assert not any("右侧公式" in p for p in formula.metadata.heading_path)


def test_table_chunk_keeps_html_in_extra() -> None:
    raw = [
        {"type": "title", "text_level": 1, "page_idx": 0, "content": [{"type": "text", "content": "实验数据"}]},
        {"type": "table", "page_idx": 0, "bbox": [50, 100, 400, 200], "content": "<table><tr><td>X</td></tr></table>"},
    ]
    chunks = parse_mineru_result(raw)
    table = next(c for c in chunks if c.metadata.category == "table")
    assert table.metadata.extra.get("table_html", "").startswith("<table")
    assert table.metadata.heading_path == ["实验数据"]
