"""logic_idx + neighbor_ids 单测（在 test_mineru 基础上更聚焦）。"""

from __future__ import annotations

from app.parser.mineru.parser import parse_mineru_result


def test_logic_idx_assigned_globally_continuous_across_pages() -> None:
    raw = [
        {"type": "title", "page_idx": 0, "text_level": 1, "content": [{"type": "text", "content": "P1 标题"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "P1 段落"}]},
        {"type": "title", "page_idx": 1, "text_level": 1, "content": [{"type": "text", "content": "P2 标题"}]},
        {"type": "paragraph", "page_idx": 1, "content": [{"type": "text", "content": "P2 段落"}]},
    ]
    chunks = parse_mineru_result(raw)
    indices = sorted(c.metadata.extra.get("logic_idx", -1) for c in chunks)
    assert indices == [0, 1, 2, 3]


def test_neighbor_ids_link_same_category_consecutively() -> None:
    raw = [
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段一"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段二"}]},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段三"}]},
    ]
    chunks = parse_mineru_result(raw, doc_id="d1")
    by_content = {c.content: c for c in chunks}
    assert by_content["段一"].metadata.extra.get("next_view_id") == by_content["段二"].metadata.extra.get("chunk_id")
    assert by_content["段二"].metadata.extra.get("prev_view_id") == by_content["段一"].metadata.extra.get("chunk_id")
    assert "prev_view_id" not in by_content["段一"].metadata.extra
    assert "next_view_id" not in by_content["段三"].metadata.extra


def test_neighbor_ids_isolated_per_category() -> None:
    """公式和段落的 prev/next 互不串联。"""
    raw = [
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段一"}]},
        {"type": "equation_interline", "page_idx": 0, "content": r"E=mc^{2}"},
        {"type": "paragraph", "page_idx": 0, "content": [{"type": "text", "content": "段二"}]},
    ]
    chunks = parse_mineru_result(raw, doc_id="d1")
    paras = [c for c in chunks if c.metadata.category == "paragraph"]
    assert len(paras) == 2
    # 两段 paragraph 应当互相 link，公式独立成 chain
    assert paras[0].metadata.extra.get("next_view_id") == paras[1].metadata.extra.get("chunk_id")
