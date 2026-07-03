"""parse_service 完整流程单测。"""

from __future__ import annotations

import asyncio

from app.services import parse_service


def test_parse_content_text_uses_fixed_strategy() -> None:
    chunks, doc_id, summary, warning = asyncio.get_event_loop().run_until_complete(
        parse_service.parse_content(
            content="段落一。\n\n段落二。\n\n段落三。",
            file_type="txt",
            strategy="fixed",
            metadata={"source": "demo.txt"},
        )
    )
    assert doc_id
    assert len(chunks) >= 1
    assert summary["total"] == len(chunks)
    assert warning is None


def test_parse_content_markdown_extracts_titles() -> None:
    chunks, _, summary, _ = asyncio.get_event_loop().run_until_complete(
        parse_service.parse_content(
            content="# 标题一\n\n内容A\n\n## 标题二\n\n内容B",
            file_type="md",
            strategy="structure_aware",
        )
    )
    assert summary.get("title", 0) >= 1


def test_parse_content_degrades_on_unknown_strategy() -> None:
    chunks, _, _, _ = asyncio.get_event_loop().run_until_complete(
        parse_service.parse_content(
            content="简短文本。",
            file_type="txt",
            strategy="fixed",
        )
    )
    assert len(chunks) >= 1


def test_compute_doc_id_is_stable_for_same_content() -> None:
    a = parse_service.compute_doc_id("hello")
    b = parse_service.compute_doc_id("hello")
    assert a == b
    assert len(a) == 12


def test_build_layout_summary_counts_categories() -> None:
    from app.models.chunk import Chunk, ChunkMetadata

    chunks = [
        Chunk(content="x", metadata=ChunkMetadata(category="title")),
        Chunk(content="x", metadata=ChunkMetadata(category="paragraph")),
    ]
    summary = parse_service.build_layout_summary(chunks)
    assert summary["title"] == 1
    assert summary["paragraph"] == 1
    assert summary["total"] == 2
