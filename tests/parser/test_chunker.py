"""分块器与 cleaner 单测。"""

from __future__ import annotations

from app.models.chunk import Chunk, ChunkMetadata
from app.models.query import ChunkingStrategy
from app.parser.chunker import chunk_with_strategy, select_default_strategy
from app.parser.chunker.fixed import chunk as fixed_chunk
from app.parser.chunker.semantic import chunk as semantic_chunk
from app.parser.chunker.structure_aware import chunk as structure_chunk
from app.parser.cleaner import clean as clean_chunks


def _mk(content: str, *, category: str = "paragraph", path: list[str] | None = None, idx: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            category=category,
            heading_path=path or [],
            chunk_index=idx,
        ),
    )


def test_select_default_strategy_matches_file_type() -> None:
    assert select_default_strategy("pdf") == ChunkingStrategy.STRUCTURE_AWARE
    assert select_default_strategy("md") == ChunkingStrategy.SEMANTIC
    assert select_default_strategy("txt") == ChunkingStrategy.FIXED


def test_fixed_chunker_respects_size_and_overlap() -> None:
    content = "字" * 1200
    chunks = fixed_chunk([_mk(content)], chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    assert all(c.metadata.chunk_index == i for i, c in enumerate(chunks))


def test_structure_aware_merges_under_same_heading() -> None:
    raw = [
        _mk("标题A", category="title", path=["标题A"], idx=0),
        _mk("段落1。", path=["标题A"], idx=1),
        _mk("段落2。", path=["标题A"], idx=2),
        _mk("段落3。", path=["标题A"], idx=3),
    ]
    merged = structure_chunk(raw)
    paragraph_chunks = [c for c in merged if c.metadata.category == "paragraph"]
    assert len(paragraph_chunks) == 1
    assert "段落1" in paragraph_chunks[0].content
    assert "段落3" in paragraph_chunks[0].content


def test_semantic_chunker_groups_by_similarity() -> None:
    raw = [_mk("这是关于 RAG 的介绍。这是关于 RAG 的细节。这是关于服务的部署。")]
    chunks = semantic_chunk(raw)
    assert len(chunks) >= 1
    assert all(c.metadata.category == "paragraph" for c in chunks)


def test_chunk_with_strategy_unknown_falls_back_to_fixed() -> None:
    raw = [_mk("内容" * 400)]
    chunks = chunk_with_strategy(raw, ChunkingStrategy.FIXED, options={"chunk_size": 200, "chunk_overlap": 20})
    assert len(chunks) >= 2


def test_cleaner_removes_ocr_noise_and_merges_short() -> None:
    raw = [
        _mk("正常段落，足够长，应该被保留。", idx=0),
        _mk("□■", idx=1),
        _mk("短。", idx=2),
    ]
    cleaned = clean_chunks(raw)
    assert all("□" not in c.content and "■" not in c.content for c in cleaned)
    assert len(cleaned) <= len(raw)
    assert [c.metadata.chunk_index for c in cleaned] == list(range(len(cleaned)))
