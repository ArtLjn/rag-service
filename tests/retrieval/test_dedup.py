"""dedup 单测。"""

from __future__ import annotations

from app.models.chunk import ChunkMetadata
from app.models.query import RetrieveResult
from app.retrieval.dedup import dedup, jaccard, tokenize


def _r(content: str, doc_id: str = "d", idx: int = 0, score: float = 0.9) -> RetrieveResult:
    return RetrieveResult(content=content, score=score, doc_id=doc_id, chunk_index=idx, metadata=ChunkMetadata(doc_id=doc_id, chunk_index=idx))


def test_tokenize_splits_chinese_bigrams_and_english_words() -> None:
    tokens = tokenize("错误码 E1001 表示交换机宕机")
    assert "错误" in tokens
    assert "宕机" in tokens
    assert "e1001" in tokens


def test_jaccard_identical_sets_returns_one() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard(set(), set()) == 0.0


def test_dedup_keeps_first_when_above_threshold() -> None:
    results = [
        _r("错误码 E1001 表示核心交换机宕机，需立即切换备份。", "doc1", 0, 0.9),
        _r("错误码 E1001 表示核心交换机宕机，需立即切换备份。", "doc1", 1, 0.85),
    ]
    kept = dedup(results, threshold=0.5)
    assert len(kept) == 1
    assert kept[0].doc_id == "doc1" and kept[0].chunk_index == 0


def test_dedup_keeps_distinct_when_below_threshold() -> None:
    results = [
        _r("网络故障排查流程", "doc1", 0, 0.9),
        _r("数据库备份与恢复", "doc2", 0, 0.85),
    ]
    kept = dedup(results, threshold=0.7)
    assert len(kept) == 2


def test_dedup_threshold_one_disables() -> None:
    results = [
        _r("重复内容 1", "doc1", 0, 0.9),
        _r("重复内容 1", "doc1", 1, 0.85),
    ]
    kept = dedup(results, threshold=1.0)
    assert len(kept) == 2
