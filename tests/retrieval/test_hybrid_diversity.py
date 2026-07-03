"""hybrid_searcher MinMax + diversity 单测。"""

from __future__ import annotations

from app.models.chunk import ChunkMetadata
from app.models.query import RetrieveResult
from app.retrieval import hybrid_searcher


def _r(doc_id: str, idx: int, score: float) -> RetrieveResult:
    return RetrieveResult(
        content=f"{doc_id}-{idx}",
        score=score,
        doc_id=doc_id,
        chunk_index=idx,
        metadata=ChunkMetadata(doc_id=doc_id, chunk_index=idx),
    )


def test_minmax_makes_dense_and_sparse_comparable() -> None:
    """dense 分数 0~1，sparse 分数 0~10，归一后才能公平加权。"""
    dense = [_r("d", 0, 0.95), _r("d", 1, 0.50)]
    sparse = [_r("d", 0, 9.0), _r("d", 2, 0.5)]

    fused_rrf = hybrid_searcher.fuse(dense, sparse, top_k=5, use_minmax=False)
    fused_mm = hybrid_searcher.fuse(dense, sparse, top_k=5, use_minmax=True)
    # 两种模式都应该返回结果，MinMax 模式不应抛异常
    assert len(fused_rrf) == 3
    assert len(fused_mm) == 3


def test_diversity_penalty_reduces_same_doc_duplicates() -> None:
    """3 个同 doc_id 的 chunk 在 diversity 启用时应被压低（用 RRF 避免 MinMax 把末位变 0）。"""
    dense = [_r("same", i, 1.0 - i * 0.01) for i in range(3)]
    sparse = []

    no_penalty = hybrid_searcher.fuse(dense, sparse, top_k=3, use_minmax=False, diversity_penalty=0.0)
    with_penalty = hybrid_searcher.fuse(dense, sparse, top_k=3, use_minmax=False, diversity_penalty=0.5, diversity_floor=0.1)

    # 第 1 个不变（factor=1.0），第 2、3 个被压低
    assert no_penalty[0].score == with_penalty[0].score
    assert with_penalty[1].score < no_penalty[1].score
    assert with_penalty[2].score < no_penalty[2].score


def test_diversity_floor_limits_how_far_scores_can_drop() -> None:
    dense = [_r("same", i, 1.0) for i in range(5)]
    fused = hybrid_searcher.fuse(
        dense, [], top_k=5, use_minmax=True, diversity_penalty=0.9, diversity_floor=0.5
    )
    # 第 5 个理论上 1 - 0.9*4 = -2.6，但 floor 0.5 兜底
    for r in fused:
        assert r.score >= 0.5 * 0.5 - 1e-6  # floor 应用到分数上
