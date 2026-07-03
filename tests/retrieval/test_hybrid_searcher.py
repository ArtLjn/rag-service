"""hybrid_searcher 单测：验证 RRF 融合公式与权重。"""

from __future__ import annotations

from app.models.query import RetrieveResult
from app.retrieval import hybrid_searcher


def _r(doc_id: str, idx: int, score: float, content: str = "") -> RetrieveResult:
    from app.models.chunk import ChunkMetadata

    return RetrieveResult(
        content=content or doc_id,
        score=score,
        doc_id=doc_id,
        chunk_index=idx,
        metadata=ChunkMetadata(doc_id=doc_id, chunk_index=idx),
    )


def test_rrf_favors_results_present_in_both_lists() -> None:
    dense = [_r("doc", 0, 0.9, "a"), _r("doc", 1, 0.85, "b")]
    sparse = [_r("doc", 0, 5.0, "a"), _r("doc", 2, 4.0, "c")]

    fused = hybrid_searcher.fuse(dense, sparse, top_k=3)

    assert fused[0].doc_id == "doc" and fused[0].chunk_index == 0
    assert fused[0].score > fused[1].score


def test_rrf_weights_change_ranking() -> None:
    dense = [_r("doc", 0, 0.9, "a")]
    sparse = [_r("doc", 1, 0.0, "b")]

    fused_default = hybrid_searcher.fuse(dense, sparse, top_k=2)
    fused_sparse_heavy = hybrid_searcher.fuse(dense, sparse, top_k=2, weights=(0.1, 0.9))

    assert fused_default[0].chunk_index == 0
    assert fused_sparse_heavy[0].chunk_index == 1


def test_normalize_weights_handles_zero_total() -> None:
    a, b = hybrid_searcher.normalize_weights(0.0, 0.0)
    assert a > 0 and b > 0
    total = sum(hybrid_searcher.normalize_weights(2.0, 3.0))
    assert abs(total - 1.0) < 1e-6


def test_fuse_caps_at_top_k() -> None:
    dense = [_r(f"d{i}", 0, 1.0 - i * 0.01) for i in range(20)]
    sparse = list(reversed(dense))
    fused = hybrid_searcher.fuse(dense, sparse, top_k=5)
    assert len(fused) == 5
