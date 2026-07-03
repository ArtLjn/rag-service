"""evaluation/metrics 单测。"""

from __future__ import annotations

from app.evaluation.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)


def test_recall_at_k_full_hit() -> None:
    assert compute_recall_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0
    assert compute_recall_at_k(["a", "b"], {"a", "c"}, k=2) == 0.5


def test_recall_empty_relevant_returns_zero() -> None:
    assert compute_recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_at_k() -> None:
    assert compute_precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 2 / 3


def test_mrr_returns_reciprocal_of_first_hit() -> None:
    assert compute_mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert compute_mrr(["a", "b"], {"a"}) == 1.0
    assert compute_mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_ideal_ranking_returns_one() -> None:
    assert compute_ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_suboptimal_ranking_below_one() -> None:
    score = compute_ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert 0.0 < score < 1.0


def test_compute_retrieval_metrics_aggregates() -> None:
    samples = [
        (["a", "b", "c"], {"a"}),
        (["x", "y", "z"], {"a"}),
        (["a", "x", "y"], {"a"}),
    ]
    agg = compute_retrieval_metrics(samples, k_values=[1, 3])
    assert agg.sample_count == 3
    assert agg.recall_at_k[1] == 2 / 3  # 2/3 在 top-1 命中
    assert agg.recall_at_k[3] == 2 / 3
    # mrr 是平均值：(1.0 + 0.0 + 1.0) / 3
    assert abs(agg.mrr - 2 / 3) < 1e-6


def pytest_approx(value: float, *, eps: float = 1e-6):
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(other - value) < eps

    return _Approx()
