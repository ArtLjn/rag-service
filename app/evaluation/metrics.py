"""RAG 检索评估指标（借鉴 airQA evaluation/metrics.py）。

毕设范围内 P2 接口预留 + 最小实现：
- recall_at_k：top-k 中包含标注答案的比例
- precision_at_k：top-k 中相关 chunk 的占比
- mrr：第一个相关结果的倒数排名
- ndcg_at_k：归一化折损累积增益（位置折扣 + DCG/iDCG）

输入：
- retrieved：检索返回的 doc_id / chunk_id 列表
- relevant：标注的相关 doc_id / chunk_id 集合

输出：每个指标的浮点值（0.0~1.0）+ 聚合 RetrievalMetrics。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0  # 至少命中一次的比例
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate": self.hit_rate,
            "sample_count": self.sample_count,
        }


def compute_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hit = sum(1 for doc_id in top_k if doc_id in relevant)
    return hit / len(relevant)


def compute_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hit = sum(1 for doc_id in top_k if doc_id in relevant)
    return hit / k


def compute_mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    samples: Iterable[tuple[list[str], set[str]]],
    *,
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """聚合多个 query 的检索结果。

    samples: 每项 (retrieved_doc_ids, relevant_doc_ids)
    """
    ks = k_values or [1, 3, 5, 10]
    agg = RetrievalMetrics()
    recall_sum: dict[int, float] = {k: 0.0 for k in ks}
    precision_sum: dict[int, float] = {k: 0.0 for k in ks}
    ndcg_sum: dict[int, float] = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    hit_count = 0
    sample_count = 0

    for retrieved, relevant in samples:
        if not relevant:
            continue
        sample_count += 1
        for k in ks:
            recall_sum[k] += compute_recall_at_k(retrieved, relevant, k)
            precision_sum[k] += compute_precision_at_k(retrieved, relevant, k)
            ndcg_sum[k] += compute_ndcg_at_k(retrieved, relevant, k)
        mrr_sum += compute_mrr(retrieved, relevant)
        if any(doc_id in relevant for doc_id in retrieved):
            hit_count += 1

    if sample_count == 0:
        return agg

    agg.recall_at_k = {k: recall_sum[k] / sample_count for k in ks}
    agg.precision_at_k = {k: precision_sum[k] / sample_count for k in ks}
    agg.ndcg_at_k = {k: ndcg_sum[k] / sample_count for k in ks}
    agg.mrr = mrr_sum / sample_count
    agg.hit_rate = hit_count / sample_count
    agg.sample_count = sample_count
    return agg


__all__ = [
    "RetrievalMetrics",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_retrieval_metrics",
]
