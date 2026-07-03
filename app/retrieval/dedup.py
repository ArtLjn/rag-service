"""Jaccard 去重（借鉴 airQA retrieval/utils.smart_deduplicate）。

阈值 0.7：超过则视为内容重复，保留首个（score 最高的）。
分词策略：保留中文 1-2 字、英文 word、错误码、数字。
"""

from __future__ import annotations

import re

from app.core.config import settings
from app.models.query import RetrieveResult


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    # 英文 word + 数字 + 错误码
    for m in re.findall(r"[A-Za-z][A-Za-z0-9_]*\d*|\d+[A-Za-z]+|\d+", text.lower()):
        if len(m) >= 2:
            tokens.add(m)
    # 中文 bigram
    chinese = re.sub(r"[^一-龥]", "", text)
    for i in range(len(chinese) - 1):
        tokens.add(chinese[i : i + 2])
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup(
    results: list[RetrieveResult],
    *,
    threshold: float | None = None,
) -> list[RetrieveResult]:
    """按内容相似度去重，保留首个出现的（已按 score 降序）。"""
    thr = settings.dedup_jaccard_threshold if threshold is None else threshold
    if thr >= 1.0 or not results:
        return results
    kept: list[RetrieveResult] = []
    kept_tokens: list[set[str]] = []
    for result in results:
        cur_tokens = tokenize(result.content)
        is_dup = False
        for prev_tokens in kept_tokens:
            if jaccard(cur_tokens, prev_tokens) >= thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(result)
            kept_tokens.append(cur_tokens)
    return kept


__all__ = ["dedup", "jaccard", "tokenize"]
