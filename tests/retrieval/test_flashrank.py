"""FlashRank provider 单测（mock 实际 ONNX 加载）。"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest


def test_build_reranker_flashrank_returns_correct_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """settings.reranker_provider=flashrank 时返回 _FlashRankReranker。"""
    from app.core.config import settings
    from app.retrieval import reranker as reranker_mod

    monkeypatch.setattr(settings, "reranker_enabled", True, raising=False)
    monkeypatch.setattr(settings, "reranker_provider", "flashrank", raising=False)
    monkeypatch.setattr(settings, "reranker_flashrank_model", "rank-T5-flan", raising=False)
    monkeypatch.setattr(settings, "reranker_flashrank_cache_dir", "data/flashrank_cache_test", raising=False)
    reranker_mod.reset_reranker()
    try:
        reranker = reranker_mod.get_reranker()
        assert reranker.provider == "flashrank"
        assert reranker.model_name == "rank-T5-flan"
    finally:
        reranker_mod.reset_reranker()


def test_disabled_reranker_when_not_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """settings.reranker_enabled=False 时返回 _DisabledReranker。"""
    from app.core.config import settings
    from app.retrieval import reranker as reranker_mod

    monkeypatch.setattr(settings, "reranker_enabled", False, raising=False)
    reranker_mod.reset_reranker()
    try:
        reranker = reranker_mod.get_reranker()
        assert reranker.provider == "disabled"
    finally:
        reranker_mod.reset_reranker()


def test_flashrank_compute_scores_via_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlashRank rerank 通过 RerankRequest + Ranker.rerank 返回结果。"""
    from app.core.config import settings
    from app.retrieval import reranker as reranker_mod

    monkeypatch.setattr(settings, "reranker_enabled", True, raising=False)
    monkeypatch.setattr(settings, "reranker_provider", "flashrank", raising=False)
    monkeypatch.setattr(settings, "reranker_flashrank_model", "rank-T5-flan", raising=False)
    monkeypatch.setattr(settings, "reranker_flashrank_cache_dir", "data/flashrank_cache_test", raising=False)

    # 注入 fake flashrank 模块
    fake_module = types.ModuleType("flashrank")

    class FakeRanker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def rerank(self, request):
            # 返回 [{index, score}]，按 passages 顺序
            return [
                {"index": i, "score": 0.9 - i * 0.1}
                for i in range(len(request.passages))
            ]

    class FakeRerankRequest:
        def __init__(self, query, passages):
            self.query = query
            # passages 可能是 list of dict（FlashRank 真实接口）
            self.passages = passages
            if passages and isinstance(passages[0], dict):
                self.passages = [p["text"] for p in passages]

    fake_module.Ranker = FakeRanker
    fake_module.RerankRequest = FakeRerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", fake_module)

    reranker_mod.reset_reranker()
    try:
        results = asyncio.get_event_loop().run_until_complete(
            reranker_mod.rerank("test query", ["doc1", "doc2", "doc3"], top_k=2)
        )
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].metadata["original_index"] == 0
    finally:
        reranker_mod.reset_reranker()
