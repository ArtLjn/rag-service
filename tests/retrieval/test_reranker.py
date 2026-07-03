"""reranker 单测：mock CrossEncoder，验证模型加载失败时降级。"""

from __future__ import annotations

import asyncio

import pytest

from app.core.exceptions import RerankerUnavailable
from app.retrieval import reranker as reranker_mod
from app.services import rerank_service


def test_rerank_returns_top_k_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReranker:
        def compute_scores_sync(self, query: str, docs: list[str]) -> list[float]:
            return [float(len(d)) for d in docs]

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: FakeReranker())

    results = asyncio.get_event_loop().run_until_complete(
        reranker_mod.rerank("q", ["a", "bb", "ccc"], top_k=2)
    )
    assert len(results) == 2
    assert results[0].content == "ccc"
    assert results[0].score > results[1].score


def test_rerank_service_degrades_when_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_, **__):
        raise RerankerUnavailable("model missing")

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: type("Bad", (), {"compute_scores_sync": boom})())

    results, warning = asyncio.get_event_loop().run_until_complete(
        rerank_service.rerank_documents(query="q", documents=["a", "b"], top_k=2)
    )
    assert warning == "reranker_degraded"
    assert results[0].metadata["original_index"] == 0
    assert results[1].metadata["original_index"] == 1


def test_rerank_handles_mixed_str_and_dict_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReranker:
        def compute_scores_sync(self, query: str, docs: list[str]) -> list[float]:
            return [0.1 * i for i in range(len(docs))]

    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: FakeReranker())

    docs: list = ["plain text", {"content": "dict text", "source": "x"}]
    results = asyncio.get_event_loop().run_until_complete(
        reranker_mod.rerank("q", docs, top_k=2)
    )
    assert len(results) == 2
    assert results[0].content == "dict text"
    assert results[0].metadata.get("source") == "x"
