"""HyDE 单测：覆盖正常调用与未配置场景。"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.retrieval import hyde


def test_maybe_rewrite_passthrough_when_disabled() -> None:
    result, warning = asyncio.get_event_loop().run_until_complete(
        hyde.maybe_rewrite("原查询", use_hyde=False)
    )
    assert result == "原查询"
    assert warning is None


def test_maybe_rewrite_returns_warning_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "hyde_llm_base_url", None)
    monkeypatch.setattr(settings, "hyde_llm_api_key", None)

    result, warning = asyncio.get_event_loop().run_until_complete(
        hyde.maybe_rewrite("原查询", use_hyde=True)
    )
    assert result == "原查询"
    assert warning == "hyde_not_configured"


def test_generate_hypothetical_document_calls_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "hyde_llm_base_url", "https://llm.local")
    monkeypatch.setattr(settings, "hyde_llm_api_key", "key")
    monkeypatch.setattr(settings, "hyde_llm_model", "fake-model")

    captured: dict = {}

    def make_response(payload: dict, status: int = 200) -> httpx.Response:
        request = httpx.Request("POST", "https://llm.local/chat/completions")
        return httpx.Response(status_code=status, json=payload, request=request)

    async def fake_post(self, url, json, headers):  # noqa: ANN001
        captured["url"] = str(url)
        captured["json"] = json
        captured["headers"] = headers
        return make_response({"choices": [{"message": {"content": "假设答案。"}}]})

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        text = asyncio.get_event_loop().run_until_complete(hyde.generate_hypothetical_document("问题"))
    assert text == "假设答案。"
    assert captured["headers"]["Authorization"] == "Bearer key"
    assert captured["json"]["model"] == "fake-model"


def test_generate_hypothetical_document_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "hyde_llm_base_url", "https://llm.local")
    monkeypatch.setattr(settings, "hyde_llm_api_key", "key")

    def make_error_response(status: int) -> httpx.Response:
        request = httpx.Request("POST", "https://llm.local/chat/completions")
        return httpx.Response(status_code=status, text="boom", request=request)

    async def fake_post(self, url, json, headers):  # noqa: ANN001
        return make_error_response(500)

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        try:
            asyncio.get_event_loop().run_until_complete(hyde.generate_hypothetical_document("问题"))
        except hyde.HyDEError as exc:
            assert exc.error_code == "HYDE_FAILED"
        else:
            raise AssertionError("expected HyDEError")
