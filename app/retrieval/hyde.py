"""HyDE：用 LLM 生成假设答案文档，用其 Embedding 去检索。

默认关闭。需要配置 HYDE_LLM_BASE_URL/HYDE_LLM_API_KEY。
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import RagServiceError
from app.core.logging import logger

_HYDE_PROMPT_TEMPLATE = (
    "请基于以下问题，写一段 200 字以内的假设性答案文档，作为知识检索的查询词。"
    "只输出文档内容，不要解释、不要前缀。\n\n问题：{query}\n\n假设答案："
)


class HyDEError(RagServiceError):
    error_code = "HYDE_FAILED"
    http_status = 500


async def generate_hypothetical_document(query: str) -> str:
    base_url = settings.hyde_llm_base_url
    api_key = settings.hyde_llm_api_key
    model = settings.hyde_llm_model
    if not base_url or not api_key:
        raise HyDEError("HyDE LLM not configured")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates hypothetical answers."},
            {"role": "user", "content": _HYDE_PROMPT_TEMPLATE.format(query=query)},
        ],
        "temperature": 0.3,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            response = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
    except Exception as exc:
        raise HyDEError(f"HyDE LLM call failed: {exc}") from exc

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise HyDEError(f"HyDE LLM response shape invalid: {exc}") from exc

    logger.debug(f"HyDE generated doc for query (len={len(text)})")
    return text.strip()


async def maybe_rewrite(query: str, *, use_hyde: bool) -> tuple[str, str | None]:
    """返回 (effective_query, warning)。

    - use_hyde=False：直接返回原 query
    - use_hyde=True 但未配置 LLM：返回原 query + warning
    - use_hyde=True 且配置就绪：返回假设答案文档
    """
    if not use_hyde:
        return query, None
    if not settings.hyde_llm_base_url or not settings.hyde_llm_api_key:
        return query, "hyde_not_configured"
    try:
        rewritten = await generate_hypothetical_document(query)
        return rewritten or query, None
    except HyDEError as exc:
        logger.warning(f"HyDE failed: {exc.message}")
        return query, f"hyde_failed:{exc.error_code.lower()}"


__all__ = ["generate_hypothetical_document", "maybe_rewrite"]
