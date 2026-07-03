"""请求级 contextvar：让一次 HTTP 请求产生的所有日志带同一 request_id。

contextvar 是 Python 3.7+ 原生的协程安全上下文存储，FastAPI/asyncio 自动透传。
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

# 一次请求的 contextvar
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_meta_ctx: ContextVar[dict[str, Any] | None] = ContextVar("request_meta", default=None)


def current_request_id() -> str | None:
    return _request_id_ctx.get()


def set_request_id(value: str | None) -> None:
    _request_id_ctx.set(value)


def gen_request_id() -> str:
    """生成新 request_id（uuid4 前 12 位，便于阅读）。"""
    return uuid.uuid4().hex[:12]


def current_request_meta() -> dict[str, Any]:
    return dict(_request_meta_ctx.get() or {})


def set_request_meta(key: str, value: Any) -> None:
    cur = dict(_request_meta_ctx.get() or {})
    cur[key] = value
    _request_meta_ctx.set(cur)


def reset_request_context() -> None:
    """测试用：清空 contextvar。"""
    _request_id_ctx.set(None)
    _request_meta_ctx.set(None)


__all__ = [
    "current_request_id",
    "current_request_meta",
    "gen_request_id",
    "reset_request_context",
    "set_request_id",
    "set_request_meta",
]
