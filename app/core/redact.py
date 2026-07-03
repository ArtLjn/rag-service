"""敏感字段脱敏工具。

API key / token / password / URL 中的凭证信息在写入日志前应当过滤。
用法：
    from app.core.redact import redact
    logger.info(f"qdrant url={redact.url(settings.qdrant_url)}")
    logger.info(f"api key={redact.secret(settings.qdrant_api_key)}")
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization|bearer)"
)


def secret(value: str | None, *, keep_prefix: int = 4, keep_suffix: int = 4) -> str:
    """将 secret 截断成 `xxxx...yyyy` 形式。"""
    if not value:
        return "(empty)"
    s = str(value)
    if len(s) <= keep_prefix + keep_suffix:
        return "***"
    return f"{s[:keep_prefix]}...{s[-keep_suffix:]}"


def url(raw: str | None) -> str:
    """去除 URL 中 user:password 部分。"""
    if not raw:
        return "(empty)"
    try:
        parsed = urlparse(str(raw))
        if parsed.password or parsed.username:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc += f":{parsed.port}"
            cleaned = parsed._replace(netloc=netloc)
            return urlunparse(cleaned)
        return str(raw)
    except Exception:
        return "(invalid-url)"


def dict_keys(payload: dict | None) -> dict:
    """对 dict 中疑似 secret 的 key 做脱敏。"""
    if not isinstance(payload, dict):
        return payload
    redacted: dict = {}
    for k, v in payload.items():
        if isinstance(k, str) and _SECRET_PATTERN.search(k):
            redacted[k] = secret(str(v)) if v else v
        elif isinstance(v, dict):
            redacted[k] = dict_keys(v)
        else:
            redacted[k] = v
    return redacted


__all__ = ["dict_keys", "secret", "url"]
