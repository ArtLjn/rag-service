"""Qdrant 客户端单例。

- 从 settings.qdrant_url 初始化
- 提供 get_client() 单例访问
- 不可用时不抛异常，由调用方 / health 检查处理
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    url = settings.qdrant_url
    api_key = settings.qdrant_api_key
    auth = "on" if api_key else "off"
    logger.info(f"initializing QdrantClient url={url} auth={auth}")
    return QdrantClient(url=url, api_key=api_key, timeout=settings.http_timeout)


def reset_client() -> None:
    """测试用：重置单例缓存。"""
    get_client.cache_clear()


def parse_qdrant_url(url: str) -> dict[str, Any]:
    """将 QDRANT_URL 解析为 host/port/protocol，便于日志输出。"""
    if "://" not in url:
        return {"host": url, "port": None, "scheme": None}
    scheme, rest = url.split("://", 1)
    if ":" in rest:
        host, port = rest.split(":", 1)
        return {"host": host, "port": int(port), "scheme": scheme}
    return {"host": rest, "port": None, "scheme": scheme}
