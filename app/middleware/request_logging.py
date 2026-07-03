"""HTTP 请求日志中间件。

为每个请求：
1. 读取 / 生成 X-Request-ID，注入 contextvar（让子日志自动带）
2. 记录请求开始 / 完成日志：method / path / status / duration_ms
3. 慢请求（> SLOW_REQUEST_MS）打 WARN

不影响业务逻辑；出错时记录到 ERROR 但不拦截。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import logger
from app.core.request_context import gen_request_id, set_request_id, set_request_meta

REQUEST_ID_HEADER = "X-Request-ID"

# 不打访问日志的路径（健康检查、UI 静态、Swagger 等）
_QUIET_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
_QUIET_PREFIXES = ("/ui/static/", "/favicon")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        quiet = path in _QUIET_PATHS or any(path.startswith(p) for p in _QUIET_PREFIXES)

        request_id = request.headers.get(REQUEST_ID_HEADER) or gen_request_id()
        set_request_id(request_id)
        set_request_meta("method", request.method)
        set_request_meta("path", path)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        except Exception:
            status_code = 500
            logger.exception(
                f"request failed method={request.method} path={path}"
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            threshold = float(getattr(settings, "slow_request_ms", 2000))
            if not quiet:
                level = "WARNING" if duration_ms > threshold else "INFO"
                msg = (
                    f"access {request.method} {path} -> {status_code} "
                    f"duration={duration_ms:.1f}ms"
                )
                if duration_ms > threshold:
                    msg += f" SLOW(>{threshold:.0f}ms)"
                logger.log(level, msg)
            set_request_id(None)


__all__ = ["REQUEST_ID_HEADER", "RequestLoggingMiddleware"]
