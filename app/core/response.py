"""统一响应模型与降级标记。"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: str = "OK"
    message: str = "ok"
    data: T | None = None
    warning: str | None = None
    error_code: str | None = None

    @classmethod
    def ok(cls, data: Any, message: str = "ok", warning: str | None = None) -> ApiResponse[Any]:
        return cls(code="OK", message=message, data=data, warning=warning)

    @classmethod
    def failed(
        cls,
        message: str,
        *,
        error_code: str = "FAILED",
        data: Any = None,
        warning: str | None = None,
    ) -> ApiResponse[Any]:
        return cls(code="FAILED", message=message, data=data, error_code=error_code, warning=warning)


class HealthResponse(BaseModel):
    status: str = "ok"
    components: dict[str, str] = Field(default_factory=dict)
    warning: str | None = None
