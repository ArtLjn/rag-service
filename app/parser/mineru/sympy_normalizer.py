"""公式深度规范化：LaTeX → SymPy 校验（借鉴 airQA evaluation/formula_sympy.py）。

策略（懒加载）：
- 未安装 latex2sympy2 / sympy 时自动降级为 no-op，不影响主流程
- 已安装时按「latex2sympy 优先 → sympy.parse_latex 兜底」解析，
  解析成功说明 LaTeX 语法合法，可写入 extra.validated=true
- 失败时记录原因（语法错 / 不支持的命令），不抛异常

毕设范围内：
- 不做完整 AST 等价比较（airQA FormulaComparisonResult 那套）
- 仅做「能解析 = 合法 LaTeX」校验 + 标记 lossy（基础规范化降级）
"""

from __future__ import annotations

from typing import Any

from app.core.logging import logger

# 缺失依赖时不抛异常，调用方按 can_use() 检查
try:
    import sympy  # type: ignore[import-not-found]
    from sympy.parsing.latex import parse_latex  # type: ignore[import-not-found]

    _SIMPY_OK = True
    _SIMPY_ERROR: str | None = None
except ImportError as exc:
    sympy = None  # type: ignore[assignment]
    parse_latex = None  # type: ignore[assignment]
    _SIMPY_OK = False
    _SIMPY_ERROR = str(exc)

try:
    from latex2sympy2 import latex2sympy  # type: ignore[import-not-found]

    _LATEX2SIMPY_OK = True
except ImportError:
    latex2sympy = None  # type: ignore[assignment]
    _LATEX2SIMPY_OK = False


def can_use() -> bool:
    """是否可用 sympy / latex2sympy 做校验。"""
    return _SIMPY_OK


def availability() -> dict[str, Any]:
    """供 /health 与日志展示可用性。"""
    return {
        "sympy": _SIMPY_OK,
        "latex2sympy": _LATEX2SIMPY_OK,
        "error": _SIMPY_ERROR if not _SIMPY_OK else None,
    }


def validate_latex(latex: str) -> dict[str, Any]:
    """校验 LaTeX 是否可被 SymPy 解析。

    返回：
        {validated: bool, lossy: bool, sympy_repr: str, reason: str | None}
        - validated=True：至少一种解析器成功
        - lossy=True：用了 fallback（latex2sympy 失败但 parse_latex 成功，或反之）
        - sympy_repr：解析后的 SymPy 字符串（可用于等价比较，毕设 P2）
    """
    if not latex:
        return {"validated": False, "lossy": False, "sympy_repr": "", "reason": "empty input"}

    if not _SIMPY_OK:
        return {
            "validated": False,
            "lossy": True,
            "sympy_repr": "",
            "reason": f"sympy not installed: {_SIMPY_ERROR}",
        }

    sympy_repr = ""
    lossy = False
    reason: str | None = None

    if _LATEX2SIMPY_OK:
        try:
            expr = latex2sympy(latex)
            if expr is not None:
                sympy_repr = str(expr)
                return {"validated": True, "lossy": False, "sympy_repr": sympy_repr, "reason": None}
        except Exception as exc:
            reason = f"latex2sympy failed: {type(exc).__name__}: {str(exc)[:80]}"
            lossy = True

    try:
        expr = parse_latex(latex)
        if expr is not None:
            sympy_repr = str(expr)
            return {"validated": True, "lossy": lossy, "sympy_repr": sympy_repr, "reason": reason}
    except Exception as exc:
        reason = (reason + " | " if reason else "") + f"parse_latex failed: {type(exc).__name__}: {str(exc)[:80]}"

    return {"validated": False, "lossy": True, "sympy_repr": "", "reason": reason}


def maybe_validate(latex: str, *, enabled: bool = True) -> dict[str, Any]:
    """按开关决定是否校验。供 mineru parser 调用。

    返回 dict 直接合入 chunk.extra（validated / lossy / sympy_repr）。
    """
    if not enabled or not latex:
        return {}
    if not _SIMPY_OK:
        # sympy 未装：静默跳过（不刷错误日志，主流程继续）
        logger.debug("sympy not available, skip formula validation")
        return {}
    result = validate_latex(latex)
    return {
        "validated": result["validated"],
        "lossy": result["lossy"],
        "sympy_repr": result["sympy_repr"][:200] if result["sympy_repr"] else "",
    }


__all__ = [
    "availability",
    "can_use",
    "maybe_validate",
    "validate_latex",
]
