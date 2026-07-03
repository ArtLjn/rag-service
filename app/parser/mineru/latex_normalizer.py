r"""LaTeX 公式规范化与文本化。

借鉴自 airQA src/chunking/content_preparation.py 与 multiview_classifier.py。
两阶段：
1. `normalize()`：清理 MinerU 输出噪声（合并多余空格、命令体归一）
2. `latex_to_text()`：把 LaTeX 转可读纯文本（\alpha→α、x^2→x²），
   让 BM25 / sparse 检索能命中公式内容（论文实验：检索召回率 +15%~30%）
"""

from __future__ import annotations

import re

_CMD_LETTERS_RE = re.compile(
    r"(\\(?:mathbf|boldsymbol|mathrm|mathit|mathbb|mathcal|bm))\s*\{\s*([^{}]*?)\s*\}"
)
_BRACE_OP_RE = re.compile(r"([\^_])\s*\{\s*([^{}]*?)\s*\}")
_CMD_PREFIX_RE = re.compile(
    r"\\(?:mathrm|mathbf|boldsymbol|mathit|mathbb|mathcal|mathrm|text|mbox|rm|bf|it|cal|bb)\s*\{([^}]*)\}"
)

# LaTeX 命令 → Unicode 符号（覆盖希腊字母 + 常用数学符号）
LATEX_SYMBOL_MAP: dict[str, str] = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\varpi": "ϖ", r"\rho": "ρ", r"\varrho": "ϱ",
    r"\sigma": "σ", r"\varsigma": "ς", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇", r"\forall": "∀",
    r"\exists": "∃", r"\nexists": "∄", r"\in": "∈", r"\notin": "∉",
    r"\ni": "∋", r"\subset": "⊂", r"\supset": "⊃", r"\subseteq": "⊆",
    r"\supseteq": "⊇", r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\to": "→",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\cdot": "·", r"\cdots": "⋯", r"\ldots": "…", r"\dots": "…",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\equiv": "≡", r"\sim": "∼", r"\simeq": "≃", r"\cong": "≅",
    r"\propto": "∝", r"\ll": "≪", r"\gg": "≫",
    r"\prime": "′", r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\oint": "∮", r"\sqrt": "√", r"\angle": "∠", r"\perp": "⊥",
    r"\parallel": "∥", r"\Re": "ℜ", r"\Im": "ℑ", r"\aleph": "ℵ",
    r"\hbar": "ℏ", r"\ell": "ℓ", r"\neg": "¬", r"\therefore": "∴",
    r"\because": "∵",
}

_SUP_MAP = str.maketrans("0123456789-+n=()ijklm", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ⁿ⁼⁽⁾ⁱʲᵏˡᵐ")


def normalize(latex: str) -> str:
    if not latex:
        return latex

    latex = re.sub(r"\^\s*\{\s*\\wedge\s*\}\s*T", "^T", latex)
    latex = re.sub(r"\^\s*\{\s*\\wedge\s*\}", "^T", latex)

    def _merge_cmd_letters(match: re.Match[str]) -> str:
        cmd = match.group(1)
        inner = match.group(2)
        merged = re.sub(r"([A-Za-z])\s+(?=[A-Za-z])", r"\1", inner)
        return f"{cmd}{{{merged}}}"

    latex = _CMD_LETTERS_RE.sub(_merge_cmd_letters, latex)

    for cmd in (r"\mathrm", r"\mathbb", r"\mathcal", r"\bm"):
        latex = latex.replace(cmd + " ", cmd)

    def _merge_brace(match: re.Match[str]) -> str:
        op = match.group(1)
        content = match.group(2)
        if re.search(r"\\(?:text|mbox)\b", content):
            return match.group(0)
        merged = re.sub(r"([A-Za-z])\s+(?=[A-Za-z])", r"\1", content)
        return f"{op}{{{merged}}}"

    latex = _BRACE_OP_RE.sub(_merge_brace, latex)

    latex = re.sub(r"([A-Za-z0-9])\s+_\s*\{\s*([^}]*)\s*\}", r"\1_{\2}", latex)
    latex = re.sub(r"([A-Za-z0-9])\s+\^\s*\{\s*([^}]*)\s*\}", r"\1^{\2}", latex)
    latex = re.sub(r"([A-Za-z0-9])\s+_\s*([A-Za-z0-9])", r"\1_\2", latex)
    latex = re.sub(r"([A-Za-z0-9])\s+\^\s*([A-Za-z0-9])", r"\1^\2", latex)
    latex = re.sub(r"_\s+\{", "_{", latex)
    latex = re.sub(r"\^\s+\{", "^{", latex)

    latex = re.sub(r"\s+", " ", latex)
    latex = re.sub(r"\s*\^\s*", "^", latex)
    return latex.strip()


def latex_to_text(latex: str) -> str:
    r"""把 LaTeX 转可读纯文本（用于 BM25 检索增强）。

    策略（借鉴 airQA multiview_classifier._inline_latex_to_text）：
    1. 剥离 \mathrm{} 等命令包装保留内容
    2. 已知命令替换为 Unicode 符号（\alpha → α）
    3. 清理无参命令残留（\wedge → 字面 wedge）
    4. 去除花括号
    5. 单字符上下标转 Unicode 上标（x²、a¹）
    6. 多字符上下标用 _{} ^{} 连接形式保留
    7. 合并数字空格（"0 . 7" → "0.7"）
    """
    if not latex:
        return ""

    text = latex.strip()
    text = _CMD_PREFIX_RE.sub(r"\1", text)

    for cmd, symbol in LATEX_SYMBOL_MAP.items():
        text = text.replace(cmd, symbol)

    text = re.sub(
        r"\\(wedge|max|min|mathrm|mathbf|boldsymbol|mathit|text|mbox|rm|bf|it|operatorname|mathcal|mathbb|Big|big|left|right)\b",
        r"\1",
        text,
    )

    text = re.sub(r"\{([^{}]*)\}", r"\1", text)

    def _sup_unicode(match: re.Match[str]) -> str:
        ch = match.group(1)
        if len(ch) == 1 and ch in "0123456789-+n=()ijklm":
            return ch.translate(_SUP_MAP)
        return f"^{ch}"

    text = re.sub(r"\^(\w)", _sup_unicode, text)

    text = re.sub(r"_\s*([=0-9])", r"\1", text)
    text = re.sub(r"_\s+(\w)", r"_\1", text)

    text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


__all__ = ["LATEX_SYMBOL_MAP", "latex_to_text", "normalize"]
