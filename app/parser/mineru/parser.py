"""MinerU content_list_v2 → Chunk 列表转换器。

MinerU 输出 JSON 结构（content_list_v2.json 顶层通常是 list）：
- 每项含 `type`、`content`、`bbox`、`page_idx`、`text_level`（标题用）
- 段落 content 是 list of span（每 span 含 `type`、`content`/`text`）
- 表格 content 是 HTML 字符串
- 公式 content 是 LaTeX 字符串

转换策略：
- 用 mineru.constants.map_type_to_category 决定 chunk category（忽略类型直接丢弃）
- 标题用 text_level 推断层级，构建 heading_path
- 表格保留 HTML，并生成 markdown 副本（用 PyMuPDF table extractor 的 _to_markdown 复用）
- 公式走 latex_normalizer 规范化，并附加 latex_to_text 文本化版本到 extra.text（BM25 可命中）
- 公式 / 表格 / 图：调用 _attach_semantic_anchor 把"空间最近的标题"绑到 heading_path
  （借鉴 airQA data_cleaning._establish_semantic_anchors，纯空间距离 + bbox，不用 embedding）
"""

from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.models.chunk import Chunk, ChunkMetadata
from app.parser.mineru import constants as mc
from app.parser.mineru.latex_normalizer import latex_to_text
from app.parser.mineru.latex_normalizer import normalize as normalize_latex

# 同页 / 跨页搜索语义锚点的最大页面跨度（页索引差）
_ANCHOR_MAX_PAGE_LOOKBACK = 2


def parse_mineru_result(
    raw: dict[str, Any] | list[Any],
    *,
    source: str | None = None,
    doc_id: str | None = None,
) -> list[Chunk]:
    """从 MinerU 返回的 JSON 中找 content_list_v2，转 Chunk 列表。"""
    items = _extract_content_list(raw)
    if not items:
        logger.warning("mineru result contains no content_list_v2 items")
        return []

    title_records = _collect_title_records(items)

    chunks: list[Chunk] = []
    heading_path: list[str] = []
    chunk_index = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        mineru_type = item.get("type")
        category = mc.map_type_to_category(mineru_type)
        if category is None:
            continue

        page_idx = int(item.get("page_idx", 0) or 0)
        page = page_idx + 1
        bbox = _coerce_bbox(item.get("bbox"))
        text_level = item.get("text_level") or item.get("level")

        if category == mc.CATEGORY_TITLE:
            title_text = _extract_text(item.get("content"))
            level = _infer_level(text_level, title_text)
            heading_path = heading_path[: level - 1]
            while len(heading_path) < level - 1:
                heading_path.append("")
            heading_path.append(title_text)
            cleaned_path = [p for p in heading_path if p]
            chunks.append(_make_chunk(title_text, source, page, category, cleaned_path, doc_id, chunk_index, bbox))
            chunk_index += 1
            continue

        if category == mc.CATEGORY_TABLE:
            html_content = _extract_html(item.get("content"))
            from app.parser.mineru.table_normalizer import normalize_table_content_detailed

            normalized = normalize_table_content_detailed(html_content)
            markdown = _html_to_markdown(html_content) or normalized.text
            content_str = (
                markdown
                or normalized.text
                or html_content
                or f"[table page={page} bbox={bbox}]"
            )
            anchor_path = _anchor_path_for(item, page_idx, bbox, title_records, fallback=heading_path)
            extra = {
                "table_html": html_content,
                "records": normalized.records,
                "header": normalized.header,
                "caption": normalized.caption,
            }
            if normalized.text and normalized.text not in content_str:
                content_str = f"{content_str}\n{normalized.text}".strip()
            chunks.append(_make_chunk(content_str, source, page, category, anchor_path, doc_id, chunk_index, bbox, extra=extra))
            chunk_index += 1
            continue

        if category == mc.CATEGORY_FORMULA:
            latex_raw = _extract_text(item.get("content"))
            latex = normalize_latex(latex_raw)
            text_repr = latex_to_text(latex)
            content_str = f"$${latex}$$" if latex else ""
            if text_repr and text_repr != latex:
                content_str = f"{content_str}\n{text_repr}".strip()
            anchor_path = _anchor_path_for(item, page_idx, bbox, title_records, fallback=heading_path)
            extra: dict[str, Any] = {"latex": latex}
            if text_repr:
                extra["text"] = text_repr
            from app.core.config import settings
            from app.parser.mineru.sympy_normalizer import maybe_validate

            extra.update(maybe_validate(latex, enabled=settings.formula_validation_enabled))
            chunks.append(_make_chunk(content_str, source, page, category, anchor_path, doc_id, chunk_index, bbox, extra=extra))
            chunk_index += 1
            continue

        if category == mc.CATEGORY_FIGURE:
            caption = _extract_caption_for_figure(items, item)
            if not caption:
                continue
            anchor_path = _anchor_path_for(item, page_idx, bbox, title_records, fallback=heading_path)
            chunks.append(_make_chunk(caption, source, page, category, anchor_path, doc_id, chunk_index, bbox))
            chunk_index += 1
            continue

        # paragraph / list_item / code / header / footer
        text = _extract_text(item.get("content"))
        if not text:
            continue
        chunks.append(_make_chunk(text, source, page, category, [p for p in heading_path if p], doc_id, chunk_index, bbox))
        chunk_index += 1

    logger.info(f"mineru produced {len(chunks)} chunks from {len(items)} items")
    _assign_logic_idx(chunks)
    _attach_neighbor_ids(chunks)
    return chunks


def _assign_logic_idx(chunks: list[Chunk]) -> None:
    """为所有 chunk 分配全局连续 logic_idx（按页码 + chunk_index 排序）。

    借鉴 airQA data_cleaning._assign_logic_idx：跨页连续、单调递增。
    检索结果展示按 logic_idx 排序即还原文档阅读顺序。
    """
    sorted_chunks = sorted(chunks, key=lambda c: ((c.metadata.page or 0), c.metadata.chunk_index))
    for idx, chunk in enumerate(sorted_chunks):
        chunk.metadata.extra["logic_idx"] = idx


def _attach_neighbor_ids(chunks: list[Chunk]) -> None:
    """为每个 chunk 写入 prev_view_id/next_view_id（基于 logic_idx 找同 category 的前后）。

    借鉴 airQA retrieval/context_enhancer 的逻辑邻居思路：
    - 入库阶段预计算 prev/next（同 category 的相邻 chunk_id）
    - 检索阶段可基于此扩窗，避免「最小切片丢上下文」
    """
    by_category: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_category.setdefault(chunk.metadata.category, []).append(chunk)
    for group in by_category.values():
        group.sort(key=lambda c: c.metadata.extra.get("logic_idx", c.metadata.chunk_index))
        # 第一轮：先把 chunk_id 全部赋上（避免取 next 时取到 None）
        for chunk in group:
            doc_id = chunk.metadata.doc_id or ""
            chunk_idx = chunk.metadata.chunk_index
            fallback_idx = chunk.metadata.extra.get("logic_idx", chunk_idx)
            chunk_id = f"{doc_id}:{chunk_idx}" if doc_id else f"idx:{fallback_idx}"
            chunk.metadata.extra["chunk_id"] = chunk_id
        # 第二轮：基于已就绪的 chunk_id 设置 prev/next
        for i, chunk in enumerate(group):
            if i > 0:
                chunk.metadata.extra["prev_view_id"] = group[i - 1].metadata.extra["chunk_id"]
            if i < len(group) - 1:
                chunk.metadata.extra["next_view_id"] = group[i + 1].metadata.extra["chunk_id"]


def _collect_title_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """收集所有标题项，用于后续语义锚点查找。"""
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if mc.map_type_to_category(item.get("type")) != mc.CATEGORY_TITLE:
            continue
        records.append(
            {
                "item_idx": idx,
                "page_idx": int(item.get("page_idx", 0) or 0),
                "bbox": _coerce_bbox(item.get("bbox")),
                "text": _extract_text(item.get("content")),
                "level": _infer_level(item.get("text_level") or item.get("level"), _extract_text(item.get("content"))),
            }
        )
    return records


def _anchor_path_for(
    target_item: dict[str, Any],
    target_page_idx: int,
    target_bbox: tuple[float, float, float, float] | None,
    title_records: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    """找空间最近的标题作为语义锚点。

    规则：
    1. 候选 = 同页或前 N 页、bbox 在 target 上方（y1 <= target.y0）的标题
    2. 距离 = bbox 中心点欧氏距离（同页加权 0.5，跨页加权 1.5）
    3. 取距离最近者，重建 heading_path（用其 level 截断）
    4. 无候选则用当前累积的 heading_path（fallback）
    """
    if not target_bbox or not title_records:
        return [p for p in fallback if p]

    tx, ty = (target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2

    best: tuple[float, dict[str, Any]] | None = None
    for rec in title_records:
        page_delta = target_page_idx - rec["page_idx"]
        if page_delta < -1 or page_delta > _ANCHOR_MAX_PAGE_LOOKBACK:
            continue
        if not rec["bbox"]:
            continue
        if rec["bbox"][1] > target_bbox[1] and page_delta <= 0:
            # 同页时锚点必须在 target 上方
            continue
        rx = (rec["bbox"][0] + rec["bbox"][2]) / 2
        ry = (rec["bbox"][1] + rec["bbox"][3]) / 2
        distance = ((rx - tx) ** 2 + (ry - ty) ** 2) ** 0.5
        weight = 0.5 if page_delta == 0 else (1.0 + abs(page_delta) * 0.5)
        weighted = distance * weight
        if best is None or weighted < best[0]:
            best = (weighted, rec)

    if best is None:
        return [p for p in fallback if p]

    winner = best[1]
    return _rebuild_path_to_level(title_records, winner)


def _rebuild_path_to_level(title_records: list[dict[str, Any]], winner: dict[str, Any]) -> list[str]:
    """从 winner 反向重建 heading_path（按 level 找每个层级最近的、item_idx < winner 的标题）。"""
    level = winner["level"]
    path: list[str] = [""] * level
    path[level - 1] = winner["text"]
    winner_idx = winner["item_idx"]
    for current_level in range(level - 1, 0, -1):
        # 找 item_idx < winner_idx 且 level=current_level 的最近一个
        for rec in reversed(title_records):
            if rec["item_idx"] >= winner_idx:
                continue
            if rec["level"] != current_level:
                continue
            path[current_level - 1] = rec["text"]
            winner_idx = rec["item_idx"]
            break
    return [p for p in path if p]


def _extract_content_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("content_list_v2", "content_list", "list"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    return []


def _coerce_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1)


def _extract_text(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for span in content:
            if not isinstance(span, dict):
                continue
            span_type = span.get("type")
            if span_type and span_type not in mc.KEEP_SPAN_TYPES:
                continue
            text = span.get("content") or span.get("text") or ""
            if span_type in {"equation_inline", "inline_equation"} and text:
                text = f"${text}$"
            if text:
                parts.append(str(text))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "math_content"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value.strip()
            if isinstance(value, list):
                return _extract_text(value)
    return ""


def _extract_html(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for span in content:
            if isinstance(span, dict) and span.get("type") in mc.KEEP_SPAN_TYPES:
                continue
            html = span.get("content") or span.get("html") or ""
            if html and "<" in html:
                return str(html)
    if isinstance(content, dict):
        for key in ("html", "table_html", "content"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _html_to_markdown(html: str) -> str:
    if not html or "<table" not in html.lower():
        return ""
    import re as _re

    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=_re.S | _re.I)
    if not rows:
        return ""
    parsed: list[list[str]] = []
    for row in rows:
        cells = _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=_re.S | _re.I)
        cells = [_re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        parsed.append(cells)
    if not parsed:
        return ""
    header = parsed[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in parsed[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _infer_level(level: Any, title_text: str) -> int:
    if isinstance(level, int) and level >= 1:
        return level
    if isinstance(level, str) and level.isdigit():
        return max(1, int(level))
    head = title_text.split(" ", 1)[0] if title_text else ""
    if head and all(ch.isdigit() or ch == "." for ch in head) and any(ch.isdigit() for ch in head):
        dots = head.count(".")
        return min(6, max(1, dots + 1))
    return 1


def _extract_caption_for_figure(items: list[dict[str, Any]], figure_item: dict[str, Any]) -> str:
    fbbox = _coerce_bbox(figure_item.get("bbox"))
    if not fbbox:
        return ""
    fy = (fbbox[1] + fbbox[3]) / 2
    fx = (fbbox[0] + fbbox[2]) / 2
    candidates: list[tuple[float, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if mc.map_type_to_category(item.get("type")) != mc.CATEGORY_PARAGRAPH:
            continue
        text = _extract_text(item.get("content"))
        if not text or not _looks_like_caption(text):
            continue
        bbox = _coerce_bbox(item.get("bbox"))
        if not bbox:
            continue
        ey = (bbox[1] + bbox[3]) / 2
        ex = (bbox[0] + bbox[2]) / 2
        distance = ((ey - fy) ** 2 + (ex - fx) ** 2) ** 0.5
        candidates.append((distance, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _looks_like_caption(text: str) -> bool:
    head = text.lstrip()[:8].lower()
    return head.startswith("图") or head.startswith("fig")


def _make_chunk(
    content: str,
    source: str | None,
    page: int,
    category: str,
    heading_path: list[str],
    doc_id: str | None,
    chunk_index: int,
    bbox: tuple[float, float, float, float] | None,
    *,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    metadata = ChunkMetadata(
        source=source,
        page=page,
        category=category,
        heading_path=heading_path,
        doc_id=doc_id,
        chunk_index=chunk_index,
        extra=extra or {},
    )
    if bbox:
        metadata.extra["bbox"] = list(bbox)
    return Chunk(content=content, metadata=metadata)


__all__ = ["parse_mineru_result"]



def _extract_content_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("content_list_v2", "content_list", "list"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    return []


def _coerce_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1)


def _extract_text(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for span in content:
            if not isinstance(span, dict):
                continue
            span_type = span.get("type")
            if span_type and span_type not in mc.KEEP_SPAN_TYPES:
                # 跳过 span 级别的图、表内嵌
                continue
            text = span.get("content") or span.get("text") or ""
            if span_type in {"equation_inline", "inline_equation"} and text:
                text = f"${text}$"
            if text:
                parts.append(str(text))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "math_content"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value.strip()
            if isinstance(value, list):
                return _extract_text(value)
    return ""


def _extract_html(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for span in content:
            if isinstance(span, dict) and span.get("type") in mc.KEEP_SPAN_TYPES:
                continue
            html = span.get("content") or span.get("html") or ""
            if html and "<" in html:
                return str(html)
    if isinstance(content, dict):
        for key in ("html", "table_html", "content"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _html_to_markdown(html: str) -> str:
    """简易 HTML 表格转 Markdown（最小实现，复用 airQA 思路）。"""
    if not html or "<table" not in html.lower():
        return ""
    import re as _re

    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=_re.S | _re.I)
    if not rows:
        return ""
    parsed: list[list[str]] = []
    for row in rows:
        cells = _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=_re.S | _re.I)
        cells = [_re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        parsed.append(cells)
    if not parsed:
        return ""
    header = parsed[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in parsed[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _infer_level(level: Any, title_text: str) -> int:
    if isinstance(level, int) and level >= 1:
        return level
    if isinstance(level, str) and level.isdigit():
        return max(1, int(level))
    head = title_text.split(" ", 1)[0] if title_text else ""
    if head and all(ch.isdigit() or ch == "." for ch in head) and any(ch.isdigit() for ch in head):
        dots = head.count(".")
        return min(6, max(1, dots + 1))
    return 1


def _extract_caption_for_figure(items: list[dict[str, Any]], figure_item: dict[str, Any]) -> str:
    fbbox = _coerce_bbox(figure_item.get("bbox"))
    if not fbbox:
        return ""
    fy = (fbbox[1] + fbbox[3]) / 2
    fx = (fbbox[0] + fbbox[2]) / 2
    candidates: list[tuple[float, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if mc.map_type_to_category(item.get("type")) != mc.CATEGORY_PARAGRAPH:
            continue
        text = _extract_text(item.get("content"))
        if not text or not _looks_like_caption(text):
            continue
        bbox = _coerce_bbox(item.get("bbox"))
        if not bbox:
            continue
        ey = (bbox[1] + bbox[3]) / 2
        ex = (bbox[0] + bbox[2]) / 2
        distance = ((ey - fy) ** 2 + (ex - fx) ** 2) ** 0.5
        candidates.append((distance, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _looks_like_caption(text: str) -> bool:
    head = text.lstrip()[:8].lower()
    return head.startswith("图") or head.startswith("fig")


def _make_chunk(
    content: str,
    source: str | None,
    page: int,
    category: str,
    heading_path: list[str],
    doc_id: str | None,
    chunk_index: int,
    bbox: tuple[float, float, float, float] | None,
    *,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    metadata = ChunkMetadata(
        source=source,
        page=page,
        category=category,
        heading_path=heading_path,
        doc_id=doc_id,
        chunk_index=chunk_index,
        extra=extra or {},
    )
    if bbox:
        metadata.extra["bbox"] = list(bbox)
    return Chunk(content=content, metadata=metadata)


__all__ = ["parse_mineru_result"]
