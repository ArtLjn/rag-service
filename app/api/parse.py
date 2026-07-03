"""POST /parse — 解析与分块，不写库。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.logging import logger
from app.core.metrics import metrics
from app.core.response import ApiResponse
from app.models.query import ChunkingStrategy
from app.parser.base import detect_file_type
from app.services.parse_service import parse_content

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("")
async def parse(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    file_type: str | None = Form(default=None),
    strategy: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    source: str | None = Form(default=None),
    category: str | None = Form(default=None),
) -> ApiResponse[dict[str, Any]]:
    content: bytes | str
    if file is not None:
        content = await file.read()
        file_type = file_type or detect_file_type(file.filename, file.content_type)
    elif text is not None and text.strip():
        content = text
        file_type = file_type or _infer_text_type(text, source)
    else:
        from app.core.exceptions import UnsupportedFormat

        raise UnsupportedFormat("must provide 'file' or 'text'")

    strategy_enum = _safe_strategy(strategy) if strategy else None
    metadata: dict[str, Any] = {"source": source, "category": category}
    if file is not None and file.filename:
        metadata["source"] = source or file.filename

    with metrics.time("parse_latency_seconds", "parse endpoint latency"):
        chunks, doc_id, layout_summary, warning = await parse_content(
            content=content,
            file_type=file_type,
            strategy=strategy_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata,
        )
    metrics.counter("parse_total").inc()
    logger.info(f"/parse doc_id={doc_id} chunks={len(chunks)}")

    return ApiResponse.ok(
        {
            "doc_id": doc_id,
            "chunks": [c.model_dump(mode="json") for c in chunks],
            "layout_summary": layout_summary,
        },
        warning=warning,
    )


def _safe_strategy(value: str | None) -> ChunkingStrategy | None:
    if not value:
        return None
    try:
        return ChunkingStrategy(value)
    except ValueError:
        logger.warning(f"unknown strategy {value}, will fallback to default")
        return None


_HEADING_HINT_PREFIXES = ("#", "标题")
_HEADING_HINT_LINES = ("# ", "## ", "### ")


def _infer_text_type(text: str, source: str | None) -> str:
    if source:
        suffix = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        if suffix in {"md", "markdown"}:
            return "md"
        if suffix in {"txt", "text"}:
            return "txt"
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped and any(stripped.startswith(prefix) for prefix in _HEADING_HINT_LINES):
            return "md"
    return "txt"
