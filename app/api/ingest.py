"""POST /ingest — 解析 → 分块 → 向量化 → 写入 Qdrant + SQLite。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.core.logging import logger
from app.core.metrics import metrics
from app.core.response import ApiResponse
from app.models.query import ChunkingStrategy
from app.parser.base import detect_file_type
from app.services.ingest_service import ingest_content

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
async def ingest(
    collection: str = Form(...),
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
    file_type: str | None
    if file is not None:
        content = await file.read()
        resolved_file_type = file_type or detect_file_type(file.filename, file.content_type)
        if source is None and file.filename:
            source = file.filename
    elif text is not None and text.strip():
        content = text
        from app.api.parse import _infer_text_type

        resolved_file_type = file_type or _infer_text_type(text, source)
    else:
        from app.core.exceptions import UnsupportedFormat

        raise UnsupportedFormat("must provide 'file' or 'text'")

    strategy_enum = _safe_strategy(strategy)
    metadata: dict[str, Any] = {"source": source, "category": category}

    with metrics.time("ingest_latency_seconds", "ingest endpoint latency"):
        result = await ingest_content(
            content=content,
            collection=collection,
            file_type=resolved_file_type,
            strategy=strategy_enum,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    metrics.counter("ingest_total").inc()
    metrics.counter("ingest_chunks_total").inc(result.get("chunk_count", 0))
    logger.info(f"/ingest collection={collection} doc_id={result.get('doc_id')} chunks={result.get('chunk_count')}")

    return ApiResponse.ok(result)


def _safe_strategy(value: str | None) -> ChunkingStrategy | None:
    if not value:
        return None
    try:
        return ChunkingStrategy(value)
    except ValueError:
        return None
