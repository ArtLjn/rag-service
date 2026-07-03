"""元数据清洗：断行、空白、OCR 噪声、过短 chunk 合并。"""

from __future__ import annotations

import re
import unicodedata

from app.models.chunk import Chunk

_BROKEN_LINE_RE = re.compile(r"-\n")
_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_OCR_NOISE_CHARS = {"□", "■", "◆", "◇", "▪", "▫", "●", "○", "☐"}
_ISOLATED_NOISE_RE = re.compile(r"^\s*[□■◆◇▪▫●○☐]\s*$", re.MULTILINE)
_MIN_CHUNK_CHARS = 30


def clean(chunks: list[Chunk], *, min_chars: int = _MIN_CHUNK_CHARS) -> list[Chunk]:
    if not chunks:
        return []

    cleaned: list[Chunk] = []
    for chunk in chunks:
        text = chunk.content
        text = _BROKEN_LINE_RE.sub("", text)
        text = _NEWLINE_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text)
        text = _ISOLATED_NOISE_RE.sub("", text)
        text = "".join(ch for ch in text if ch not in _OCR_NOISE_CHARS)
        text = unicodedata.normalize("NFKC", text).strip()
        if not text:
            continue
        cleaned.append(chunk.model_copy(update={"content": text}))

    merged = _merge_short_chunks(cleaned, min_chars)
    return _reindex(merged)


def _merge_short_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    if not chunks:
        return chunks
    result: list[Chunk] = []
    for chunk in chunks:
        if (
            chunk.metadata.category == "paragraph"
            and len(chunk.content) < min_chars
            and result
            and result[-1].metadata.category == "paragraph"
        ):
            prev = result[-1]
            merged = prev.model_copy(update={"content": f"{prev.content} {chunk.content}".strip()})
            result[-1] = merged
            continue
        result.append(chunk)
    return result


def _reindex(chunks: list[Chunk]) -> list[Chunk]:
    reindexed: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata.model_copy(update={"chunk_index": i})
        reindexed.append(chunk.model_copy(update={"metadata": metadata}))
    return reindexed


__all__ = ["clean"]
