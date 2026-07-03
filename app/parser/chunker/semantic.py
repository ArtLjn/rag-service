"""语义分块（基于句子 Embedding 相似度的边界检测）。

NSQA 项目不可访问，本实现采用轻量版：
- 用 jieba 分句
- 用段落级别的相邻句子相似度（基于词汇重合 + 长度惩罚，避免依赖 Embedding 模型加载）
- 相似度低于阈值时切分

完整版本（接 Embedding）作为扩展点保留接口。
"""

from __future__ import annotations

import re

import jieba

from app.core.logging import logger
from app.models.chunk import Chunk, ChunkMetadata

SENTENCE_ENDINGS = re.compile(r"([。！？!?；;\n]+)")
SIMILARITY_THRESHOLD = 0.20
TARGET_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 800


def chunk(chunks: list[Chunk]) -> list[Chunk]:
    """对原始 chunks 做语义分块。"""
    if not chunks:
        return []

    raw_text = "\n\n".join(c.content for c in chunks if c.metadata.category != "title")
    if not raw_text.strip():
        return chunks

    sentences = _split_sentences(raw_text)
    if not sentences:
        return chunks

    groups = _group_by_similarity(sentences)

    base = chunks[0].metadata
    result: list[Chunk] = []
    chunk_index = 0
    for group in groups:
        text = " ".join(group).strip()
        if not text:
            continue
        if len(text) > MAX_CHUNK_CHARS:
            for sub in _split_long(text, TARGET_CHUNK_CHARS):
                result.append(_make_chunk(sub, base, chunk_index))
                chunk_index += 1
            continue
        result.append(_make_chunk(text, base, chunk_index))
        chunk_index += 1
    return result


def _split_sentences(text: str) -> list[str]:
    pieces = SENTENCE_ENDINGS.split(text)
    sentences: list[str] = []
    buffer = ""
    for piece in pieces:
        if not piece:
            continue
        if SENTENCE_ENDINGS.fullmatch(piece):
            buffer += piece
            if buffer.strip():
                sentences.append(buffer.strip())
            buffer = ""
        else:
            buffer += piece
    if buffer.strip():
        sentences.append(buffer.strip())
    return [s for s in sentences if s]


def _tokenize(text: str) -> set[str]:
    return {t for t in jieba.lcut(text) if len(t.strip()) > 1}


def _group_by_similarity(sentences: list[str]) -> list[list[str]]:
    if not sentences:
        return []
    groups: list[list[str]] = [[sentences[0]]]
    prev_tokens = _tokenize(sentences[0])
    for sentence in sentences[1:]:
        cur_tokens = _tokenize(sentence)
        sim = _jaccard(prev_tokens, cur_tokens)
        cur_len = sum(len(s) for s in groups[-1])
        if sim < SIMILARITY_THRESHOLD or cur_len + len(sentence) > MAX_CHUNK_CHARS:
            groups.append([sentence])
            prev_tokens = cur_tokens
            continue
        groups[-1].append(sentence)
        prev_tokens = cur_tokens | prev_tokens
    return groups


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _split_long(text: str, target: int) -> list[str]:
    if len(text) <= target:
        return [text]
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = cursor + target
        if end >= len(text):
            pieces.append(text[cursor:].strip())
            break
        cut = text.rfind("。", cursor, end)
        if cut == -1 or cut < cursor + target // 2:
            cut = text.rfind(" ", cursor, end)
        if cut == -1 or cut < cursor + target // 2:
            cut = end
        pieces.append(text[cursor:cut].strip())
        cursor = cut
    return [p for p in pieces if p]


def _make_chunk(content: str, base: ChunkMetadata, chunk_index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            source=base.source,
            page=base.page,
            category="paragraph",
            heading_path=list(base.heading_path),
            doc_id=base.doc_id,
            chunk_index=chunk_index,
        ),
    )


__all__ = [
    "MAX_CHUNK_CHARS",
    "SIMILARITY_THRESHOLD",
    "TARGET_CHUNK_CHARS",
    "chunk",
]


def configure_jieba(verbose: bool = False) -> None:
    if not verbose:
        jieba.setLogLevel(logger.level)
    jieba.initialize()
