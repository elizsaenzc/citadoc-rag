"""Split text into overlapping chunks that respect paragraph and sentence boundaries.

Strategy (recursive, like most production RAG pipelines):
1. Split by paragraphs; merge small paragraphs up to ``chunk_size``.
2. Paragraphs longer than ``chunk_size`` are split by sentences.
3. Sentences still too long are hard-split by characters.
4. Each new chunk starts with the tail of the previous one (``overlap``) so that
   an idea cut in two still appears whole in at least one chunk.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?¿¡])\s+")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(piece: str, size: int) -> list[str]:
    sentences = _SENTENCE_END.split(piece)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                out.append(current)
                current = ""
            out.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > size and current:
            out.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")

    text = _normalize(text)
    if not text:
        return []

    pieces: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces.extend(_split_long(paragraph, size) if len(paragraph) > size else [paragraph])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip()
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail}\n\n{piece}".strip() if tail else piece
            if len(current) > size:  # overlap made it too long: drop the overlap
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks
