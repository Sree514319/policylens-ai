"""Deterministic, page-aware text chunking for citation-ready retrieval.

Takes the page-level output of ``pdf_processor.process_pdf`` and splits each
page's text independently into overlapping, word-boundary-aware chunks.
Chunking never crosses a page boundary, so every chunk's ``page_number``
stays an accurate citation back to the source PDF page.

No third-party chunking library (e.g. LangChain) is used -- this is plain,
dependency-free string slicing so behavior is fully deterministic and easy
to reason about. Extracted page/chunk text is held in memory only long
enough to build the returned ``Chunk`` objects; callers must not log it.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List

from app.core.exceptions import InvalidChunkConfigurationError
from app.services.ingestion.pdf_processor import ExtractedDocument, PageContent

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class Chunk:
    """A single citation-addressable slice of a document's extracted text."""

    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    source_filename: str
    text: str
    character_count: int
    start_character: int
    end_character: int
    content_hash: str


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space.

    Only whitespace characters are touched -- digits, currency symbols,
    decimal points, and percentages are never modified, so financial
    meaning (e.g. "$1,000.00", "12.5%") is preserved exactly.
    """

    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int, min_chunk_length: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkConfigurationError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise InvalidChunkConfigurationError("chunk_overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkConfigurationError("chunk_overlap must be smaller than chunk_size.")
    if min_chunk_length < 0:
        raise InvalidChunkConfigurationError("min_chunk_length must not be negative.")
    if min_chunk_length > chunk_size:
        raise InvalidChunkConfigurationError("min_chunk_length must not exceed chunk_size.")


def _snap_to_word_boundary(text: str, start: int, end: int) -> int:
    """Pull a candidate cut point back to the nearest preceding space.

    If the text already breaks cleanly at ``end`` (end of string, or a
    space), no adjustment is made. Otherwise this looks backward for a
    space within the current window; if none exists (a single token longer
    than chunk_size, e.g. a long account number), it falls back to a hard
    cut so chunking always makes forward progress.

    The snap is only accepted if it keeps at least half of the window.
    Without this bound, a long word starting shortly after ``start`` (but
    with an earlier, unrelated space still in range) would repeatedly snap
    back to that same distant space on every iteration -- since the next
    window still contains it -- producing a cascade of near-duplicate,
    near-empty chunks instead of making real forward progress. Falling
    back to a hard mid-word cut in that case is the lesser evil.
    """

    if end >= len(text) or text[end] == " ":
        return end

    boundary = text.rfind(" ", start, end)
    if boundary <= start or (end - boundary) > (end - start) // 2:
        return end

    return boundary


def _split_text(text: str, chunk_size: int, chunk_overlap: int, min_chunk_length: int) -> List[tuple]:
    """Return a list of (start, end) character spans covering ``text``."""

    length = len(text)
    if length == 0:
        return []
    if length <= chunk_size:
        return [(0, length)]

    spans: List[tuple] = []
    start = 0

    while start < length:
        raw_end = min(start + chunk_size, length)
        end = _snap_to_word_boundary(text, start, raw_end)
        spans.append((start, end))

        if end >= length:
            break

        next_start = end - chunk_overlap
        # Skip the space we just cut on, and guarantee forward progress
        # even in pathological cases (e.g. overlap larger than the gap
        # produced by word-boundary snapping).
        if next_start <= spans[-1][0]:
            next_start = spans[-1][0] + 1
        while next_start < length and text[next_start] == " ":
            next_start += 1
        start = next_start

    # A short trailing remainder (an artifact of the sliding window, not
    # a deliberate short chunk) is merged into the previous chunk rather
    # than emitted as its own tiny, low-context chunk.
    if len(spans) > 1:
        last_start, last_end = spans[-1]
        if (last_end - last_start) < min_chunk_length:
            prev_start, _ = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()

    return spans


def _build_chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    payload = f"{document_id}:{page_number}:{chunk_index}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_pages(
    pages: List[PageContent],
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int,
) -> List[Chunk]:
    """Chunk each page's text independently, in page order.

    Raises `InvalidChunkConfigurationError` if the size/overlap/minimum
    settings are inconsistent. Deterministic: identical pages and settings
    always produce identical chunks (same text, spans, and chunk_ids).
    """

    _validate_chunk_configuration(chunk_size, chunk_overlap, min_chunk_length)

    chunks: List[Chunk] = []
    chunk_index = 0

    for page in pages:
        normalized = _normalize_whitespace(page.text)
        for start, end in _split_text(normalized, chunk_size, chunk_overlap, min_chunk_length):
            chunk_text = normalized[start:end]
            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    chunk_id=_build_chunk_id(document_id, page.page_number, chunk_index, chunk_text),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    source_filename=page.source_filename,
                    text=chunk_text,
                    character_count=len(chunk_text),
                    start_character=start,
                    end_character=end,
                    content_hash=hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                )
            )
            chunk_index += 1

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int,
) -> List[Chunk]:
    """Convenience wrapper: chunk every page of an already-extracted PDF."""

    return chunk_pages(document.pages, document.document_id, chunk_size, chunk_overlap, min_chunk_length)
