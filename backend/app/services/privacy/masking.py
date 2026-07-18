"""PII masking orchestration: applying `PIIDetector` results to real data.

This is the only module that mutates `ExtractedDocument`/`PageContent` for
privacy purposes; `pdf_processor.py` and `text_chunker.py` stay entirely
privacy-agnostic. Masking is deterministic and irreversible: matched spans
are replaced with fixed placeholder strings (see `detectors.py`), and the
original values are never stored, logged, or returned anywhere -- once
`apply_masking` returns, the original substrings are simply gone.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from app.core.exceptions import PIIConfigurationError, PIIProcessingError
from app.services.ingestion.pdf_processor import ExtractedDocument, PageContent, build_preview
from app.services.privacy.detectors import PIIDetector, PIIEntity

logger = logging.getLogger(__name__)


@dataclass
class PIIMaskingSummary:
    """Category names and counts only -- never matched values or positions."""

    detected: bool
    entity_count: int
    categories: List[str]


def validate_pii_configuration(pii_protection_enabled: bool, pii_redaction_version: str) -> None:
    """Fail fast on an inconsistent PII configuration.

    `pii_mode` itself is validated by Pydantic (a `Literal["mask"]` field
    on `Settings`); this covers the one additional invariant Pydantic
    can't express as a simple type: a non-empty redaction version is
    required whenever protection is enabled, since it's what the vector
    store uses to detect/refuse stale or unmasked data (see
    `VectorStore`'s privacy-version check).
    """

    if pii_protection_enabled and not (pii_redaction_version or "").strip():
        raise PIIConfigurationError("PII_REDACTION_VERSION must be a non-empty value when PII protection is enabled.")


def apply_masking(text: str, entities: List[PIIEntity]) -> str:
    """Replace each entity's span with its placeholder. `entities` need not be sorted."""

    if not entities:
        return text

    pieces = []
    cursor = 0
    for entity in sorted(entities, key=lambda e: e.start):
        pieces.append(text[cursor : entity.start])
        pieces.append(entity.placeholder)
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _detect(detector: PIIDetector, text: str, context: str) -> List[PIIEntity]:
    try:
        return detector.detect(text)
    except Exception as exc:
        logger.error("PII detection failed while processing %s.", context)
        raise PIIProcessingError(f"PII detection failed while processing {context}.") from exc


def mask_query(text: str, detector: PIIDetector) -> Tuple[str, bool]:
    """Mask a search query or RAG question. Returns (masked_text, was_masked)."""

    entities = _detect(detector, text, "a query")
    return apply_masking(text, entities), len(entities) > 0


def summarize_entities(categories: List[str]) -> PIIMaskingSummary:
    return PIIMaskingSummary(
        detected=len(categories) > 0,
        entity_count=len(categories),
        categories=sorted(set(categories)),
    )


def mask_extracted_document(document: ExtractedDocument, detector: PIIDetector) -> PIIMaskingSummary:
    """Mask `document.filename` and every page of `document` in place.

    A client-supplied filename (e.g. "jane.doe@example.com-statement.pdf")
    is itself untrusted, PII-bearing text -- it is stored as chunk/vector
    metadata, returned in API responses, shown in search/RAG citations,
    and embedded directly into the prompt text sent to external LLM
    providers (see `rag._build_user_prompt`'s "(source: ..., page N)"
    header). It is masked here, once, using the same detector as page
    text, and the masked filename is what every downstream `PageContent`
    carries as `source_filename` -- so a raw filename never reaches
    chunking, indexing, search results, citations, or a provider prompt.

    Also replaces `document.pages` with masked `PageContent` instances,
    and recomputes `document.character_count`/`document.preview` from
    that masked text -- so the raw extracted text this function was given
    is never referenced again by any caller (chunking, indexing, or the
    upload response all run after this returns).

    Limitation: masking runs independently per page (and on the filename
    as one unit), so a sensitive value split across a page boundary --
    e.g. an SSN with its first digits at the very end of page 1 and the
    rest at the start of page 2 -- is not detected as a single entity and
    will not be masked. See the README's PII "Limitations" section.
    """

    filename_entities = _detect(detector, document.filename, "an uploaded document's filename")
    masked_filename = apply_masking(document.filename, filename_entities)

    masked_pages: List[PageContent] = []
    all_categories: List[str] = [entity.category for entity in filename_entities]

    for page in document.pages:
        entities = _detect(detector, page.text, "an uploaded document page")
        masked_text = apply_masking(page.text, entities)
        masked_pages.append(
            PageContent(
                page_number=page.page_number,
                source_filename=masked_filename,
                text=masked_text,
                character_count=len(masked_text),
            )
        )
        all_categories.extend(entity.category for entity in entities)

    document.filename = masked_filename
    document.pages = masked_pages
    document.character_count = sum(page.character_count for page in masked_pages)
    document.preview = build_preview(masked_pages)

    return summarize_entities(all_categories)
