"""Document ingestion API routes."""

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.schemas.document import DocumentUploadResponse
from app.services.ingestion.pdf_processor import process_pdf
from app.services.ingestion.text_chunker import chunk_document
from app.services.privacy.detectors import PIIDetector, get_pii_detector
from app.services.privacy.masking import PIIMaskingSummary, mask_extracted_document
from app.services.retrieval.vector_store import VectorStore, get_vector_store
from app.utils.uploads import read_upload_within_limit

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a single PDF policy document for ingestion",
)
async def upload_document(
    file: UploadFile = File(..., description="A single PDF file."),
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    pii_detector: PIIDetector = Depends(get_pii_detector),
) -> DocumentUploadResponse:
    data = await read_upload_within_limit(file, settings.max_upload_size_bytes)

    result = process_pdf(
        data=data,
        original_filename=file.filename or "",
        content_type=file.content_type,
    )

    # Mask PII page-by-page BEFORE the preview is (re)built, before
    # chunking, and before anything is embedded/indexed -- everything
    # downstream of this point (preview, chunks, the vector store, and
    # this response) only ever sees masked text. `result` is mutated in
    # place; the raw extracted text is never referenced again.
    if settings.pii_protection_enabled:
        pii_summary = mask_extracted_document(result, pii_detector)
    else:
        pii_summary = PIIMaskingSummary(detected=False, entity_count=0, categories=[])

    chunks = chunk_document(
        result,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_length=settings.min_chunk_length,
    )
    pages_with_text = sum(1 for page in result.pages if page.text.strip())

    # Upsert is keyed by each chunk's deterministic chunk_id (content-derived),
    # so re-uploading the same PDF re-indexes the same chunks in place instead
    # of duplicating them.
    indexed_chunk_count = vector_store.upsert_chunks(chunks)

    logger.info(
        "Document ingested: pages=%d characters=%d chunks=%d pages_with_text=%d indexed_chunks=%d "
        "pii_detected=%s pii_entity_count=%d",
        result.page_count,
        result.character_count,
        len(chunks),
        pages_with_text,
        indexed_chunk_count,
        pii_summary.detected,
        pii_summary.entity_count,
    )

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        page_count=result.page_count,
        character_count=result.character_count,
        status="processed",
        preview=result.preview,
        chunk_count=len(chunks),
        pages_with_text=pages_with_text,
        indexed_chunk_count=indexed_chunk_count,
        pii_detected=pii_summary.detected,
        pii_entity_count=pii_summary.entity_count,
        pii_categories=pii_summary.categories,
    )
