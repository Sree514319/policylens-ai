"""Document ingestion API routes."""

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.schemas.document import DocumentUploadResponse
from app.services.ingestion.pdf_processor import process_pdf
from app.services.ingestion.text_chunker import chunk_document
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
) -> DocumentUploadResponse:
    data = await read_upload_within_limit(file, settings.max_upload_size_bytes)

    result = process_pdf(
        data=data,
        original_filename=file.filename or "",
        content_type=file.content_type,
    )

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
        "Document ingested: pages=%d characters=%d chunks=%d pages_with_text=%d indexed_chunks=%d",
        result.page_count,
        result.character_count,
        len(chunks),
        pages_with_text,
        indexed_chunk_count,
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
    )
