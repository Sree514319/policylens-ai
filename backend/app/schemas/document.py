"""Request/response schemas for the document ingestion API."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Metadata returned after a PDF has been validated and processed.

    Deliberately excludes extracted page text: only a short preview is
    returned so the full document content is never exposed in an API
    response.
    """

    document_id: str = Field(
        ..., description="SHA-256 hex digest of the document content; stable across re-uploads."
    )
    filename: str = Field(..., description="Sanitized version of the uploaded filename.")
    page_count: int = Field(..., ge=0, description="Number of pages in the PDF.")
    character_count: int = Field(..., ge=0, description="Total extracted character count across all pages.")
    status: str = Field(..., description="Processing status, e.g. 'processed'.")
    preview: str = Field(..., description="Short, truncated preview of the extracted text (not the full document).")
    chunk_count: int = Field(..., ge=0, description="Number of citation-ready text chunks produced.")
    pages_with_text: int = Field(..., ge=0, description="Number of pages that contained extractable text.")
    indexed_chunk_count: int = Field(
        ..., ge=0, description="Number of chunks successfully upserted into the vector store."
    )


class ErrorResponse(BaseModel):
    """Shape of error responses returned to API clients."""

    detail: str = Field(..., description="Client-safe error message. Never contains stack traces or file paths.")
