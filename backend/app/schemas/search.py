"""Request/response schemas for the semantic search API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

MAX_TOP_K = 50


class SearchRequest(BaseModel):
    """A semantic search request, optionally scoped to a single document."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language search query.")
    document_id: Optional[str] = Field(
        default=None, description="Restrict the search to a single previously uploaded document."
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description=f"Number of results to return (1-{MAX_TOP_K}). Defaults to the server's configured RETRIEVAL_TOP_K.",
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace-only.")
        return stripped


class SearchResultItem(BaseModel):
    """A single citation-ready search hit. Never carries the full chunk text."""

    chunk_id: str = Field(..., description="The matching chunk's deterministic ID.")
    document_id: str = Field(..., description="The parent document's content hash.")
    source_filename: str = Field(..., description="The sanitized filename the chunk came from.")
    page_number: int = Field(..., ge=1, description="The 1-based source PDF page -- the citation anchor.")
    excerpt: str = Field(..., description="Capped excerpt of the matching chunk text (not the full chunk).")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance, 1.0 = best match.")


class SearchResponse(BaseModel):
    """Ranked semantic search results."""

    query: str = Field(..., description="The (trimmed) query that was searched.")
    result_count: int = Field(..., ge=0, description="Number of results returned.")
    results: List[SearchResultItem] = Field(..., description="Results ordered by relevance, best match first.")
