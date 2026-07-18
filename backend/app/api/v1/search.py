"""Semantic search API routes."""

import logging
import time

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.search import MAX_TOP_K, SearchRequest, SearchResponse, SearchResultItem
from app.services.privacy.detectors import PIIDetector, get_pii_detector
from app.services.privacy.masking import mask_query
from app.services.retrieval.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search across indexed document chunks",
)
async def search_documents(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    pii_detector: PIIDetector = Depends(get_pii_detector),
) -> SearchResponse:
    # request.top_k is already schema-bounded to [1, MAX_TOP_K] by Pydantic;
    # the server-configured default is clamped here too, in case
    # RETRIEVAL_TOP_K was misconfigured outside that range in .env.
    top_k = request.top_k or max(1, min(settings.retrieval_top_k, MAX_TOP_K))
    started = time.perf_counter()

    # The query is masked BEFORE it is embedded/searched, so PII pasted
    # into a query is never sent to the embedding model or stored in any
    # search history -- only the masked version is ever used or returned.
    if settings.pii_protection_enabled:
        query, query_was_masked = mask_query(request.query, pii_detector)
    else:
        query, query_was_masked = request.query, False

    # The query text itself is never logged -- only IDs, counts, and timing.
    results = vector_store.search(
        query=query,
        top_k=top_k,
        document_id=request.document_id,
        min_relevance_score=settings.min_relevance_score,
    )

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Search completed: document_id=%s top_k=%d result_count=%d duration_ms=%.1f "
        "query_was_masked=%s status=ok",
        request.document_id or "<all>",
        top_k,
        len(results),
        duration_ms,
        query_was_masked,
    )

    return SearchResponse(
        query=query,
        query_was_masked=query_was_masked,
        result_count=len(results),
        results=[
            SearchResultItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                source_filename=result.source_filename,
                page_number=result.page_number,
                excerpt=result.excerpt,
                relevance_score=result.relevance_score,
            )
            for result in results
        ],
    )
