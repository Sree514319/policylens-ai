"""Semantic search API routes."""

import logging
import time

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.search import MAX_TOP_K, SearchRequest, SearchResponse, SearchResultItem
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
) -> SearchResponse:
    # request.top_k is already schema-bounded to [1, MAX_TOP_K] by Pydantic;
    # the server-configured default is clamped here too, in case
    # RETRIEVAL_TOP_K was misconfigured outside that range in .env.
    top_k = request.top_k or max(1, min(settings.retrieval_top_k, MAX_TOP_K))
    started = time.perf_counter()

    # The query text itself is never logged -- only IDs, counts, and timing.
    results = vector_store.search(
        query=request.query,
        top_k=top_k,
        document_id=request.document_id,
        min_relevance_score=settings.min_relevance_score,
    )

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Search completed: document_id=%s top_k=%d result_count=%d duration_ms=%.1f status=ok",
        request.document_id or "<all>",
        top_k,
        len(results),
        duration_ms,
    )

    return SearchResponse(
        query=request.query,
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
