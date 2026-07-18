"""Multi-model grounded RAG answer API routes."""

import logging
from typing import Dict

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.answer import AnswerRequest, AnswerResponse, CitationSchema, ModelResultSchema
from app.schemas.search import MAX_TOP_K
from app.services.llm.providers import LLMProvider, get_llm_provider_registry
from app.services.llm.rag import ModelAnswer, answer_question
from app.services.retrieval.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_model_result(model_answer: ModelAnswer) -> ModelResultSchema:
    return ModelResultSchema(
        provider=model_answer.provider,
        model=model_answer.model,
        status=model_answer.status,
        answer=model_answer.answer,
        citations=[
            CitationSchema(
                source_label=citation.source_label,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                source_filename=citation.source_filename,
                page_number=citation.page_number,
                excerpt=citation.excerpt,
                relevance_score=citation.relevance_score,
            )
            for citation in model_answer.citations
        ],
        latency_ms=model_answer.latency_ms,
        input_tokens=model_answer.input_tokens,
        output_tokens=model_answer.output_tokens,
        error=model_answer.error,
    )


@router.post(
    "/answers",
    response_model=AnswerResponse,
    summary="Grounded, multi-model RAG answer",
)
async def get_answers(
    request: AnswerRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_providers: Dict[str, LLMProvider] = Depends(get_llm_provider_registry),
) -> AnswerResponse:
    provider_names = request.providers or ["anthropic", "openai"]
    top_k = request.top_k or max(1, min(settings.retrieval_top_k, MAX_TOP_K))

    # The question and evidence text are never logged -- only IDs, counts,
    # provider names/statuses, and timing.
    evidence_count, model_answers = await answer_question(
        question=request.question,
        document_id=request.document_id,
        top_k=top_k,
        vector_store=vector_store,
        providers=llm_providers,
        provider_names=provider_names,
        min_relevance_score=settings.min_relevance_score,
        max_context_characters=settings.max_rag_context_characters,
        allow_external_calls=settings.allow_external_llm_calls,
    )

    logger.info(
        "Answer request completed: document_id=%s providers=%s evidence_count=%d results=%s",
        request.document_id or "<all>",
        ",".join(provider_names),
        evidence_count,
        ",".join(f"{answer.provider}:{answer.status}" for answer in model_answers),
    )

    return AnswerResponse(
        question=request.question,
        evidence_count=evidence_count,
        model_results=[_to_model_result(answer) for answer in model_answers],
    )
