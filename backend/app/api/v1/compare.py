"""Transparent Claude-vs-OpenAI comparison API route.

Reuses the exact same RAG orchestration as `/answers` (`answer_question`)
-- exactly once, for both providers together -- then evaluates and
compares the two results. This route never makes a second/duplicate
provider call and never re-embeds or re-retrieves anything.
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends

from app.api.v1._shared import to_model_result_schema
from app.core.config import Settings, get_settings
from app.schemas.answer import ALLOWED_PROVIDERS
from app.schemas.evaluation import CompareRequest, CompareResponse, ComparisonSchema, ProviderMetricsSchema
from app.schemas.search import MAX_TOP_K
from app.services.evaluation.metrics import compare_providers, evaluate_providers
from app.services.llm.providers import LLMProvider, get_llm_provider_registry
from app.services.llm.rag import answer_question
from app.services.privacy.detectors import PIIDetector, get_pii_detector
from app.services.privacy.masking import mask_query
from app.services.retrieval.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Transparent, metric-by-metric Claude-vs-OpenAI comparison for one question",
)
async def compare_models(
    request: CompareRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_providers: Dict[str, LLMProvider] = Depends(get_llm_provider_registry),
    pii_detector: PIIDetector = Depends(get_pii_detector),
) -> CompareResponse:
    provider_names = request.providers or list(ALLOWED_PROVIDERS)
    top_k = request.top_k or max(1, min(settings.retrieval_top_k, MAX_TOP_K))

    # Masked BEFORE retrieval and before it's placed in any provider prompt
    # -- identical privacy handling to /answers. See app.services.privacy.
    if settings.pii_protection_enabled:
        question, query_was_masked = mask_query(request.question, pii_detector)
    else:
        question, query_was_masked = request.question, False

    # Exactly one RAG orchestration call, covering both providers -- no
    # duplicate provider calls, no second retrieval.
    evidence_count, model_answers = await answer_question(
        question=question,
        document_id=request.document_id,
        top_k=top_k,
        vector_store=vector_store,
        providers=llm_providers,
        provider_names=provider_names,
        min_relevance_score=settings.min_relevance_score,
        max_context_characters=settings.max_rag_context_characters,
        allow_external_calls=settings.allow_external_llm_calls,
    )

    metrics_by_provider = evaluate_providers(model_answers, evidence_count, settings)
    answers_by_provider = {answer.provider: answer for answer in model_answers}

    comparison = compare_providers(
        anthropic_answer=answers_by_provider["anthropic"],
        openai_answer=answers_by_provider["openai"],
        anthropic_metrics=metrics_by_provider["anthropic"],
        openai_metrics=metrics_by_provider["openai"],
        embedding_provider=vector_store.embedding_provider,
        tie_threshold=settings.model_comparison_tie_threshold,
    )

    # Never log question/answer/evidence/citation text -- only IDs, counts,
    # statuses, and the comparison outcome (all already client-safe).
    logger.info(
        "Compare request completed: document_id=%s evidence_count=%d query_was_masked=%s "
        "comparison_status=%s results=%s",
        request.document_id or "<all>",
        evidence_count,
        query_was_masked,
        comparison.comparison_status,
        ",".join(f"{answer.provider}:{answer.status}" for answer in model_answers),
    )

    return CompareResponse(
        question=question,
        query_was_masked=query_was_masked,
        evidence_count=evidence_count,
        model_results=[to_model_result_schema(answer) for answer in model_answers],
        provider_metrics=[
            ProviderMetricsSchema(
                provider=metrics.provider,
                model=metrics.model,
                status=metrics.status,
                latency_ms=metrics.latency_ms,
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                estimated_cost_usd=metrics.estimated_cost_usd,
                valid_citation_count=metrics.valid_citation_count,
                citation_coverage=metrics.citation_coverage,
                mean_citation_relevance=metrics.mean_citation_relevance,
                grounded_term_ratio=metrics.grounded_term_ratio,
                answer_length=metrics.answer_length,
                evaluation_notes=metrics.evaluation_notes,
            )
            for metrics in (metrics_by_provider["anthropic"], metrics_by_provider["openai"])
        ],
        comparison=ComparisonSchema(
            answer_agreement_score=comparison.answer_agreement_score,
            latency_difference_ms=comparison.latency_difference_ms,
            estimated_cost_difference_usd=comparison.estimated_cost_difference_usd,
            comparison_status=comparison.comparison_status,
            comparison_notes=comparison.comparison_notes,
        ),
    )
