"""Small helpers shared by more than one `app/api/v1` route module.

Kept deliberately tiny: this is not a place for business logic (that
belongs in `app/services/*`), just response-shape adapters that would
otherwise be duplicated verbatim across routes that both build on the
same RAG orchestration (`/answers` and `/compare`).
"""

from app.schemas.answer import CitationSchema, ModelResultSchema
from app.services.llm.rag import ModelAnswer


def to_model_result_schema(model_answer: ModelAnswer) -> ModelResultSchema:
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
