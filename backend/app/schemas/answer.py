"""Request/response schemas for the multi-model grounded-answer API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.search import MAX_TOP_K

ALLOWED_PROVIDERS = ("anthropic", "openai")


class AnswerRequest(BaseModel):
    """A grounded-answer request, optionally scoped to a document and/or a provider subset."""

    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    document_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single previously uploaded document."
    )
    providers: Optional[List[str]] = Field(
        default=None,
        description=f"Which providers to query: any of {list(ALLOWED_PROVIDERS)}. Defaults to both.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description=f"Number of evidence chunks to retrieve (1-{MAX_TOP_K}). Defaults to the server's configured RETRIEVAL_TOP_K.",
    )

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace-only.")
        return stripped

    @field_validator("providers")
    @classmethod
    def _validate_providers(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if not value:
            raise ValueError("providers must not be an empty list.")
        unknown = sorted(set(value) - set(ALLOWED_PROVIDERS))
        if unknown:
            raise ValueError(f"Unknown provider(s): {unknown}. Allowed: {list(ALLOWED_PROVIDERS)}.")
        return list(dict.fromkeys(value))  # dedupe, preserve order


class CitationSchema(BaseModel):
    """A single validated citation. Every field is our own stored evidence metadata."""

    source_label: str = Field(..., description='The evidence label this citation refers to, e.g. "S1".')
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int = Field(..., ge=1)
    excerpt: str = Field(..., description="Capped excerpt of the cited chunk text (not the full chunk).")
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ModelResultSchema(BaseModel):
    """One provider's grounded-answer attempt."""

    provider: str
    model: str
    status: str = Field(..., description='"success", "insufficient_evidence", or "error".')
    answer: str
    citations: List[CitationSchema] = Field(default_factory=list)
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: Optional[int] = Field(default=None, description="Prompt/input token count, if the provider reported one.")
    output_tokens: Optional[int] = Field(
        default=None, description="Completion/output token count, if the provider reported one."
    )
    error: Optional[str] = Field(default=None, description="Safe error message, present only when status is 'error'.")


class AnswerResponse(BaseModel):
    """Grounded answers from every requested model, evaluated independently."""

    question: str = Field(
        ...,
        description="The question that was answered. If PII was detected in the submitted "
        "question, this is the masked version -- never the original.",
    )
    query_was_masked: bool = Field(..., description="Whether PII was detected and masked in the submitted question.")
    evidence_count: int = Field(..., ge=0, description="Number of evidence chunks actually supplied to the models.")
    model_results: List[ModelResultSchema]
