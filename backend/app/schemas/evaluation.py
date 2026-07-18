"""Request/response schemas for the transparent Claude-vs-OpenAI comparison API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.answer import ALLOWED_PROVIDERS, ModelResultSchema
from app.schemas.search import MAX_TOP_K


class CompareRequest(BaseModel):
    """A direct Claude-vs-OpenAI comparison request for one question.

    Shaped like `AnswerRequest`, but `/compare` always evaluates exactly
    Anthropic and OpenAI against each other -- `providers`, if given at
    all, must name exactly those two (in either order).
    """

    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    document_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single previously uploaded document."
    )
    providers: Optional[List[str]] = Field(
        default=None,
        description=f"Must be exactly {list(ALLOWED_PROVIDERS)} (in either order) if given; defaults to both.",
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
    def _require_exactly_both_providers(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        # `set(value)` is only used to check *membership* (ignoring order/
        # duplicates) against the required pair; the normalized value
        # returned below is always `list(ALLOWED_PROVIDERS)` -- a tuple
        # literal with a fixed, deterministic ("anthropic", "openai")
        # order -- never derived from set iteration order. This also
        # collapses any duplicates (e.g. ["openai", "openai", "anthropic"])
        # down to the canonical two-element, deterministically-ordered list.
        if set(value) != set(ALLOWED_PROVIDERS):
            raise ValueError(
                f"Direct comparison requires exactly {list(ALLOWED_PROVIDERS)} (in either order); got {value}."
            )
        return list(ALLOWED_PROVIDERS)


class ProviderMetricsSchema(BaseModel):
    """Evaluation metrics computed for one provider's answer to one question.

    Every value here is derived only from data already present in that
    provider's own `ModelResultSchema` entry (citations, tokens, latency)
    plus the total evidence pool size -- nothing here re-queries or
    re-calls the provider.
    """

    provider: str
    model: str
    status: str = Field(..., description='Copied from the matching model_results entry: "success", "insufficient_evidence", or "error".')
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="input_tokens/1e6*input_price + output_tokens/1e6*output_price. Null if token usage or pricing is unavailable.",
    )
    valid_citation_count: int = Field(..., ge=0, description="Number of the model's citations that referenced real, supplied evidence.")
    citation_coverage: float = Field(
        ..., ge=0.0, le=1.0, description="Unique valid cited sources / available evidence sources, clamped [0,1]."
    )
    mean_citation_relevance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Mean relevance_score across cited sources. Null if there were no citations."
    )
    grounded_term_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Lexical heuristic only, NOT factual verification: proportion of the answer's meaningful "
        "(normalized, stop-word-filtered) terms that also appear in the cited excerpts. Null if the answer "
        "had no meaningful terms.",
    )
    answer_length: int = Field(..., ge=0, description="Word count of the answer text. Context only -- never used to judge quality.")
    evaluation_notes: List[str] = Field(
        default_factory=list, description="Short, factual notes on what could/could not be computed and why."
    )


class ComparisonSchema(BaseModel):
    """Cross-model comparison. Deliberately has no overall "winner" or accuracy score."""

    answer_agreement_score: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between the two providers' answer-text embeddings via the configured "
        "EmbeddingProvider. Semantic similarity only, NOT proof either answer is factually correct. Null unless "
        "both providers returned status='success'.",
    )
    latency_difference_ms: float = Field(
        ..., description="anthropic.latency_ms - openai.latency_ms. Positive means Anthropic was slower."
    )
    estimated_cost_difference_usd: Optional[float] = Field(
        default=None, description="anthropic.estimated_cost_usd - openai.estimated_cost_usd. Null if either is null."
    )
    comparison_status: str = Field(
        ...,
        description='One of "both_successful", "anthropic_succeeded_openai_did_not", '
        '"openai_succeeded_anthropic_did_not", "neither_succeeded".',
    )
    comparison_notes: List[str] = Field(
        default_factory=list,
        description="Factual, per-metric notes on which measurable differences exist (or are a tie within "
        "MODEL_COMPARISON_TIE_THRESHOLD). Never declares a universal winner or an accuracy verdict.",
    )


class CompareResponse(BaseModel):
    """Grounded answers from Anthropic and OpenAI, each independently evaluated and compared."""

    question: str = Field(
        ...,
        description="The question that was answered. If PII was detected in the submitted "
        "question, this is the masked version -- never the original.",
    )
    query_was_masked: bool = Field(..., description="Whether PII was detected and masked in the submitted question.")
    evidence_count: int = Field(..., ge=0, description="Number of evidence chunks actually supplied to the models.")
    model_results: List[ModelResultSchema]
    provider_metrics: List[ProviderMetricsSchema]
    comparison: ComparisonSchema
