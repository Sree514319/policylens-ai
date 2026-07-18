"""Transparent, deterministic evaluation metrics for grounded RAG answers.

Two kinds of metric, computed only from data the API already produced (no
re-querying a provider, no second retrieval, no external "judge" model):

- `ProviderMetrics`: per-provider metrics derived from one `ModelAnswer`
  (its own citations, tokens, latency) plus the total evidence pool size
  for that request.
- `ComparisonResult`: a Claude-vs-OpenAI comparison built from exactly two
  `ProviderMetrics`. It deliberately never produces a single "winner" or
  an accuracy score -- see `compare_providers`.

Two metrics here are explicitly *heuristics*, not correctness signals, and
every place they appear (schema descriptions, evaluation/comparison notes)
says so:

- `grounded_term_ratio` is lexical overlap (does the answer's vocabulary
  appear in what it cited?), not fact-checking.
- `answer_agreement_score` is embedding cosine similarity (do the two
  answers say similar things?), not proof either one is correct.

Privacy: every function here operates on already-masked question/answer/
citation-excerpt text (masking happens upstream, before retrieval and
before any provider call -- see `app.services.privacy.masking`). Nothing
here logs question, answer, or excerpt text; `evaluation_notes`/
`comparison_notes` are short, fixed-vocabulary, template-based strings
that never interpolate raw answer/citation content, only numbers,
provider names, and status strings that are already safe to return to a
client (see `app.services.llm.providers`/`rag`).
"""

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.config import Settings
from app.services.llm.rag import Citation, ModelAnswer
from app.services.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

_STATUS_SUCCESS = "success"

# A small, fixed, deterministic English stop-word list -- no NLP dependency.
# Intentionally conservative (common function words only); it does not aim
# to be linguistically exhaustive, only to keep grounded_term_ratio from
# being dominated by words that carry no topical meaning.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
        "were", "be", "been", "being", "of", "to", "in", "on", "at", "for",
        "with", "by", "from", "as", "it", "its", "and", "or", "but", "if",
        "then", "than", "so", "such", "not", "no", "nor", "do", "does", "did",
        "doing", "have", "has", "had", "having", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "i", "you", "he",
        "she", "we", "they", "them", "his", "her", "their", "our", "your",
        "my", "what", "which", "who", "whom", "where", "when", "why", "how",
        "all", "each", "few", "more", "most", "other", "some", "any", "only",
        "own", "same", "too", "very", "just", "about", "into", "through",
        "during", "before", "after", "above", "below", "up", "down", "out",
        "off", "over", "under", "again", "further", "once", "here", "there",
        "also",
    }
)

# Numbers are tokenized as a whole (not split into digit fragments) so two
# *different* financial values never spuriously "overlap" just because they
# share a substring -- e.g. without this, "$1,000.00" and "$5,000.00" would
# both fragment down to a shared "000"/"00" token and register as grounded
# overlap despite being different amounts. `$`/`,` are stripped (so
# "$1,000.00" and "1000.00" normalize identically); sign and "%" are kept
# (so "-2.5%" and "2.5%" stay distinct, and "4.95%" still reads as one
# meaningful token instead of the sub-3-character fragments "4"/"95").
_NUMBER_TOKEN_PATTERN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")
_WORD_TOKEN_PATTERN = re.compile(r"[a-z]+")
_COST_DECIMALS = 6  # micro-dollar precision -- enough for sub-cent LLM costs

_GROUNDED_TERM_RATIO_DISCLAIMER = (
    "grounded_term_ratio is a lexical overlap heuristic (normalized answer "
    "terms found in cited excerpts) and does NOT verify factual correctness."
)
_ANSWER_AGREEMENT_DISCLAIMER = (
    "answer_agreement_score is embedding cosine similarity (semantic "
    "closeness) and does NOT prove either answer is factually correct."
)
_CITATION_COUNT_DISCLAIMER = (
    "Citation counts are reported for transparency only, not as a quality "
    "signal -- a model can be fully grounded while citing fewer sources."
)

_PRICE_SETTINGS_BY_PROVIDER: Dict[str, tuple] = {
    "anthropic": ("anthropic_input_cost_per_million", "anthropic_output_cost_per_million"),
    "openai": ("openai_input_cost_per_million", "openai_output_cost_per_million"),
}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp `value` into `[low, high]`. NaN is treated as undefined, not a
    number to be forced into range -- silently mapping it to `high` (which
    is what Python's plain `max(low, min(high, value))` does, since NaN
    comparisons are always False) would make a broken/undefined input look
    like a confident, valid score. Callers with an Optional[float] return
    type should check `math.isnan` themselves and return `None` instead of
    calling this at all; this is a last-resort guard for callers that don't.
    """

    if math.isnan(value):
        return low
    return max(low, min(high, value))


def _strip_diacritics(text: str) -> str:
    """Best-effort accent/diacritic removal for Latin-script text (e.g.
    "café" -> "cafe") using only the stdlib -- no new dependency. This is
    a real but narrow improvement: it does nothing for non-Latin scripts
    (Chinese, Arabic, Cyrillic, etc.), which the ASCII-only token pattern
    below still drops entirely -- a documented, now-narrower limitation.
    """

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _tokenize(text: str) -> List[str]:
    lowered = _strip_diacritics(text).lower()
    number_tokens = [match.group().replace(",", "").replace("$", "") for match in _NUMBER_TOKEN_PATTERN.finditer(lowered)]
    word_tokens = _WORD_TOKEN_PATTERN.findall(lowered)
    return number_tokens + word_tokens


def _meaningful_terms(text: str, min_length: int) -> set:
    return {token for token in _tokenize(text) if token not in _STOP_WORDS and len(token) >= min_length}


@dataclass(frozen=True)
class ProviderMetrics:
    """Evaluation metrics for one provider's answer to one question."""

    provider: str
    model: str
    status: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    valid_citation_count: int
    citation_coverage: float
    mean_citation_relevance: Optional[float]
    grounded_term_ratio: Optional[float]
    answer_length: int
    evaluation_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonResult:
    """Cross-model comparison. Never a single "winner" or accuracy score."""

    answer_agreement_score: Optional[float]
    latency_difference_ms: float
    estimated_cost_difference_usd: Optional[float]
    comparison_status: str
    comparison_notes: List[str] = field(default_factory=list)


# --- Per-provider metric formulas -----------------------------------------------


def _estimate_cost_usd(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    input_price_per_million: Optional[float],
    output_price_per_million: Optional[float],
) -> Optional[float]:
    """input_tokens/1e6*input_price + output_tokens/1e6*output_price.

    Null whenever token usage OR pricing is unavailable -- a partial
    calculation (e.g. only input tokens known) would understate the real
    cost, so this never guesses.
    """

    if input_tokens is None or output_tokens is None:
        return None
    if input_price_per_million is None or output_price_per_million is None:
        return None
    cost = (input_tokens / 1_000_000) * input_price_per_million + (output_tokens / 1_000_000) * output_price_per_million
    # Rounded to a consistent precision (micro-dollars) so floating-point
    # noise (e.g. 0.00045000000000000004) never leaks into API responses
    # or benchmark output, and so a difference computed from two already-
    # rounded costs is itself stable and reproducible.
    return round(max(0.0, cost), _COST_DECIMALS)


def _citation_coverage(citations: List[Citation], evidence_count: int) -> float:
    """Unique valid cited sources / available evidence sources, clamped [0,1].

    0.0 when there was no evidence at all (nothing to cover), never a
    division error.
    """

    if evidence_count <= 0:
        return 0.0
    unique_sources = {citation.chunk_id for citation in citations}
    return _clamp(len(unique_sources) / evidence_count, 0.0, 1.0)


def _mean_citation_relevance(citations: List[Citation]) -> Optional[float]:
    """Arithmetic mean of relevance_score across cited sources.

    Null if there are no citations, or -- defensively -- if any relevance
    score is NaN (a malformed/undefined score should never silently
    average out to a confident-looking number).
    """

    if not citations:
        return None
    mean = sum(citation.relevance_score for citation in citations) / len(citations)
    if math.isnan(mean):
        return None
    return _clamp(mean, 0.0, 1.0)


def _grounded_term_ratio(answer_text: str, citations: List[Citation], min_length: int) -> Optional[float]:
    """Proportion of the answer's unique meaningful terms also present in its
    cited excerpts, after lowercasing/punctuation-stripping and stop-word
    removal. Null when the answer has no meaningful terms (undefined
    proportion) -- 0.0 when it has terms but zero citations to check against.
    """

    meaningful_terms = _meaningful_terms(answer_text, min_length)
    if not meaningful_terms:
        return None
    evidence_text = " ".join(citation.excerpt for citation in citations)
    evidence_terms = set(_tokenize(evidence_text))
    grounded = meaningful_terms & evidence_terms
    return _clamp(len(grounded) / len(meaningful_terms), 0.0, 1.0)


def evaluate_provider(
    answer: ModelAnswer,
    evidence_count: int,
    input_price_per_million: Optional[float],
    output_price_per_million: Optional[float],
    cost_tracking_enabled: bool,
    grounded_term_min_length: int,
) -> ProviderMetrics:
    """Evaluate a single provider's `ModelAnswer`. Never raises: any metric
    that cannot be computed is `None`, with a note explaining why."""

    notes: List[str] = []
    citations = answer.citations

    coverage = _citation_coverage(citations, evidence_count)

    mean_relevance = _mean_citation_relevance(citations)
    if mean_relevance is None:
        notes.append("No valid citations; mean_citation_relevance not computed.")

    grounded_ratio: Optional[float] = None
    if answer.status == _STATUS_SUCCESS:
        grounded_ratio = _grounded_term_ratio(answer.answer, citations, grounded_term_min_length)
        if grounded_ratio is None:
            notes.append("Answer had no meaningful terms after normalization; grounded_term_ratio not computed.")
        else:
            notes.append(_GROUNDED_TERM_RATIO_DISCLAIMER)
    else:
        notes.append(f"status='{answer.status}': grounded_term_ratio is only computed for successful answers.")

    if not cost_tracking_enabled:
        estimated_cost: Optional[float] = None
        notes.append("Cost tracking is disabled (ENABLE_COST_TRACKING=false); estimated_cost_usd not computed.")
    else:
        estimated_cost = _estimate_cost_usd(
            answer.input_tokens, answer.output_tokens, input_price_per_million, output_price_per_million
        )
        if estimated_cost is None:
            if answer.input_tokens is None or answer.output_tokens is None:
                notes.append("Provider did not report token usage; estimated_cost_usd not computed.")
            else:
                notes.append("Per-token pricing is not configured for this provider; estimated_cost_usd not computed.")

    return ProviderMetrics(
        provider=answer.provider,
        model=answer.model,
        status=answer.status,
        latency_ms=answer.latency_ms,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        estimated_cost_usd=estimated_cost,
        valid_citation_count=len(citations),
        citation_coverage=coverage,
        mean_citation_relevance=mean_relevance,
        grounded_term_ratio=grounded_ratio,
        answer_length=len(answer.answer.split()),
        evaluation_notes=notes,
    )


def evaluate_providers(model_answers: List[ModelAnswer], evidence_count: int, settings: Settings) -> Dict[str, ProviderMetrics]:
    """Evaluate every provider's answer, keyed by provider name."""

    results: Dict[str, ProviderMetrics] = {}
    for answer in model_answers:
        input_attr, output_attr = _PRICE_SETTINGS_BY_PROVIDER.get(answer.provider, (None, None))
        input_price = getattr(settings, input_attr) if input_attr else None
        output_price = getattr(settings, output_attr) if output_attr else None
        results[answer.provider] = evaluate_provider(
            answer=answer,
            evidence_count=evidence_count,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            cost_tracking_enabled=settings.enable_cost_tracking,
            grounded_term_min_length=settings.grounded_term_min_length,
        )
    return results


# --- Cross-model comparison -------------------------------------------------------


def _is_finite_vector(vector: List[float]) -> bool:
    return all(math.isfinite(value) for value in vector)


def _cosine_similarity(vector_a: List[float], vector_b: List[float]) -> Optional[float]:
    """Cosine similarity, or `None` if the vectors can't yield one: empty,
    mismatched dimension, or containing a NaN/infinite component (a
    malformed embedding must never silently produce a fake in-range
    score)."""

    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return None
    if not _is_finite_vector(vector_a) or not _is_finite_vector(vector_b):
        return None

    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None

    result = dot / (norm_a * norm_b)
    if math.isnan(result):
        return None
    return _clamp(result, -1.0, 1.0)


def _is_tie(anthropic_value: float, openai_value: float, threshold: float) -> bool:
    """Two values are a tie when their absolute difference is within
    `threshold` (a fraction, e.g. 0.05 = 5%) of the larger magnitude.
    Handles the both-zero case (a real tie) without dividing by zero.
    """

    scale = max(abs(anthropic_value), abs(openai_value), 1e-9)
    return abs(anthropic_value - openai_value) <= threshold * scale


def _describe_metric_diff(
    label: str,
    unit: str,
    anthropic_value: Optional[float],
    openai_value: Optional[float],
    threshold: float,
    higher_is_better: bool,
    decimals: int = 2,
) -> str:
    """One factual, non-judgmental sentence comparing a single metric.

    Never says "won" -- says which provider had the (higher|lower) value,
    or reports a tie/unavailability. `higher_is_better` only picks the
    wording ("X had the higher/lower ..."), it never implies overall
    superiority.
    """

    if anthropic_value is None or openai_value is None:
        return f"{label}: could not be compared (unavailable for one or both providers)."

    anthropic_str = f"{anthropic_value:.{decimals}f}{unit}"
    openai_str = f"{openai_value:.{decimals}f}{unit}"

    if _is_tie(anthropic_value, openai_value, threshold):
        return (
            f"{label}: tie within the configured {threshold:.0%} tolerance "
            f"(anthropic={anthropic_str}, openai={openai_str})."
        )

    anthropic_is_higher = anthropic_value > openai_value
    comparison_word = "higher" if higher_is_better else "lower"
    leader = "anthropic" if anthropic_is_higher == higher_is_better else "openai"
    return f"{label}: {leader} had the {comparison_word} value (anthropic={anthropic_str}, openai={openai_str})."


def compare_providers(
    anthropic_answer: ModelAnswer,
    openai_answer: ModelAnswer,
    anthropic_metrics: ProviderMetrics,
    openai_metrics: ProviderMetrics,
    embedding_provider: EmbeddingProvider,
    tie_threshold: float,
) -> ComparisonResult:
    """Build a transparent, non-judgmental Claude-vs-OpenAI comparison.

    Deliberately never declares an overall "winner" or an accuracy score:
    `comparison_notes` reports each measurable metric's difference (or tie)
    separately, and `comparison_status` makes success/failure explicit
    rather than folding it into a score. Answer length is never compared
    here -- a longer answer is not, by itself, a better one.
    """

    anthropic_ok = anthropic_answer.status == _STATUS_SUCCESS
    openai_ok = openai_answer.status == _STATUS_SUCCESS

    if anthropic_ok and openai_ok:
        comparison_status = "both_successful"
    elif anthropic_ok:
        comparison_status = "anthropic_succeeded_openai_did_not"
    elif openai_ok:
        comparison_status = "openai_succeeded_anthropic_did_not"
    else:
        comparison_status = "neither_succeeded"

    notes: List[str] = []

    if not anthropic_ok:
        detail = f" ({anthropic_answer.error})" if anthropic_answer.error else ""
        notes.append(f"anthropic did not return a successful answer (status='{anthropic_answer.status}'){detail}.")
    if not openai_ok:
        detail = f" ({openai_answer.error})" if openai_answer.error else ""
        notes.append(f"openai did not return a successful answer (status='{openai_answer.status}'){detail}.")

    notes.append(
        _describe_metric_diff(
            "latency", " ms", anthropic_metrics.latency_ms, openai_metrics.latency_ms, tie_threshold, higher_is_better=False
        )
    )
    notes.append(
        _describe_metric_diff(
            "estimated cost",
            " USD",
            anthropic_metrics.estimated_cost_usd,
            openai_metrics.estimated_cost_usd,
            tie_threshold,
            higher_is_better=False,
            decimals=6,
        )
    )

    if comparison_status != "neither_succeeded":
        notes.append(
            _describe_metric_diff(
                "citation coverage",
                "",
                anthropic_metrics.citation_coverage,
                openai_metrics.citation_coverage,
                tie_threshold,
                higher_is_better=True,
            )
        )
        notes.append(
            _describe_metric_diff(
                "mean citation relevance",
                "",
                anthropic_metrics.mean_citation_relevance,
                openai_metrics.mean_citation_relevance,
                tie_threshold,
                higher_is_better=True,
            )
        )
        if anthropic_metrics.grounded_term_ratio is not None or openai_metrics.grounded_term_ratio is not None:
            notes.append(
                _describe_metric_diff(
                    "grounded-term ratio",
                    "",
                    anthropic_metrics.grounded_term_ratio,
                    openai_metrics.grounded_term_ratio,
                    tie_threshold,
                    higher_is_better=True,
                )
            )
            notes.append(_GROUNDED_TERM_RATIO_DISCLAIMER)

        notes.append(
            f"citation count: anthropic cited {anthropic_metrics.valid_citation_count} source(s); "
            f"openai cited {openai_metrics.valid_citation_count}."
        )
        notes.append(_CITATION_COUNT_DISCLAIMER)

    agreement_score: Optional[float] = None
    if comparison_status == "both_successful":
        try:
            anthropic_vector = embedding_provider.embed_query(anthropic_answer.answer)
            openai_vector = embedding_provider.embed_query(openai_answer.answer)
            agreement_score = _cosine_similarity(anthropic_vector, openai_vector)
        except Exception:
            logger.error("Failed to compute answer_agreement_score.")
            agreement_score = None

        if agreement_score is not None:
            notes.append(_ANSWER_AGREEMENT_DISCLAIMER)
        else:
            notes.append("answer_agreement_score could not be computed.")

    if anthropic_metrics.estimated_cost_usd is not None and openai_metrics.estimated_cost_usd is not None:
        cost_difference = round(anthropic_metrics.estimated_cost_usd - openai_metrics.estimated_cost_usd, _COST_DECIMALS)
    else:
        cost_difference = None

    return ComparisonResult(
        answer_agreement_score=agreement_score,
        latency_difference_ms=anthropic_metrics.latency_ms - openai_metrics.latency_ms,
        estimated_cost_difference_usd=cost_difference,
        comparison_status=comparison_status,
        comparison_notes=notes,
    )
