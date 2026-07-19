"""Pure, Streamlit-independent formatting helpers.

Every function here takes plain data and returns a plain string (or a
small tuple/constant) -- no `streamlit` import, no side effects. Kept
separate from `render.py` specifically so this formatting logic (null
handling, status labels, "not accuracy" phrasing) is unit-testable
without any Streamlit runtime.
"""

import math
from typing import List, Optional

NOT_AVAILABLE = "Not available"
NOT_REPORTED = "Not reported"


def _is_missing(value: Optional[float]) -> bool:
    """True for `None`, and defensively for NaN/+-inf too.

    The backend already guards against non-finite values reaching a
    response (see Phase 7's `metrics.py`/config validation), but every
    numeric formatter here checks this anyway as a second, independent
    line of defense -- a stray `float('nan')` must render as the same
    "Not available" text as a missing value, never as the literal string
    "nan" or a bogus-looking number like "$nan" or "inf%".
    """

    if value is None:
        return True
    return isinstance(value, float) and not math.isfinite(value)

_STATUS_LABELS = {
    "success": "Success",
    "insufficient_evidence": "Insufficient evidence",
    "error": "Error",
}

_COMPARISON_STATUS_LABELS = {
    "both_successful": "Both models answered successfully",
    "anthropic_succeeded_openai_did_not": "Only Claude (Anthropic) answered successfully",
    "openai_succeeded_anthropic_did_not": "Only OpenAI answered successfully",
    "neither_succeeded": "Neither model answered successfully",
}


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status.replace("_", " ").capitalize())


def status_icon(status: str) -> str:
    return {"success": "✅", "insufficient_evidence": "⚠️", "error": "❌"}.get(status, "ℹ️")


def comparison_status_label(comparison_status: str) -> str:
    return _COMPARISON_STATUS_LABELS.get(comparison_status, comparison_status.replace("_", " ").capitalize())


def format_percentage(value: Optional[float]) -> str:
    """A [0,1] float as a percentage string, or `NOT_AVAILABLE` if missing."""

    if _is_missing(value):
        return NOT_AVAILABLE
    return f"{value * 100:.0f}%"


def format_relevance(score: Optional[float]) -> str:
    return format_percentage(score)


def format_ratio_metric(value: Optional[float]) -> str:
    """Same as `format_percentage`, used for heuristic ratio metrics
    (citation coverage, mean relevance, grounded-term ratio) -- a
    distinct name so call sites read as "this is a heuristic ratio," not
    an accuracy figure."""

    return format_percentage(value)


def format_latency(latency_ms: Optional[float]) -> str:
    if _is_missing(latency_ms):
        return NOT_AVAILABLE
    if latency_ms < 1000:
        return f"{latency_ms:.0f} ms"
    return f"{latency_ms / 1000:.2f} s"


def format_signed_latency_difference(latency_difference_ms: Optional[float]) -> str:
    if _is_missing(latency_difference_ms):
        return NOT_AVAILABLE
    magnitude = format_latency(abs(latency_difference_ms))
    if latency_difference_ms > 0:
        return f"Claude was slower by {magnitude}"
    if latency_difference_ms < 0:
        return f"OpenAI was slower by {magnitude}"
    return "Identical latency"


def format_tokens(input_tokens: Optional[int], output_tokens: Optional[int]) -> str:
    if input_tokens is None and output_tokens is None:
        return NOT_REPORTED
    input_part = str(input_tokens) if input_tokens is not None else "?"
    output_part = str(output_tokens) if output_tokens is not None else "?"
    return f"{input_part} in / {output_part} out"


def format_cost(estimated_cost_usd: Optional[float]) -> str:
    if _is_missing(estimated_cost_usd):
        return NOT_AVAILABLE
    if estimated_cost_usd == 0:
        return "$0.00"
    if estimated_cost_usd < 0.01:
        return f"${estimated_cost_usd:.6f}"
    return f"${estimated_cost_usd:.4f}"


def format_signed_cost_difference(cost_difference_usd: Optional[float]) -> str:
    if _is_missing(cost_difference_usd):
        return NOT_AVAILABLE
    magnitude = format_cost(abs(cost_difference_usd))
    if cost_difference_usd > 0:
        return f"Claude cost {magnitude} more (estimated)"
    if cost_difference_usd < 0:
        return f"OpenAI cost {magnitude} more (estimated)"
    return "Identical estimated cost"


def format_agreement_score(score: Optional[float]) -> str:
    if _is_missing(score):
        return NOT_AVAILABLE
    return f"{score:.2f} (cosine similarity, semantic closeness only)"


def format_word_count(count: int) -> str:
    return f"{count} word{'s' if count != 1 else ''}"


def pii_summary_text(detected: bool, entity_count: int, categories: List[str]) -> str:
    if not detected or entity_count == 0:
        return "No PII detected in this document."
    category_list = ", ".join(categories) if categories else "unspecified categories"
    entity_word = "entity" if entity_count == 1 else "entities"
    return f"{entity_count} PII {entity_word} masked ({category_list})."


def truncate(text: str, limit: int) -> str:
    """Defense-in-depth display cap. The backend already caps previews/
    excerpts server-side; this only guards against an unexpectedly long
    value ever reaching the UI unbounded."""

    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
