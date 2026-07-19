"""Tests for `components/formatting.py` -- pure functions, no Streamlit
runtime needed.
"""

import pytest

from streamlit_app.components import formatting as fmt


# --- Null / None handling -------------------------------------------------------------


@pytest.mark.parametrize(
    "func,args",
    [
        (fmt.format_percentage, (None,)),
        (fmt.format_relevance, (None,)),
        (fmt.format_ratio_metric, (None,)),
        (fmt.format_latency, (None,)),
        (fmt.format_cost, (None,)),
        (fmt.format_signed_cost_difference, (None,)),
        (fmt.format_agreement_score, (None,)),
    ],
)
def test_none_input_never_renders_as_a_fake_zero(func, args):
    result = func(*args)
    assert result == fmt.NOT_AVAILABLE
    # Explicitly must not silently render as "0" / "0%" / "$0.00" -- that
    # would misrepresent "unavailable" as "measured to be zero".
    assert result not in {"0", "0%", "$0.00", "0.00", "0 ms"}


def test_format_tokens_both_none_is_not_reported():
    assert fmt.format_tokens(None, None) == fmt.NOT_REPORTED


def test_format_tokens_partial_is_rendered_with_placeholder():
    assert fmt.format_tokens(100, None) == "100 in / ? out"
    assert fmt.format_tokens(None, 50) == "? in / 50 out"


def test_format_tokens_both_present():
    assert fmt.format_tokens(100, 50) == "100 in / 50 out"


# --- Percentages / ratios ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "0%"), (1.0, "100%"), (0.5, "50%"), (0.873, "87%")],
)
def test_format_percentage_values(value, expected):
    assert fmt.format_percentage(value) == expected


# --- Latency -----------------------------------------------------------------------------


def test_format_latency_sub_second():
    assert fmt.format_latency(842.3) == "842 ms"


def test_format_latency_seconds():
    assert fmt.format_latency(1500.0) == "1.50 s"


def test_format_latency_zero():
    assert fmt.format_latency(0.0) == "0 ms"


def test_format_signed_latency_difference_directions():
    assert "Claude was slower" in fmt.format_signed_latency_difference(100.0)
    assert "OpenAI was slower" in fmt.format_signed_latency_difference(-100.0)
    assert fmt.format_signed_latency_difference(0.0) == "Identical latency"


# --- Cost --------------------------------------------------------------------------------


def test_format_cost_zero_is_exact_not_none():
    assert fmt.format_cost(0.0) == "$0.00"


def test_format_cost_small_value_uses_more_precision():
    assert fmt.format_cost(0.000045) == "$0.000045"


def test_format_cost_larger_value_uses_standard_precision():
    assert fmt.format_cost(1.5) == "$1.5000"


def test_format_signed_cost_difference_directions():
    assert "Claude cost" in fmt.format_signed_cost_difference(0.01)
    assert "OpenAI cost" in fmt.format_signed_cost_difference(-0.01)
    assert fmt.format_signed_cost_difference(0.0) == "Identical estimated cost"


# --- Agreement score: must read as a heuristic, never a correctness score ------------------


def test_format_agreement_score_labels_it_as_similarity_not_correctness():
    result = fmt.format_agreement_score(0.94)
    assert "0.94" in result
    assert "cosine similarity" in result
    assert "semantic" in result


# --- Status labels -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [("success", "Success"), ("insufficient_evidence", "Insufficient evidence"), ("error", "Error")],
)
def test_status_label(status, expected):
    assert fmt.status_label(status) == expected


def test_status_label_falls_back_gracefully_for_unknown_status():
    assert fmt.status_label("something_new") == "Something new"


@pytest.mark.parametrize(
    "comparison_status",
    ["both_successful", "anthropic_succeeded_openai_did_not", "openai_succeeded_anthropic_did_not", "neither_succeeded"],
)
def test_comparison_status_label_covers_every_backend_value(comparison_status):
    label = fmt.comparison_status_label(comparison_status)
    assert label and label[0].isupper()


# --- No "accuracy"/"winner"/"correct" language anywhere in formatting output ------------


@pytest.mark.parametrize(
    "text",
    [
        fmt.format_agreement_score(0.94),
        fmt.format_agreement_score(None),
        fmt.format_ratio_metric(0.8),
        fmt.format_ratio_metric(None),
        fmt.comparison_status_label("both_successful"),
        fmt.status_label("success"),
    ],
)
def test_no_winner_or_accuracy_language_in_formatted_output(text):
    lowered = text.lower()
    assert "winner" not in lowered
    assert "accuracy" not in lowered
    assert " won" not in lowered


# --- PII summary text ------------------------------------------------------------------------


def test_pii_summary_when_nothing_detected():
    assert fmt.pii_summary_text(False, 0, []) == "No PII detected in this document."


def test_pii_summary_when_detected_lists_categories():
    text = fmt.pii_summary_text(True, 2, ["EMAIL", "SSN"])
    assert "2 PII entities masked" in text
    assert "EMAIL" in text
    assert "SSN" in text


def test_pii_summary_singular_entity_wording():
    text = fmt.pii_summary_text(True, 1, ["SSN"])
    assert "1 PII entity masked" in text


# --- Truncate (defense in depth) ----------------------------------------------------------


def test_truncate_short_text_is_unchanged():
    assert fmt.truncate("short", 100) == "short"


def test_truncate_long_text_is_capped_with_ellipsis():
    result = fmt.truncate("a" * 500, 50)
    assert len(result) <= 53
    assert result.endswith("...")


# --- NaN / infinity: must render as "Not available", never a literal "nan"/"inf" ------


_NAN_AND_INF_FORMATTERS = [
    fmt.format_percentage,
    fmt.format_relevance,
    fmt.format_ratio_metric,
    fmt.format_latency,
    fmt.format_cost,
    fmt.format_agreement_score,
]


@pytest.mark.parametrize("formatter", _NAN_AND_INF_FORMATTERS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_render_as_not_available(formatter, bad_value):
    result = formatter(bad_value)
    assert result == fmt.NOT_AVAILABLE
    assert "nan" not in result.lower()
    assert "inf" not in result.lower()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_signed_latency_difference_handles_non_finite_values(bad_value):
    result = fmt.format_signed_latency_difference(bad_value)
    assert result == fmt.NOT_AVAILABLE


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_signed_cost_difference_handles_non_finite_values(bad_value):
    result = fmt.format_signed_cost_difference(bad_value)
    assert result == fmt.NOT_AVAILABLE


def test_is_missing_treats_plain_ints_as_present_not_missing():
    # `_is_missing` only special-cases `float` NaN/inf -- an ordinary int
    # (e.g. a token count) is never "missing" just because it's not a float.
    assert fmt._is_missing(0) is False
    assert fmt._is_missing(42) is False


def test_is_missing_none_is_missing():
    assert fmt._is_missing(None) is True


def test_is_missing_ordinary_float_is_not_missing():
    assert fmt._is_missing(0.5) is False
    assert fmt._is_missing(0.0) is False
