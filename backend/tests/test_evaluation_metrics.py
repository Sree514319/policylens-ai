"""Tests for `app.services.evaluation.metrics` (per-provider metrics and
Claude-vs-OpenAI comparison). Pure computation -- no network, no model.
"""

import pytest

from app.services.evaluation.metrics import (
    ComparisonResult,
    ProviderMetrics,
    _clamp,
    _cosine_similarity,
    _is_tie,
    _mean_citation_relevance,
    _tokenize,
    compare_providers,
    evaluate_provider,
    evaluate_providers,
)
from app.services.llm.rag import Citation, ModelAnswer
from app.services.retrieval.embeddings import FakeEmbeddingProvider
from app.core.config import Settings


def _citation(chunk_id="c1", label="S1", relevance_score=0.8, excerpt="the overdraft fee is thirty five dollars"):
    return Citation(
        source_label=label,
        chunk_id=chunk_id,
        document_id="doc1",
        source_filename="policy.pdf",
        page_number=1,
        excerpt=excerpt,
        relevance_score=relevance_score,
    )


def _answer(**overrides):
    defaults = dict(
        provider="anthropic",
        model="m",
        status="success",
        answer="The overdraft fee is thirty five dollars.",
        citations=[],
        latency_ms=100.0,
        input_tokens=None,
        output_tokens=None,
        error=None,
    )
    defaults.update(overrides)
    return ModelAnswer(**defaults)


# --- citation_coverage -----------------------------------------------------------


def test_citation_coverage_is_fraction_of_evidence_covered():
    metrics = evaluate_provider(
        _answer(citations=[_citation("c1"), _citation("c2")]),
        evidence_count=4,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.citation_coverage == 0.5


def test_citation_coverage_deduplicates_citations_with_the_same_chunk_id():
    metrics = evaluate_provider(
        _answer(citations=[_citation("c1"), _citation("c1"), _citation("c1")]),
        evidence_count=2,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.valid_citation_count == 3  # raw count, not deduplicated
    assert metrics.citation_coverage == 0.5  # 1 unique / 2 evidence


def test_citation_coverage_is_zero_with_no_evidence_at_all():
    metrics = evaluate_provider(
        _answer(status="insufficient_evidence", answer="no evidence", citations=[]),
        evidence_count=0,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.citation_coverage == 0.0


def test_citation_coverage_is_clamped_to_one():
    # Defensive: more unique citations than the advertised evidence pool
    # should never happen, but the formula must not exceed 1.0 if it does.
    metrics = evaluate_provider(
        _answer(citations=[_citation("c1"), _citation("c2"), _citation("c3")]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.citation_coverage == 1.0


# --- mean_citation_relevance -------------------------------------------------------


def test_mean_citation_relevance_is_none_with_no_citations():
    metrics = evaluate_provider(
        _answer(citations=[]),
        evidence_count=2,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.mean_citation_relevance is None
    assert "mean_citation_relevance not computed" in " ".join(metrics.evaluation_notes)


def test_mean_citation_relevance_is_the_arithmetic_mean():
    metrics = evaluate_provider(
        _answer(citations=[_citation("c1", relevance_score=0.4), _citation("c2", relevance_score=0.8)]),
        evidence_count=2,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.mean_citation_relevance == pytest.approx(0.6)


def test_mean_citation_relevance_is_clamped_to_valid_range():
    metrics = evaluate_provider(
        _answer(citations=[_citation("c1", relevance_score=1.5)]),  # defensively out-of-range
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.mean_citation_relevance == 1.0


# --- grounded_term_ratio: normalization, stop words, punctuation, unicode ----------


def test_grounded_term_ratio_ignores_stopwords_and_punctuation():
    answer = "The overdraft fee, is thirty-five dollars!"
    citation = _citation(excerpt="overdraft fee thirty five dollars mentioned here")
    metrics = evaluate_provider(
        _answer(answer=answer, citations=[citation]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    # "the"/"is" are stop words and excluded from the denominator; every
    # remaining meaningful term ("overdraft", "fee", "thirty", "five",
    # "dollars") appears in the excerpt -- ratio must be a full 1.0, not
    # penalized for stop words or the comma/exclamation punctuation.
    assert metrics.grounded_term_ratio == 1.0


def test_grounded_term_ratio_is_none_when_answer_has_no_meaningful_terms():
    metrics = evaluate_provider(
        _answer(answer="The is a an of to."),  # all stop words
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.grounded_term_ratio is None
    assert any("no meaningful terms" in note for note in metrics.evaluation_notes)


def test_grounded_term_ratio_is_zero_when_no_citations_exist():
    metrics = evaluate_provider(
        _answer(answer="Overdraft policy details apply broadly.", citations=[]),
        evidence_count=0,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.grounded_term_ratio == 0.0


def test_grounded_term_ratio_respects_minimum_term_length():
    answer = "Fee is 35 usd for overdrafts."
    citation = _citation(excerpt="35 usd overdraft fee schedule")
    short_min = evaluate_provider(
        _answer(answer=answer, citations=[citation]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=2,
    )
    long_min = evaluate_provider(
        _answer(answer=answer, citations=[citation]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=5,
    )
    # A higher minimum length excludes more short terms ("fee", "usd") from
    # the denominator -- the two configurations must not be identical.
    assert short_min.grounded_term_ratio != long_min.grounded_term_ratio


def test_grounded_term_ratio_only_computed_for_successful_answers():
    metrics = evaluate_provider(
        _answer(status="insufficient_evidence", answer="Overdraft fee details.", citations=[]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.grounded_term_ratio is None
    assert any("only computed for successful answers" in note for note in metrics.evaluation_notes)


def test_grounded_term_ratio_handles_unicode_answer_text():
    # Non-ASCII characters are stripped by the [a-z0-9]+ tokenizer (a
    # documented limitation, not a crash) -- "café" tokenizes to "caf".
    answer = "The café overdraft fee is thirty five dollars."
    citation = _citation(excerpt="overdraft fee thirty five dollars at this café")
    metrics = evaluate_provider(
        _answer(answer=answer, citations=[citation]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.grounded_term_ratio == 1.0  # "caf" matches on both sides consistently


def test_grounded_term_ratio_disclaimer_present_whenever_computed():
    metrics = evaluate_provider(
        _answer(answer="Overdraft fee applies.", citations=[_citation(excerpt="overdraft fee applies here")]),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.grounded_term_ratio is not None
    assert any("does NOT verify factual correctness" in note for note in metrics.evaluation_notes)


# --- estimated_cost_usd -------------------------------------------------------------


def test_cost_is_none_when_token_usage_is_absent():
    metrics = evaluate_provider(
        _answer(input_tokens=None, output_tokens=None),
        evidence_count=1,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd is None
    assert any("did not report token usage" in note for note in metrics.evaluation_notes)


def test_cost_is_none_when_only_one_of_input_or_output_tokens_is_present():
    metrics = evaluate_provider(
        _answer(input_tokens=100, output_tokens=None),
        evidence_count=1,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd is None


def test_cost_is_none_when_pricing_is_not_configured():
    metrics = evaluate_provider(
        _answer(input_tokens=100, output_tokens=50),
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd is None
    assert any("pricing is not configured" in note for note in metrics.evaluation_notes)


def test_cost_formula_is_correct():
    # 1,000,000 input tokens @ $3/M + 500,000 output tokens @ $15/M
    # = $3.00 + $7.50 = $10.50
    metrics = evaluate_provider(
        _answer(input_tokens=1_000_000, output_tokens=500_000),
        evidence_count=1,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd == pytest.approx(10.50)


def test_cost_is_none_when_cost_tracking_is_disabled_even_with_full_pricing():
    metrics = evaluate_provider(
        _answer(input_tokens=100, output_tokens=50),
        evidence_count=1,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=False,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd is None
    assert any("Cost tracking is disabled" in note for note in metrics.evaluation_notes)


def test_cost_of_zero_tokens_is_zero_not_none():
    metrics = evaluate_provider(
        _answer(input_tokens=0, output_tokens=0),
        evidence_count=1,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.estimated_cost_usd == 0.0


# --- answer_length / provider error passthrough -------------------------------------


def test_answer_length_is_a_word_count():
    metrics = evaluate_provider(
        _answer(answer="one two three four"),
        evidence_count=0,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.answer_length == 4


def test_error_status_produces_safe_zeroed_metrics_not_a_crash():
    metrics = evaluate_provider(
        _answer(status="error", answer="", citations=[], error="The Anthropic API rate limit was exceeded."),
        evidence_count=3,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics.status == "error"
    assert metrics.valid_citation_count == 0
    assert metrics.citation_coverage == 0.0
    assert metrics.grounded_term_ratio is None
    assert metrics.answer_length == 0


# --- evaluate_providers() resolves per-provider pricing ------------------------------


def test_evaluate_providers_resolves_pricing_per_provider_name():
    settings = Settings(
        anthropic_input_cost_per_million=3.0,
        anthropic_output_cost_per_million=15.0,
        openai_input_cost_per_million=0.5,
        openai_output_cost_per_million=1.5,
    )
    answers = [
        _answer(provider="anthropic", input_tokens=1_000_000, output_tokens=1_000_000),
        _answer(provider="openai", input_tokens=1_000_000, output_tokens=1_000_000),
    ]
    results = evaluate_providers(answers, evidence_count=0, settings=settings)

    assert results["anthropic"].estimated_cost_usd == pytest.approx(18.0)
    assert results["openai"].estimated_cost_usd == pytest.approx(2.0)


# --- _cosine_similarity / _is_tie (low-level helpers) ---------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_is_none_for_mismatched_or_empty_vectors():
    assert _cosine_similarity([], [1.0]) is None
    assert _cosine_similarity([1.0, 2.0], [1.0]) is None


def test_is_tie_true_when_both_values_are_zero():
    assert _is_tie(0.0, 0.0, threshold=0.05) is True


def test_is_tie_respects_relative_threshold():
    assert _is_tie(100.0, 104.0, threshold=0.05) is True  # 4% diff, within 5%
    assert _is_tie(100.0, 110.0, threshold=0.05) is False  # 10% diff, outside 5%


# --- compare_providers(): status, ties, agreement, one/both failures ------------------


def _metrics_for(answer, evidence_count=1, **price_kwargs):
    defaults = dict(
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    defaults.update(price_kwargs)
    return evaluate_provider(answer, evidence_count=evidence_count, **defaults)


def test_compare_both_successful_computes_agreement_and_no_winner_language():
    anthropic_answer = _answer(provider="anthropic", answer="The fee is thirty five dollars.", latency_ms=100.0)
    openai_answer = _answer(provider="openai", answer="The fee is thirty five dollars.", latency_ms=100.0)

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        tie_threshold=0.05,
    )

    assert comparison.comparison_status == "both_successful"
    assert comparison.answer_agreement_score == pytest.approx(1.0)  # identical text -> identical fake embedding
    joined_notes = " ".join(comparison.comparison_notes)
    assert "won" not in joined_notes.lower()
    assert "winner" not in joined_notes.lower()
    assert "accuracy" not in joined_notes.lower()


def test_compare_identical_answers_have_higher_agreement_than_different_answers():
    anthropic_answer = _answer(provider="anthropic", answer="The fee is thirty five dollars.")
    same = _answer(provider="openai", answer="The fee is thirty five dollars.")
    different = _answer(provider="openai", answer="Interest accrues daily on the outstanding balance.")

    identical_comparison = compare_providers(
        anthropic_answer, same, _metrics_for(anthropic_answer), _metrics_for(same), FakeEmbeddingProvider(), 0.05
    )
    different_comparison = compare_providers(
        anthropic_answer,
        different,
        _metrics_for(anthropic_answer),
        _metrics_for(different),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert identical_comparison.answer_agreement_score > different_comparison.answer_agreement_score


def test_compare_status_when_only_anthropic_succeeds():
    anthropic_answer = _answer(provider="anthropic", status="success")
    openai_answer = _answer(provider="openai", status="error", answer="", error="The OpenAI API rate limit was exceeded.")

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert comparison.comparison_status == "anthropic_succeeded_openai_did_not"
    assert comparison.answer_agreement_score is None
    assert any("openai" in note and "rate limit" in note for note in comparison.comparison_notes)


def test_compare_status_when_only_openai_succeeds():
    anthropic_answer = _answer(provider="anthropic", status="error", answer="", error="Authentication with the Anthropic API failed.")
    openai_answer = _answer(provider="openai", status="success")

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert comparison.comparison_status == "openai_succeeded_anthropic_did_not"
    assert comparison.answer_agreement_score is None


def test_compare_status_when_neither_succeeds():
    anthropic_answer = _answer(provider="anthropic", status="error", answer="", error="Could not connect to the Anthropic API.")
    openai_answer = _answer(provider="openai", status="error", answer="", error="Could not connect to the OpenAI API.")

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert comparison.comparison_status == "neither_succeeded"
    assert comparison.answer_agreement_score is None
    # No citation-quality comparisons are meaningful when neither succeeded.
    joined_notes = " ".join(comparison.comparison_notes)
    assert "citation coverage" not in joined_notes
    assert "grounded-term ratio" not in joined_notes


def test_compare_reports_a_tie_within_threshold_not_an_advantage():
    anthropic_answer = _answer(provider="anthropic", latency_ms=100.0)
    openai_answer = _answer(provider="openai", latency_ms=102.0)  # 2% diff

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        tie_threshold=0.05,
    )

    latency_note = next(note for note in comparison.comparison_notes if note.startswith("latency"))
    assert "tie" in latency_note


def test_compare_reports_an_advantage_outside_the_tie_threshold():
    anthropic_answer = _answer(provider="anthropic", latency_ms=100.0)
    openai_answer = _answer(provider="openai", latency_ms=500.0)  # far outside 5%

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        tie_threshold=0.05,
    )

    latency_note = next(note for note in comparison.comparison_notes if note.startswith("latency"))
    assert "tie" not in latency_note
    assert "anthropic had the lower value" in latency_note


def test_compare_latency_difference_sign_convention():
    anthropic_answer = _answer(provider="anthropic", latency_ms=150.0)
    openai_answer = _answer(provider="openai", latency_ms=100.0)

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert comparison.latency_difference_ms == pytest.approx(50.0)


def test_compare_cost_difference_is_none_when_either_cost_is_unavailable():
    anthropic_answer = _answer(provider="anthropic", input_tokens=100, output_tokens=50)
    openai_answer = _answer(provider="openai", input_tokens=None, output_tokens=None)

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer, input_price_per_million=3.0, output_price_per_million=15.0),
        _metrics_for(openai_answer, input_price_per_million=0.5, output_price_per_million=1.5),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert comparison.estimated_cost_difference_usd is None


def test_compare_cost_difference_is_computed_when_both_costs_are_available():
    anthropic_answer = _answer(provider="anthropic", input_tokens=1_000_000, output_tokens=1_000_000)
    openai_answer = _answer(provider="openai", input_tokens=1_000_000, output_tokens=1_000_000)

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer, input_price_per_million=3.0, output_price_per_million=15.0),
        _metrics_for(openai_answer, input_price_per_million=0.5, output_price_per_million=1.5),
        FakeEmbeddingProvider(),
        0.05,
    )

    # anthropic: 3+15=18.0, openai: 0.5+1.5=2.0 -> diff 16.0
    assert comparison.estimated_cost_difference_usd == pytest.approx(16.0)


def test_compare_never_declares_a_winner_from_answer_length_alone():
    anthropic_answer = _answer(provider="anthropic", answer="Short answer.")
    openai_answer = _answer(
        provider="openai",
        answer="This is a much, much longer answer that repeats itself many times over many words.",
    )

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        FakeEmbeddingProvider(),
        0.05,
    )

    joined_notes = " ".join(comparison.comparison_notes).lower()
    assert "length" not in joined_notes
    assert "longer" not in joined_notes
    assert "shorter" not in joined_notes


def test_compare_citation_count_note_is_informational_not_a_verdict():
    anthropic_answer = _answer(provider="anthropic", citations=[_citation("c1")])
    openai_answer = _answer(provider="openai", citations=[_citation("c1"), _citation("c2")])

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer, evidence_count=2),
        _metrics_for(openai_answer, evidence_count=2),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert any("citation count" in note for note in comparison.comparison_notes)
    assert any("not as a quality signal" in note for note in comparison.comparison_notes)


# --- Financial-value tokenization: no misleading overlap between different amounts ----


def test_different_dollar_amounts_do_not_spuriously_overlap():
    # Before number-aware tokenization, "$1,000.00" and "$5,000.00" both
    # fragmented down to a shared "000"/"00" token and would register as
    # (misleading) overlap despite being different amounts.
    tokens_1000 = set(_tokenize("The fee is $1,000.00 per year."))
    tokens_5000 = set(_tokenize("The fee is $5,000.00 per year."))
    assert "1000.00" in tokens_1000
    assert "5000.00" in tokens_5000
    assert "1000.00" not in tokens_5000
    assert "5000.00" not in tokens_1000
    # No stray digit-only fragments ("000", "00", "1", "5") should survive
    # as separate tokens that could accidentally match each other.
    assert not ({"000", "00"} & tokens_1000)
    assert not ({"000", "00"} & tokens_5000)


def test_dollar_sign_and_comma_formatting_does_not_affect_matching():
    # The same numeric value, with or without currency symbol/commas,
    # should normalize identically.
    assert _tokenize("$1,000.00") == _tokenize("1000.00")


def test_negative_and_positive_percentages_remain_distinct():
    tokens_negative = set(_tokenize("APR is -2.5% this quarter."))
    tokens_positive = set(_tokenize("APR is 2.5% this quarter."))
    assert "-2.5%" in tokens_negative
    assert "2.5%" in tokens_positive
    assert "-2.5%" not in tokens_positive
    assert "2.5%" not in tokens_negative


def test_percentage_values_are_grounded_when_actually_quoted_and_not_when_mismatched():
    matching_citation = _citation(excerpt="The current APR is 4.95% for this account.")
    answer_matching = _answer(answer="The APR is 4.95%.", citations=[matching_citation])
    metrics_matching = evaluate_provider(
        answer_matching,
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    assert metrics_matching.grounded_term_ratio == 1.0

    mismatched_citation = _citation(excerpt="The current APR is 9.99% for this account.")
    answer_mismatched = _answer(answer="The APR is 4.95%.", citations=[mismatched_citation])
    metrics_mismatched = evaluate_provider(
        answer_mismatched,
        evidence_count=1,
        input_price_per_million=None,
        output_price_per_million=None,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    # "4.95%" must NOT be reported as grounded when the cited excerpt
    # actually says a different percentage.
    assert metrics_mismatched.grounded_term_ratio < metrics_matching.grounded_term_ratio


# --- Unicode: diacritics are now normalized (a safe, narrow improvement) -------------


def test_diacritics_are_stripped_so_accented_and_plain_forms_match():
    # A real improvement over pure-ASCII stripping: "café" now normalizes
    # to "cafe" and matches the unaccented spelling, instead of truncating
    # to "caf" and failing to match either form.
    assert _tokenize("café") == ["cafe"]
    assert _tokenize("cafe") == ["cafe"]


def test_non_latin_scripts_are_still_documented_as_dropped():
    # Diacritic-stripping only helps Latin-script accents; non-Latin
    # scripts remain outside the ASCII-only token pattern. This is a
    # known, narrower (not eliminated) limitation -- not a crash.
    assert _tokenize("北京") == []


# --- NaN safety: a malformed input must never silently look like a valid score -------


def test_clamp_treats_nan_as_the_low_bound_not_a_silent_pass_through():
    assert _clamp(float("nan"), 0.0, 1.0) == 0.0


def test_mean_citation_relevance_is_none_when_a_relevance_score_is_nan():
    citation = _citation(relevance_score=float("nan"))
    assert _mean_citation_relevance([citation]) is None


def test_cosine_similarity_is_none_for_a_vector_containing_nan():
    assert _cosine_similarity([1.0, float("nan")], [1.0, 2.0]) is None


def test_cosine_similarity_is_none_for_a_vector_containing_infinity():
    assert _cosine_similarity([1.0, float("inf")], [1.0, 2.0]) is None
    assert _cosine_similarity([1.0, 2.0], [float("-inf"), 2.0]) is None


# --- Cost rounding -------------------------------------------------------------------


def test_estimated_cost_is_rounded_to_a_stable_precision():
    metrics = evaluate_provider(
        _answer(input_tokens=333, output_tokens=777),
        evidence_count=0,
        input_price_per_million=3.0,
        output_price_per_million=15.0,
        cost_tracking_enabled=True,
        grounded_term_min_length=3,
    )
    # 333/1e6*3 + 777/1e6*15 = 0.000999 + 0.011655 = 0.012654 -- exactly,
    # with no floating-point tail digits leaking through.
    assert metrics.estimated_cost_usd == 0.012654
    assert round(metrics.estimated_cost_usd, 6) == metrics.estimated_cost_usd


def test_cost_difference_is_rounded_to_a_stable_precision():
    anthropic_answer = _answer(provider="anthropic", input_tokens=1, output_tokens=1)
    openai_answer = _answer(provider="openai", input_tokens=1, output_tokens=1)

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer, input_price_per_million=0.30000001, output_price_per_million=0.1),
        _metrics_for(openai_answer, input_price_per_million=0.1, output_price_per_million=0.1),
        FakeEmbeddingProvider(),
        0.05,
    )

    assert round(comparison.estimated_cost_difference_usd, 6) == comparison.estimated_cost_difference_usd


# --- Agreement-computation failure does not discard valid model answers --------------


class _RaisingEmbeddingProvider(FakeEmbeddingProvider):
    def embed_query(self, text):
        raise RuntimeError("embedding backend unavailable")


class _MalformedEmbeddingProvider(FakeEmbeddingProvider):
    """Returns a vector with fewer dimensions each call -- simulates a
    provider silently returning inconsistent/malformed output rather than
    raising."""

    def __init__(self):
        self._call_count = 0

    def embed_query(self, text):
        self._call_count += 1
        return [1.0] * self._call_count  # dimension changes between calls


def test_embedding_exception_during_agreement_yields_null_with_a_safe_note_not_a_crash():
    anthropic_answer = _answer(provider="anthropic", answer="The fee is thirty five dollars.")
    openai_answer = _answer(provider="openai", answer="The fee is thirty five dollars.")

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        _RaisingEmbeddingProvider(),
        tie_threshold=0.05,
    )

    assert comparison.answer_agreement_score is None
    assert comparison.comparison_status == "both_successful"
    assert any("answer_agreement_score could not be computed" in note for note in comparison.comparison_notes)


def test_embedding_dimension_mismatch_during_agreement_yields_null_not_a_crash():
    anthropic_answer = _answer(provider="anthropic", answer="The fee is thirty five dollars.")
    openai_answer = _answer(provider="openai", answer="The fee is thirty five dollars.")

    comparison = compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        _MalformedEmbeddingProvider(),
        tie_threshold=0.05,
    )

    assert comparison.answer_agreement_score is None


def test_agreement_failure_does_not_discard_the_model_results_themselves():
    # The caller (the /compare route) builds model_results from
    # `model_answers` independently of `compare_providers` -- this test
    # locks in that `compare_providers` itself never mutates or empties
    # the answers it was given, even when embedding fails.
    anthropic_answer = _answer(provider="anthropic", answer="The fee is thirty five dollars.")
    openai_answer = _answer(provider="openai", answer="The fee is thirty five dollars.")

    compare_providers(
        anthropic_answer,
        openai_answer,
        _metrics_for(anthropic_answer),
        _metrics_for(openai_answer),
        _RaisingEmbeddingProvider(),
        tie_threshold=0.05,
    )

    assert anthropic_answer.answer == "The fee is thirty five dollars."
    assert openai_answer.answer == "The fee is thirty five dollars."


# --- Reversed / duplicate provider order is irrelevant -- mapping is by name ---------


def test_evaluate_providers_keys_are_correct_regardless_of_input_list_order():
    settings = Settings(
        anthropic_input_cost_per_million=3.0,
        anthropic_output_cost_per_million=15.0,
        openai_input_cost_per_million=0.5,
        openai_output_cost_per_million=1.5,
    )
    reversed_order = [
        _answer(provider="openai", input_tokens=1_000_000, output_tokens=1_000_000),
        _answer(provider="anthropic", input_tokens=1_000_000, output_tokens=1_000_000),
    ]
    results = evaluate_providers(reversed_order, evidence_count=0, settings=settings)

    # Keyed by provider name, not list position -- "anthropic" must get
    # anthropic pricing even though it was second in the input list.
    assert results["anthropic"].estimated_cost_usd == pytest.approx(18.0)
    assert results["openai"].estimated_cost_usd == pytest.approx(2.0)


def test_compare_providers_result_identical_regardless_of_caller_side_ordering():
    # compare_providers takes explicit anthropic_*/openai_* parameters
    # (not a list), so there is no "order" for it to get wrong -- this
    # locks in that invariant by calling it with the two answers/metrics
    # built from a deliberately reversed source list.
    settings = Settings()
    source_answers = [
        _answer(provider="openai", latency_ms=200.0),
        _answer(provider="anthropic", latency_ms=100.0),
    ]
    metrics_by_provider = evaluate_providers(source_answers, evidence_count=0, settings=settings)
    answers_by_provider = {answer.provider: answer for answer in source_answers}

    comparison = compare_providers(
        answers_by_provider["anthropic"],
        answers_by_provider["openai"],
        metrics_by_provider["anthropic"],
        metrics_by_provider["openai"],
        FakeEmbeddingProvider(),
        0.05,
    )

    # anthropic (100ms) - openai (200ms) = -100ms, regardless of the fact
    # that openai appeared first in `source_answers`.
    assert comparison.latency_difference_ms == pytest.approx(-100.0)


# --- Duplicate providers never cause duplicate evaluation/calls ----------------------


def test_evaluate_providers_with_a_duplicated_provider_name_keeps_only_one_entry():
    # evaluate_providers is keyed by provider name -- if it were ever
    # given two answers for the same provider (shouldn't happen given
    # CompareRequest's validation, but defensively), the dict naturally
    # collapses to one entry rather than silently duplicating work.
    settings = Settings()
    duplicated = [
        _answer(provider="anthropic", latency_ms=100.0),
        _answer(provider="anthropic", latency_ms=999.0),
    ]
    results = evaluate_providers(duplicated, evidence_count=0, settings=settings)

    assert list(results.keys()) == ["anthropic"]
    # The later entry wins (last-write-wins dict semantics) -- deterministic,
    # not duplicated work.
    assert results["anthropic"].latency_ms == 999.0


# --- Additional tie-threshold edge cases ----------------------------------------------


def test_is_tie_for_very_small_nonzero_values_uses_the_epsilon_floor():
    # Both values are far smaller than the 1e-9 epsilon floor used as the
    # minimum scale -- the comparison is effectively against that floor,
    # not the (nearly meaningless) values themselves.
    assert _is_tie(1e-10, 2e-10, threshold=0.05) is False
    assert _is_tie(1e-10, 1.01e-10, threshold=0.05) is True


def test_is_tie_at_exactly_the_threshold_boundary_is_a_tie():
    # |100 - 105| = 5, which is exactly 5% of 105 (the larger magnitude) --
    # the boundary itself must count as a tie ("<=", not "<").
    assert _is_tie(100.0, 105.0, threshold=0.05) is True


def test_is_tie_just_outside_the_threshold_boundary_is_not_a_tie():
    # |100 - 110| = 10, which is ~9.1% of 110 -- clearly outside a 5% tolerance.
    assert _is_tie(100.0, 110.0, threshold=0.05) is False


def test_is_tie_symmetric_regardless_of_argument_order():
    assert _is_tie(100.0, 104.0, threshold=0.05) == _is_tie(104.0, 100.0, threshold=0.05)
