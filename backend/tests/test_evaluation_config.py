"""Tests for numeric bounds on the Phase 7 model-evaluation settings.

A misconfigured `.env` should fail fast at Settings-construction time
(Pydantic validation) rather than silently accepting a negative price, an
infinite/NaN price, or an out-of-range tie threshold.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

PRICE_FIELDS = [
    "anthropic_input_cost_per_million",
    "anthropic_output_cost_per_million",
    "openai_input_cost_per_million",
    "openai_output_cost_per_million",
]


def test_default_settings_leave_pricing_unset():
    settings = Settings()

    for field in PRICE_FIELDS:
        assert getattr(settings, field) is None
    assert settings.model_comparison_tie_threshold == 0.05
    assert settings.grounded_term_min_length == 3


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_accept_a_valid_value(field):
    settings = Settings(**{field: 3.0})
    assert getattr(settings, field) == 3.0


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_accept_zero(field):
    settings = Settings(**{field: 0.0})
    assert getattr(settings, field) == 0.0


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_reject_negative_values(field):
    with pytest.raises(ValidationError):
        Settings(**{field: -0.01})


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_reject_positive_infinity(field):
    # `ge=0` alone does NOT reject +inf (inf >= 0 is True) -- this is the
    # specific gap `allow_inf_nan=False` closes.
    with pytest.raises(ValidationError):
        Settings(**{field: float("inf")})


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_reject_negative_infinity(field):
    with pytest.raises(ValidationError):
        Settings(**{field: float("-inf")})


@pytest.mark.parametrize("field", PRICE_FIELDS)
def test_price_fields_reject_nan(field):
    with pytest.raises(ValidationError):
        Settings(**{field: float("nan")})


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_tie_threshold_rejects_out_of_range_values(value):
    with pytest.raises(ValidationError):
        Settings(model_comparison_tie_threshold=value)


def test_tie_threshold_rejects_infinity():
    with pytest.raises(ValidationError):
        Settings(model_comparison_tie_threshold=float("inf"))


def test_tie_threshold_rejects_nan():
    with pytest.raises(ValidationError):
        Settings(model_comparison_tie_threshold=float("nan"))


def test_tie_threshold_accepts_boundary_values():
    assert Settings(model_comparison_tie_threshold=0.0).model_comparison_tie_threshold == 0.0
    assert Settings(model_comparison_tie_threshold=1.0).model_comparison_tie_threshold == 1.0


@pytest.mark.parametrize("value", [0, -1])
def test_grounded_term_min_length_rejects_non_positive(value):
    with pytest.raises(ValidationError):
        Settings(grounded_term_min_length=value)


def test_grounded_term_min_length_accepts_one():
    assert Settings(grounded_term_min_length=1).grounded_term_min_length == 1
