"""Tests for numeric bounds on the Phase 5 LLM/RAG settings.

A misconfigured `.env` should fail fast at Settings-construction time
(Pydantic validation) rather than silently producing a degenerate value
(a zero/negative timeout, an unbounded retry count, a context budget too
small to hold any evidence, or one so large it defeats the cap).
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_settings_are_within_bounds():
    settings = Settings()

    assert settings.llm_timeout_seconds > 0
    assert settings.llm_max_output_tokens >= 1
    assert settings.llm_max_retries >= 0
    assert settings.max_rag_context_characters >= 100


@pytest.mark.parametrize("value", [0, -1, -30.0])
def test_llm_timeout_seconds_rejects_non_positive(value):
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=value)


def test_llm_timeout_seconds_rejects_too_large():
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=301)


@pytest.mark.parametrize("value", [0, -1])
def test_llm_max_output_tokens_rejects_non_positive(value):
    with pytest.raises(ValidationError):
        Settings(llm_max_output_tokens=value)


def test_llm_max_output_tokens_rejects_too_large():
    with pytest.raises(ValidationError):
        Settings(llm_max_output_tokens=8193)


def test_llm_max_retries_rejects_negative():
    with pytest.raises(ValidationError):
        Settings(llm_max_retries=-1)


def test_llm_max_retries_rejects_too_large():
    with pytest.raises(ValidationError):
        Settings(llm_max_retries=11)


@pytest.mark.parametrize("value", [0, 99])
def test_max_rag_context_characters_rejects_too_small(value):
    with pytest.raises(ValidationError):
        Settings(max_rag_context_characters=value)


def test_max_rag_context_characters_rejects_too_large():
    with pytest.raises(ValidationError):
        Settings(max_rag_context_characters=100_001)


def test_max_rag_context_characters_accepts_lower_and_upper_bound():
    assert Settings(max_rag_context_characters=100).max_rag_context_characters == 100
    assert Settings(max_rag_context_characters=100_000).max_rag_context_characters == 100_000
