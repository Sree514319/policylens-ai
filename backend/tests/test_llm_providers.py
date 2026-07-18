"""Tests for the provider-neutral LLM abstraction.

`AnthropicProvider`/`OpenAIProvider` are exercised with their real SDK
client objects present, but every network-making method
(`messages.create` / `chat.completions.create`) is monkeypatched with an
`AsyncMock` -- no live API calls, no API keys, no network access.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx
import openai
import pytest

from app.core.config import get_settings
from app.services.llm.providers import (
    AnthropicProvider,
    FakeLLMProvider,
    OpenAIProvider,
    TokenUsage,
    get_llm_provider_registry,
)


def _http_request():
    return httpx.Request("POST", "https://api.example.invalid/v1/completions")


def _http_response(status_code):
    return httpx.Response(status_code=status_code, request=_http_request())


def _anthropic_message(text="The overdraft fee is $35.", input_tokens=12, output_tokens=8):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _openai_completion(text="The overdraft fee is $35.", prompt_tokens=12, completion_tokens=8):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _anthropic_provider(api_key="sk-ant-test-key"):
    return AnthropicProvider(
        api_key=api_key, model="claude-test-model", timeout_seconds=5.0, max_output_tokens=256, max_retries=0
    )


def _openai_provider(api_key="sk-openai-test-key"):
    return OpenAIProvider(
        api_key=api_key, model="gpt-test-model", timeout_seconds=5.0, max_output_tokens=256, max_retries=0
    )


# --- AnthropicProvider --------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_missing_api_key_returns_safe_error_without_network_call(monkeypatch):
    provider = _anthropic_provider(api_key=None)
    spy = AsyncMock()
    monkeypatch.setattr(provider._client.messages, "create", spy)

    response = await provider.generate("system", "user")

    assert response.error == "Anthropic API key is not configured."
    assert response.text is None
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_provider_success(monkeypatch):
    provider = _anthropic_provider()
    monkeypatch.setattr(
        provider._client.messages, "create", AsyncMock(return_value=_anthropic_message(input_tokens=12, output_tokens=8))
    )

    response = await provider.generate("system", "user")

    assert response.error is None
    assert response.text == "The overdraft fee is $35."
    assert response.usage == TokenUsage(input_tokens=12, output_tokens=8)
    assert response.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_anthropic_provider_authentication_error_is_safe(monkeypatch):
    provider = _anthropic_provider()
    exc = anthropic.AuthenticationError("invalid x-api-key: sk-ant-super-secret", response=_http_response(401), body=None)
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.text is None
    assert response.error == "Authentication with the Anthropic API failed."
    assert "sk-ant-super-secret" not in response.error


@pytest.mark.asyncio
async def test_anthropic_provider_rate_limit_error_is_safe(monkeypatch):
    provider = _anthropic_provider()
    exc = anthropic.RateLimitError("rate limited", response=_http_response(429), body=None)
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.error == "The Anthropic API rate limit was exceeded."


@pytest.mark.asyncio
async def test_anthropic_provider_timeout_is_safe(monkeypatch):
    provider = _anthropic_provider()
    monkeypatch.setattr(
        provider._client.messages, "create", AsyncMock(side_effect=anthropic.APITimeoutError(request=_http_request()))
    )

    response = await provider.generate("system", "user")

    assert response.error == "The Anthropic API request timed out."


@pytest.mark.asyncio
async def test_anthropic_provider_connection_error_is_safe(monkeypatch):
    provider = _anthropic_provider()
    monkeypatch.setattr(
        provider._client.messages,
        "create",
        AsyncMock(side_effect=anthropic.APIConnectionError(request=_http_request())),
    )

    response = await provider.generate("system", "user")

    assert response.error == "Could not connect to the Anthropic API."


@pytest.mark.asyncio
async def test_anthropic_provider_generic_status_error_is_safe(monkeypatch):
    provider = _anthropic_provider()
    exc = anthropic.APIStatusError("internal server error, trace=abc123", response=_http_response(500), body=None)
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.error == "The Anthropic API returned an error."
    assert "trace=abc123" not in response.error


@pytest.mark.asyncio
async def test_anthropic_provider_empty_response_is_an_error(monkeypatch):
    provider = _anthropic_provider()
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(return_value=_anthropic_message(text="")))

    response = await provider.generate("system", "user")

    assert response.error == "The Anthropic API returned an empty response."


@pytest.mark.asyncio
async def test_anthropic_provider_unexpected_exception_does_not_leak_internal_message(monkeypatch):
    provider = _anthropic_provider()
    monkeypatch.setattr(
        provider._client.messages, "create", AsyncMock(side_effect=RuntimeError("/internal/secret/path leaked"))
    )

    response = await provider.generate("system", "user")

    assert response.error == "The Anthropic API request failed unexpectedly."
    assert "/internal/secret/path" not in response.error


# --- OpenAIProvider ------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_provider_missing_api_key_returns_safe_error_without_network_call(monkeypatch):
    provider = _openai_provider(api_key=None)
    spy = AsyncMock()
    monkeypatch.setattr(provider._client.chat.completions, "create", spy)

    response = await provider.generate("system", "user")

    assert response.error == "OpenAI API key is not configured."
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_openai_provider_success(monkeypatch):
    provider = _openai_provider()
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        AsyncMock(return_value=_openai_completion(prompt_tokens=12, completion_tokens=8)),
    )

    response = await provider.generate("system", "user")

    assert response.error is None
    assert response.text == "The overdraft fee is $35."
    assert response.usage == TokenUsage(input_tokens=12, output_tokens=8)


@pytest.mark.asyncio
async def test_openai_provider_authentication_error_is_safe(monkeypatch):
    provider = _openai_provider()
    exc = openai.AuthenticationError("invalid api key: sk-openai-super-secret", response=_http_response(401), body=None)
    monkeypatch.setattr(provider._client.chat.completions, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.error == "Authentication with the OpenAI API failed."
    assert "sk-openai-super-secret" not in response.error


@pytest.mark.asyncio
async def test_openai_provider_rate_limit_error_is_safe(monkeypatch):
    provider = _openai_provider()
    exc = openai.RateLimitError("rate limited", response=_http_response(429), body=None)
    monkeypatch.setattr(provider._client.chat.completions, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.error == "The OpenAI API rate limit was exceeded."


@pytest.mark.asyncio
async def test_openai_provider_timeout_is_safe(monkeypatch):
    provider = _openai_provider()
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        AsyncMock(side_effect=openai.APITimeoutError(request=_http_request())),
    )

    response = await provider.generate("system", "user")

    assert response.error == "The OpenAI API request timed out."


@pytest.mark.asyncio
async def test_openai_provider_connection_error_is_safe(monkeypatch):
    provider = _openai_provider()
    monkeypatch.setattr(
        provider._client.chat.completions,
        "create",
        AsyncMock(side_effect=openai.APIConnectionError(request=_http_request())),
    )

    response = await provider.generate("system", "user")

    assert response.error == "Could not connect to the OpenAI API."


@pytest.mark.asyncio
async def test_openai_provider_generic_status_error_is_safe(monkeypatch):
    provider = _openai_provider()
    exc = openai.APIStatusError("server error, trace=xyz789", response=_http_response(500), body=None)
    monkeypatch.setattr(provider._client.chat.completions, "create", AsyncMock(side_effect=exc))

    response = await provider.generate("system", "user")

    assert response.error == "The OpenAI API returned an error."
    assert "trace=xyz789" not in response.error


@pytest.mark.asyncio
async def test_openai_provider_empty_response_is_an_error(monkeypatch):
    provider = _openai_provider()
    monkeypatch.setattr(
        provider._client.chat.completions, "create", AsyncMock(return_value=_openai_completion(text=""))
    )

    response = await provider.generate("system", "user")

    assert response.error == "The OpenAI API returned an empty response."


@pytest.mark.asyncio
async def test_openai_provider_unexpected_exception_does_not_leak_internal_message(monkeypatch):
    provider = _openai_provider()
    monkeypatch.setattr(
        provider._client.chat.completions, "create", AsyncMock(side_effect=RuntimeError("/internal/secret/path leaked"))
    )

    response = await provider.generate("system", "user")

    assert response.error == "The OpenAI API request failed unexpectedly."
    assert "/internal/secret/path" not in response.error


# --- FakeLLMProvider (sanity) ---------------------------------------------------


@pytest.mark.asyncio
async def test_fake_llm_provider_returns_canned_json():
    provider = FakeLLMProvider(response_json={"insufficient_evidence": False, "answer": "Hi", "citations": ["S1"]})

    response = await provider.generate("system", "user")

    assert response.error is None
    assert response.text == '{"insufficient_evidence": false, "answer": "Hi", "citations": ["S1"]}'


@pytest.mark.asyncio
async def test_fake_llm_provider_can_simulate_an_error():
    provider = FakeLLMProvider(error="simulated failure")

    response = await provider.generate("system", "user")

    assert response.error == "simulated failure"
    assert response.text is None


@pytest.mark.asyncio
async def test_fake_llm_provider_can_raise_for_defense_in_depth_testing():
    provider = FakeLLMProvider(raise_exception=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await provider.generate("system", "user")


# --- get_llm_provider_registry() safety -----------------------------------------


@pytest.mark.asyncio
async def test_get_llm_provider_registry_never_makes_a_live_call_with_no_keys_configured(monkeypatch):
    """Mirrors `test_get_vector_store_wraps_provider_construction_failure`'s
    direct-construction pattern: this is the REAL, non-overridden registry
    that FastAPI would use in production if a test ever forgot to override
    `get_llm_provider_registry` via `app.dependency_overrides`. In the test
    environment (no .env, no API keys), it must still be fully safe to
    construct and call -- proving that even an accidentally-unmocked
    dependency can never reach the network here.
    """

    get_settings.cache_clear()
    get_llm_provider_registry.cache_clear()
    try:
        registry = get_llm_provider_registry()

        assert isinstance(registry["anthropic"], AnthropicProvider)
        assert isinstance(registry["openai"], OpenAIProvider)

        # No API keys are configured in the test environment, so calling
        # generate() must short-circuit on the missing-key check -- never
        # attempt a network call, regardless of ALLOW_EXTERNAL_LLM_CALLS.
        anthropic_response = await registry["anthropic"].generate("system", "user")
        openai_response = await registry["openai"].generate("system", "user")

        assert anthropic_response.error == "Anthropic API key is not configured."
        assert openai_response.error == "OpenAI API key is not configured."
    finally:
        get_settings.cache_clear()
        get_llm_provider_registry.cache_clear()
