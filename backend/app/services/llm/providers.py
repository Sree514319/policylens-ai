"""Provider-neutral LLM abstraction.

`LLMProvider` is a minimal, provider-agnostic interface: "send this system
+ user prompt, get text back." It knows nothing about RAG, evidence,
citations, or JSON schemas -- that logic lives in `rag.py`. No
provider-specific SDK object (an Anthropic `Message`, an OpenAI
`ChatCompletion`, an SDK exception) ever leaves this module; every call
site outside this file only ever sees `ProviderResponse`.

Three implementations:

- `AnthropicProvider` / `OpenAIProvider` (production) -- thin wrappers
  around the official async SDK clients. Every SDK exception is caught
  and translated into a safe, generic `ProviderResponse.error` string;
  raw exception messages (which can include request/response bodies)
  never escape this module.
- `FakeLLMProvider` (tests only) -- deterministic, network-free, fully
  configurable (canned JSON/raw text, a canned error, a simulated delay,
  or a raised exception for defense-in-depth testing).
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import anthropic
import openai

from app.core.config import Settings, get_settings


@dataclass
class TokenUsage:
    """Token counts for one LLM call, when the provider reports them."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]


@dataclass
class ProviderResponse:
    """The generic result of one LLM call.

    `text` is the raw model output (expected to be a JSON string, but this
    layer does not know or care about that -- parsing is `rag.py`'s job).
    `error` is `None` on success; when set, it is already a short,
    client-safe message with no stack trace, credentials, or raw SDK
    payload embedded in it.
    """

    text: Optional[str]
    usage: Optional[TokenUsage]
    latency_ms: float
    error: Optional[str]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class LLMProvider(ABC):
    """A single LLM backend: "send a system+user prompt, get text back."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable provider identifier, e.g. "anthropic"."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The configured model name this provider calls."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        """Run one completion. Never raises -- failures come back as `ProviderResponse.error`."""


class AnthropicProvider(LLMProvider):
    """Wraps `anthropic.AsyncAnthropic` (Messages API)."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        max_retries: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or "not-configured",
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        started = time.perf_counter()

        if not self._api_key:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="Anthropic API key is not configured."
            )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.AuthenticationError:
            return ProviderResponse(
                text=None,
                usage=None,
                latency_ms=_elapsed_ms(started),
                error="Authentication with the Anthropic API failed.",
            )
        except anthropic.RateLimitError:
            return ProviderResponse(
                text=None,
                usage=None,
                latency_ms=_elapsed_ms(started),
                error="The Anthropic API rate limit was exceeded.",
            )
        except anthropic.APITimeoutError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The Anthropic API request timed out."
            )
        except anthropic.APIConnectionError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="Could not connect to the Anthropic API."
            )
        except anthropic.APIStatusError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The Anthropic API returned an error."
            )
        except Exception:
            return ProviderResponse(
                text=None,
                usage=None,
                latency_ms=_elapsed_ms(started),
                error="The Anthropic API request failed unexpectedly.",
            )

        text = "".join(
            block.text for block in (response.content or []) if getattr(block, "type", None) == "text"
        ).strip()
        usage = TokenUsage(
            input_tokens=getattr(response.usage, "input_tokens", None) if response.usage else None,
            output_tokens=getattr(response.usage, "output_tokens", None) if response.usage else None,
        )

        if not text:
            return ProviderResponse(
                text=None,
                usage=usage,
                latency_ms=_elapsed_ms(started),
                error="The Anthropic API returned an empty response.",
            )

        return ProviderResponse(text=text, usage=usage, latency_ms=_elapsed_ms(started), error=None)


class OpenAIProvider(LLMProvider):
    """Wraps `openai.AsyncOpenAI` (Chat Completions API)."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        max_retries: int,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "not-configured",
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        started = time.perf_counter()

        if not self._api_key:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="OpenAI API key is not configured."
            )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_output_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.AuthenticationError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="Authentication with the OpenAI API failed."
            )
        except openai.RateLimitError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The OpenAI API rate limit was exceeded."
            )
        except openai.APITimeoutError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The OpenAI API request timed out."
            )
        except openai.APIConnectionError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="Could not connect to the OpenAI API."
            )
        except openai.APIStatusError:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The OpenAI API returned an error."
            )
        except Exception:
            return ProviderResponse(
                text=None, usage=None, latency_ms=_elapsed_ms(started), error="The OpenAI API request failed unexpectedly."
            )

        choice = (response.choices or [None])[0]
        text = (choice.message.content or "").strip() if choice and choice.message else ""
        usage = TokenUsage(
            input_tokens=getattr(response.usage, "prompt_tokens", None) if response.usage else None,
            output_tokens=getattr(response.usage, "completion_tokens", None) if response.usage else None,
        )

        if not text:
            return ProviderResponse(
                text=None, usage=usage, latency_ms=_elapsed_ms(started), error="The OpenAI API returned an empty response."
            )

        return ProviderResponse(text=text, usage=usage, latency_ms=_elapsed_ms(started), error=None)


class FakeLLMProvider(LLMProvider):
    """Deterministic, network-free provider for tests only.

    Exactly one of `response_json`, `raw_text`, `error`, or
    `raise_exception` should be meaningfully set to control behavior;
    `response_json` is the common case (a dict serialized to JSON text).
    """

    def __init__(
        self,
        name: str = "fake",
        model: str = "fake-model",
        response_json: Optional[dict] = None,
        raw_text: Optional[str] = None,
        error: Optional[str] = None,
        input_tokens: Optional[int] = 10,
        output_tokens: Optional[int] = 10,
        latency_ms: float = 5.0,
        delay_seconds: float = 0.0,
        raise_exception: Optional[BaseException] = None,
    ) -> None:
        self._name = name
        self._model = model
        self._response_json = response_json
        self._raw_text = raw_text
        self._error = error
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._latency_ms = latency_ms
        self._delay_seconds = delay_seconds
        self._raise_exception = raise_exception

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)

        if self._raise_exception is not None:
            raise self._raise_exception

        if self._error is not None:
            return ProviderResponse(text=None, usage=None, latency_ms=self._latency_ms, error=self._error)

        if self._raw_text is not None:
            text = self._raw_text
        else:
            text = json.dumps(self._response_json if self._response_json is not None else {})

        usage = TokenUsage(input_tokens=self._input_tokens, output_tokens=self._output_tokens)
        return ProviderResponse(text=text, usage=usage, latency_ms=self._latency_ms, error=None)


@lru_cache
def get_llm_provider_registry() -> Dict[str, LLMProvider]:
    """Process-wide singleton provider registry, wired from Settings.

    Tests override this dependency (via `app.dependency_overrides`) with
    `FakeLLMProvider` instances, so real SDK clients are never constructed
    or called during the test suite.
    """

    settings: Settings = get_settings()
    return {
        "anthropic": AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_max_retries,
        ),
        "openai": OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_max_retries,
        ),
    }
