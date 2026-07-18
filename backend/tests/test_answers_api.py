"""Tests for POST /api/v1/answers."""

from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from app.core.config import Settings, get_settings
from app.main import app
from app.services.llm.providers import AnthropicProvider, FakeLLMProvider, get_llm_provider_registry

UPLOAD_URL = "/api/v1/documents/upload"
ANSWERS_URL = "/api/v1/answers"


def _upload(client, pdf_bytes, filename="policy.pdf"):
    response = client.post(UPLOAD_URL, files={"file": (filename, pdf_bytes, "application/pdf")})
    assert response.status_code == 201
    return response.json()


def _enable_external_calls():
    app.dependency_overrides[get_settings] = lambda: Settings(allow_external_llm_calls=True)


def _success_providers():
    return {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            model="fake-anthropic-model",
            response_json={"insufficient_evidence": False, "answer": "The fee is $35 [S1].", "citations": ["S1"]},
        ),
        "openai": FakeLLMProvider(
            name="openai",
            model="fake-openai-model",
            response_json={"insufficient_evidence": False, "answer": "It costs $35 [S1].", "citations": ["S1"]},
        ),
    }


# --- External calls blocked by default -------------------------------------------


def test_external_calls_are_blocked_by_default(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(ANSWERS_URL, json={"question": "Hello World is on which page?"})

    assert response.status_code == 200
    body = response.json()
    for result in body["model_results"]:
        assert result["status"] == "error"
        assert "ALLOW_EXTERNAL_LLM_CALLS" in result["error"]


# --- Success path / exact key sets -------------------------------------------------


def test_both_providers_succeed_end_to_end(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(ANSWERS_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_count"] >= 1
    assert len(body["model_results"]) == 2
    assert {r["provider"] for r in body["model_results"]} == {"anthropic", "openai"}
    assert all(r["status"] == "success" for r in body["model_results"])


def test_answer_response_exact_key_set(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(ANSWERS_URL, json={"question": "What page is 'Hello World' on?"})
    body = response.json()

    assert set(body.keys()) == {"question", "query_was_masked", "evidence_count", "model_results"}
    for result in body["model_results"]:
        assert set(result.keys()) == {
            "provider",
            "model",
            "status",
            "answer",
            "citations",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "error",
        }
        for citation in result["citations"]:
            assert set(citation.keys()) == {
                "source_label",
                "chunk_id",
                "document_id",
                "source_filename",
                "page_number",
                "excerpt",
                "relevance_score",
            }


def test_successful_result_has_no_error_and_error_result_has_no_citations(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": FakeLLMProvider(name="openai", error="The OpenAI API rate limit was exceeded."),
    }

    response = client.post(ANSWERS_URL, json={"question": "What page is 'Hello World' on?"})
    body = response.json()

    by_provider = {r["provider"]: r for r in body["model_results"]}
    assert by_provider["anthropic"]["error"] is None
    assert len(by_provider["anthropic"]["citations"]) == 1
    assert by_provider["openai"]["status"] == "error"
    assert by_provider["openai"]["citations"] == []
    assert by_provider["openai"]["answer"] == ""


# --- Providers field validation ----------------------------------------------------


def test_unknown_provider_name_is_rejected(client):
    response = client.post(ANSWERS_URL, json={"question": "Anything?", "providers": ["anthropic", "made-up"]})

    assert response.status_code == 422


def test_empty_providers_list_is_rejected(client):
    response = client.post(ANSWERS_URL, json={"question": "Anything?", "providers": []})

    assert response.status_code == 422


def test_single_provider_selection_only_queries_that_provider(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL, json={"question": "What page is 'Hello World' on?", "providers": ["anthropic"]}
    )

    body = response.json()
    assert len(body["model_results"]) == 1
    assert body["model_results"][0]["provider"] == "anthropic"


# --- Question validation -----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\t\n "])
def test_empty_or_whitespace_question_is_rejected(client, question):
    response = client.post(ANSWERS_URL, json={"question": question})

    assert response.status_code == 422


# --- top_k validation ---------------------------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_top_k_out_of_bounds_is_rejected(client, top_k):
    response = client.post(ANSWERS_URL, json={"question": "Anything?", "top_k": top_k})

    assert response.status_code == 422


# --- document_id behavior ------------------------------------------------------------


def test_unknown_document_id_returns_404(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL, json={"question": "Anything?", "document_id": "0" * 64}
    )

    assert response.status_code == 404


def test_document_scoped_answer_via_api(client, valid_pdf_bytes):
    from tests.conftest import _build_pdf

    other_pdf = _build_pdf(["Completely unrelated content about routing numbers."])
    first = _upload(client, valid_pdf_bytes, filename="first.pdf")
    _upload(client, other_pdf, filename="second.pdf")

    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL,
        json={"question": "What page is 'Hello World' on?", "document_id": first["document_id"]},
    )

    assert response.status_code == 200
    body = response.json()
    for result in body["model_results"]:
        for citation in result["citations"]:
            assert citation["document_id"] == first["document_id"]


# --- No evidence -> insufficient_evidence without calling providers -----------------


def test_no_evidence_returns_insufficient_evidence_for_every_provider(client):
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(raise_exception=AssertionError("must not be called")),
        "openai": FakeLLMProvider(raise_exception=AssertionError("must not be called")),
    }

    response = client.post(ANSWERS_URL, json={"question": "Anything at all?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_count"] == 0
    assert all(r["status"] == "insufficient_evidence" for r in body["model_results"])


# --- Privacy: nothing sensitive in the error path -----------------------------------


def test_error_result_never_contains_question_or_evidence_text(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(name="anthropic", error="The Anthropic API rate limit was exceeded."),
    }

    secret_question = "What is the super-secret-marker-12345 policy?"
    response = client.post(ANSWERS_URL, json={"question": secret_question, "providers": ["anthropic"]})

    body = response.json()
    assert body["model_results"][0]["error"] == "The Anthropic API rate limit was exceeded."
    assert "super-secret-marker-12345" not in body["model_results"][0]["error"]


def test_a_real_sdk_exception_never_leaks_into_the_api_response_or_logs(client, valid_pdf_bytes, monkeypatch, caplog):
    # End-to-end: a REAL AnthropicProvider (not FakeLLMProvider) with only
    # its network-making SDK method mocked, wired into the actual endpoint
    # via dependency override, so this exercises the full
    # SDK-exception -> provider -> rag -> API-response path.
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()

    provider = AnthropicProvider(
        api_key="sk-ant-should-never-appear-in-output",
        model="claude-test-model",
        timeout_seconds=5.0,
        max_output_tokens=256,
        max_retries=0,
    )
    secret = "Bearer sk-ant-should-never-appear-in-output"
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=401, request=request)
    exc = anthropic.AuthenticationError(f"invalid x-api-key: {secret}", response=response, body=None)
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(side_effect=exc))

    app.dependency_overrides[get_llm_provider_registry] = lambda: {"anthropic": provider}

    with caplog.at_level("INFO"):
        api_response = client.post(ANSWERS_URL, json={"question": "What is the fee?", "providers": ["anthropic"]})

    body = api_response.json()
    assert body["model_results"][0]["status"] == "error"
    assert body["model_results"][0]["error"] == "Authentication with the Anthropic API failed."
    assert "sk-ant-should-never-appear-in-output" not in api_response.text
    assert "sk-ant-should-never-appear-in-output" not in caplog.text


# --- Duplicate / provider list handling ----------------------------------------------


def test_duplicate_requested_providers_are_deduplicated(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL,
        json={"question": "What page is 'Hello World' on?", "providers": ["anthropic", "anthropic", "anthropic"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["model_results"]) == 1
    assert body["model_results"][0]["provider"] == "anthropic"


# --- PII masking in the question -----------------------------------------------------


def test_question_containing_pii_is_masked_before_retrieval_and_response(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL, json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert "123-45-6789" not in body["question"]
    assert "[SSN_REDACTED]" in body["question"]
    assert body["query_was_masked"] is True


def test_masked_question_is_what_the_provider_actually_receives(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()

    captured = {}

    class _CapturingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CapturingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        )
    }

    client.post(
        ANSWERS_URL,
        json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?", "providers": ["anthropic"]},
    )

    assert "123-45-6789" not in captured["user_prompt"]
    assert "[SSN_REDACTED]" in captured["user_prompt"]


def test_original_question_pii_never_appears_anywhere_in_the_response(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL, json={"question": "Email jane.doe@example.com -- what page is 'Hello World' on?"}
    )

    assert "jane.doe@example.com" not in response.text
    assert "[EMAIL_REDACTED]" in response.text


def test_pii_protection_disabled_leaves_question_unmasked(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    app.dependency_overrides[get_settings] = lambda: Settings(
        allow_external_llm_calls=True, pii_protection_enabled=False
    )
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        ANSWERS_URL, json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?"}
    )

    body = response.json()
    assert "123-45-6789" in body["question"]
    assert body["query_was_masked"] is False


def test_pii_protection_disabled_also_sends_the_unmasked_question_to_the_provider(client, valid_pdf_bytes):
    # Documents the two-flag interaction: PII_PROTECTION_ENABLED=false alone
    # does not gate external calls (ALLOW_EXTERNAL_LLM_CALLS does that
    # independently) -- but once an operator has explicitly opted into
    # *both* disabling PII masking and enabling external calls, the raw
    # (unmasked) question really is what reaches the provider prompt. This
    # is the deliberate, documented behavior of that explicit combination,
    # not a silent leak of either flag alone.
    _upload(client, valid_pdf_bytes)
    app.dependency_overrides[get_settings] = lambda: Settings(
        allow_external_llm_calls=True, pii_protection_enabled=False
    )

    captured = {}

    class _CapturingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CapturingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        )
    }

    client.post(
        ANSWERS_URL,
        json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?", "providers": ["anthropic"]},
    )

    assert "123-45-6789" in captured["user_prompt"]


# --- PII masking in the uploaded filename (citations) ---------------------------------


def test_pii_in_uploaded_filename_never_appears_in_answer_citations(client):
    from tests.conftest import _build_pdf

    pdf_bytes = _build_pdf(["Hello World, this is page one of the policy document."])
    upload_response = client.post(
        UPLOAD_URL, files={"file": ("Customer 555-123-4567 statement.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_response.status_code == 201

    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(ANSWERS_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    assert "555-123-4567" not in response.text
    body = response.json()
    citations = [c for r in body["model_results"] for c in r["citations"]]
    assert citations
    assert all("[PHONE_REDACTED]" in c["source_filename"] for c in citations)


def test_pii_in_uploaded_filename_never_reaches_the_provider_prompt(client):
    from tests.conftest import _build_pdf

    pdf_bytes = _build_pdf(["Hello World, this is page one of the policy document."])
    upload_response = client.post(
        UPLOAD_URL, files={"file": ("Customer 555-123-4567 statement.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_response.status_code == 201

    _enable_external_calls()

    captured = {}

    class _CapturingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CapturingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        )
    }

    client.post(ANSWERS_URL, json={"question": "What page is 'Hello World' on?", "providers": ["anthropic"]})

    assert "555-123-4567" not in captured["user_prompt"]
    assert "[PHONE_REDACTED]" in captured["user_prompt"]
