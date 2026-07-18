"""Tests for POST /api/v1/compare."""

import pytest

from app.core.config import Settings, get_settings
from app.main import app
from app.services.llm.providers import FakeLLMProvider, get_llm_provider_registry

UPLOAD_URL = "/api/v1/documents/upload"
COMPARE_URL = "/api/v1/compare"


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
            input_tokens=100,
            output_tokens=20,
        ),
        "openai": FakeLLMProvider(
            name="openai",
            model="fake-openai-model",
            response_json={"insufficient_evidence": False, "answer": "It costs $35 [S1].", "citations": ["S1"]},
            input_tokens=90,
            output_tokens=18,
        ),
    }


# --- Success path / exact key sets --------------------------------------------------


def test_both_providers_succeed_end_to_end(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_count"] >= 1
    assert len(body["model_results"]) == 2
    assert len(body["provider_metrics"]) == 2
    assert body["comparison"]["comparison_status"] == "both_successful"


def test_compare_response_exact_key_set(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})
    body = response.json()

    assert set(body.keys()) == {
        "question",
        "query_was_masked",
        "evidence_count",
        "model_results",
        "provider_metrics",
        "comparison",
    }
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
    for metrics in body["provider_metrics"]:
        assert set(metrics.keys()) == {
            "provider",
            "model",
            "status",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "estimated_cost_usd",
            "valid_citation_count",
            "citation_coverage",
            "mean_citation_relevance",
            "grounded_term_ratio",
            "answer_length",
            "evaluation_notes",
        }
    assert set(body["comparison"].keys()) == {
        "answer_agreement_score",
        "latency_difference_ms",
        "estimated_cost_difference_usd",
        "comparison_status",
        "comparison_notes",
    }


def test_provider_metrics_are_returned_in_anthropic_then_openai_order(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})
    body = response.json()

    assert [m["provider"] for m in body["provider_metrics"]] == ["anthropic", "openai"]


# --- Compare calls RAG orchestration exactly once (no duplicate provider calls) -----


def test_compare_calls_each_provider_exactly_once(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()

    call_counts = {"anthropic": 0, "openai": 0}

    class _CountingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            call_counts[self.name] += 1
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CountingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": _CountingProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    assert call_counts == {"anthropic": 1, "openai": 1}


# --- Exactly-two-provider validation --------------------------------------------------


def test_single_provider_is_rejected(client):
    response = client.post(COMPARE_URL, json={"question": "Anything?", "providers": ["anthropic"]})
    assert response.status_code == 422


def test_unknown_provider_is_rejected(client):
    response = client.post(COMPARE_URL, json={"question": "Anything?", "providers": ["anthropic", "made-up"]})
    assert response.status_code == 422


def test_empty_providers_list_is_rejected(client):
    response = client.post(COMPARE_URL, json={"question": "Anything?", "providers": []})
    assert response.status_code == 422


def test_providers_in_either_order_are_accepted_and_normalized(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        COMPARE_URL, json={"question": "What page is 'Hello World' on?", "providers": ["openai", "anthropic"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert {r["provider"] for r in body["model_results"]} == {"anthropic", "openai"}


def test_duplicate_providers_still_resolve_to_exactly_two(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        COMPARE_URL,
        json={"question": "What page is 'Hello World' on?", "providers": ["anthropic", "anthropic", "openai"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["model_results"]) == 2


def test_duplicate_providers_in_the_request_do_not_cause_duplicate_calls(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()

    call_counts = {"anthropic": 0, "openai": 0}

    class _CountingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            call_counts[self.name] += 1
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CountingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": _CountingProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    response = client.post(
        COMPARE_URL,
        json={"question": "What page is 'Hello World' on?", "providers": ["anthropic", "anthropic", "openai"]},
    )

    assert response.status_code == 200
    assert call_counts == {"anthropic": 1, "openai": 1}


def test_no_providers_field_defaults_to_both(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    assert len(response.json()["model_results"]) == 2


# --- External calls blocked by default; no-evidence behavior -------------------------


def test_external_calls_blocked_by_default_yields_neither_succeeded(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(COMPARE_URL, json={"question": "Hello World is on which page?"})

    assert response.status_code == 200
    body = response.json()
    for result in body["model_results"]:
        assert result["status"] == "error"
        assert "ALLOW_EXTERNAL_LLM_CALLS" in result["error"]
    assert body["comparison"]["comparison_status"] == "neither_succeeded"
    assert body["comparison"]["answer_agreement_score"] is None


def test_no_evidence_returns_insufficient_evidence_without_calling_providers(client):
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(raise_exception=AssertionError("must not be called")),
        "openai": FakeLLMProvider(raise_exception=AssertionError("must not be called")),
    }

    response = client.post(COMPARE_URL, json={"question": "Anything at all?"})

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_count"] == 0
    assert all(r["status"] == "insufficient_evidence" for r in body["model_results"])
    assert body["comparison"]["comparison_status"] == "neither_succeeded"


# --- One provider fails ---------------------------------------------------------------


def test_comparison_status_when_only_one_provider_succeeds(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": FakeLLMProvider(name="openai", error="The OpenAI API rate limit was exceeded."),
    }

    response = client.post(COMPARE_URL, json={"question": "What page is 'Hello World' on?"})

    assert response.status_code == 200
    body = response.json()
    assert body["comparison"]["comparison_status"] == "anthropic_succeeded_openai_did_not"
    assert body["comparison"]["answer_agreement_score"] is None


# --- Question / top_k / document_id validation (mirrors /answers) --------------------


@pytest.mark.parametrize("question", ["", "   ", "\t\n "])
def test_empty_or_whitespace_question_is_rejected(client, question):
    response = client.post(COMPARE_URL, json={"question": question})
    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_top_k_out_of_bounds_is_rejected(client, top_k):
    response = client.post(COMPARE_URL, json={"question": "Anything?", "top_k": top_k})
    assert response.status_code == 422


def test_unknown_document_id_returns_404(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "Anything?", "document_id": "0" * 64})

    assert response.status_code == 404


def test_document_scoped_compare_via_api(client, valid_pdf_bytes):
    from tests.conftest import _build_pdf

    other_pdf = _build_pdf(["Completely unrelated content about routing numbers."])
    first = _upload(client, valid_pdf_bytes, filename="first.pdf")
    _upload(client, other_pdf, filename="second.pdf")

    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        COMPARE_URL,
        json={"question": "What page is 'Hello World' on?", "document_id": first["document_id"]},
    )

    assert response.status_code == 200
    body = response.json()
    for result in body["model_results"]:
        for citation in result["citations"]:
            assert citation["document_id"] == first["document_id"]


# --- PII masking --------------------------------------------------------------------


def test_question_containing_pii_is_masked_before_retrieval_and_response(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(COMPARE_URL, json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?"})

    assert response.status_code == 200
    body = response.json()
    assert "123-45-6789" not in body["question"]
    assert "[SSN_REDACTED]" in body["question"]
    assert body["query_was_masked"] is True


def test_original_question_pii_never_appears_anywhere_in_the_response(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    response = client.post(
        COMPARE_URL, json={"question": "Email jane.doe@example.com -- what page is 'Hello World' on?"}
    )

    assert "jane.doe@example.com" not in response.text
    assert "[EMAIL_REDACTED]" in response.text


def test_masked_question_is_what_each_provider_actually_receives(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()

    captured = {}

    class _CapturingProvider(FakeLLMProvider):
        async def generate(self, system_prompt, user_prompt):
            captured[self.name] = user_prompt
            return await super().generate(system_prompt, user_prompt)

    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": _CapturingProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": _CapturingProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    client.post(COMPARE_URL, json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?"})

    for prompt in captured.values():
        assert "123-45-6789" not in prompt
        assert "[SSN_REDACTED]" in prompt


def test_no_pii_anywhere_in_logs(client, valid_pdf_bytes, caplog):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = _success_providers

    with caplog.at_level("INFO"):
        client.post(COMPARE_URL, json={"question": "For SSN 123-45-6789, what page is 'Hello World' on?"})

    assert "123-45-6789" not in caplog.text


# --- Privacy: nothing sensitive in the error path -------------------------------------


def test_error_result_never_contains_question_or_evidence_text(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)
    _enable_external_calls()
    app.dependency_overrides[get_llm_provider_registry] = lambda: {
        "anthropic": FakeLLMProvider(name="anthropic", error="The Anthropic API rate limit was exceeded."),
        "openai": FakeLLMProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    secret_question = "What is the super-secret-marker-12345 policy?"
    response = client.post(COMPARE_URL, json={"question": secret_question})

    body = response.json()
    anthropic_result = next(r for r in body["model_results"] if r["provider"] == "anthropic")
    assert anthropic_result["error"] == "The Anthropic API rate limit was exceeded."
    assert "super-secret-marker-12345" not in anthropic_result["error"]
    assert "super-secret-marker-12345" not in str(body["comparison"])
