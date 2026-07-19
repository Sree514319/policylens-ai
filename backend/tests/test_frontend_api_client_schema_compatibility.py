"""Cross-checks that `frontend/streamlit_app/api_client.py`'s response
dataclasses parse the backend's real Pydantic response schemas exactly.

This is the one place a backend test is allowed to import frontend code
(and vice versa, in spirit) -- it exists specifically to prove the two
sides agree on the wire format, which is the frontend's only contract
with the backend (see `api_client.py`'s docstring: the frontend never
imports backend service code in its own production code). The frontend
application code itself still never imports anything from `app.*`.
"""

from app.schemas.answer import AnswerResponse, CitationSchema, ModelResultSchema
from app.schemas.document import DocumentUploadResponse
from app.schemas.evaluation import CompareResponse, ComparisonSchema, ProviderMetricsSchema
from app.schemas.search import SearchResponse, SearchResultItem

from streamlit_app.api_client import (
    AnswerResult,
    CompareResult,
    DocumentUploadResult,
    SearchResult,
)


def test_document_upload_response_is_fully_compatible():
    backend_response = DocumentUploadResponse(
        document_id="a" * 64,
        filename="policy.pdf",
        page_count=3,
        character_count=1200,
        status="processed",
        preview="Some masked preview...",
        chunk_count=4,
        pages_with_text=3,
        indexed_chunk_count=4,
        pii_detected=True,
        pii_entity_count=2,
        pii_categories=["EMAIL", "SSN"],
    )

    payload = backend_response.model_dump(mode="json")
    parsed = DocumentUploadResult.from_dict(payload)

    assert parsed.document_id == backend_response.document_id
    assert parsed.filename == backend_response.filename
    assert parsed.page_count == backend_response.page_count
    assert parsed.character_count == backend_response.character_count
    assert parsed.status == backend_response.status
    assert parsed.preview == backend_response.preview
    assert parsed.chunk_count == backend_response.chunk_count
    assert parsed.pages_with_text == backend_response.pages_with_text
    assert parsed.indexed_chunk_count == backend_response.indexed_chunk_count
    assert parsed.pii_detected == backend_response.pii_detected
    assert parsed.pii_entity_count == backend_response.pii_entity_count
    assert parsed.pii_categories == backend_response.pii_categories


def test_document_upload_response_with_no_pii_is_compatible():
    backend_response = DocumentUploadResponse(
        document_id="b" * 64,
        filename="clean.pdf",
        page_count=1,
        character_count=50,
        status="processed",
        preview="Nothing sensitive here.",
        chunk_count=1,
        pages_with_text=1,
        indexed_chunk_count=1,
        pii_detected=False,
        pii_entity_count=0,
        pii_categories=[],
    )

    parsed = DocumentUploadResult.from_dict(backend_response.model_dump(mode="json"))

    assert parsed.pii_detected is False
    assert parsed.pii_categories == []


def test_search_response_is_fully_compatible():
    backend_response = SearchResponse(
        query="overdraft fee",
        query_was_masked=False,
        result_count=1,
        results=[
            SearchResultItem(
                chunk_id="c1",
                document_id="d1",
                source_filename="policy.pdf",
                page_number=4,
                excerpt="Overdraft fees are $35.",
                relevance_score=0.87,
            )
        ],
    )

    parsed = SearchResult.from_dict(backend_response.model_dump(mode="json"))

    assert parsed.query == backend_response.query
    assert parsed.query_was_masked == backend_response.query_was_masked
    assert parsed.result_count == backend_response.result_count
    assert len(parsed.results) == 1
    assert parsed.results[0].chunk_id == "c1"
    assert parsed.results[0].page_number == 4
    assert parsed.results[0].relevance_score == 0.87


def test_answer_response_is_fully_compatible_including_error_and_null_tokens():
    backend_response = AnswerResponse(
        question="What is the overdraft fee?",
        query_was_masked=True,
        evidence_count=1,
        model_results=[
            ModelResultSchema(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                status="success",
                answer="The fee is $35 [S1].",
                citations=[
                    CitationSchema(
                        source_label="S1",
                        chunk_id="c1",
                        document_id="d1",
                        source_filename="policy.pdf",
                        page_number=4,
                        excerpt="Overdraft fees are $35.",
                        relevance_score=0.87,
                    )
                ],
                latency_ms=842.3,
                input_tokens=512,
                output_tokens=41,
                error=None,
            ),
            ModelResultSchema(
                provider="openai",
                model="gpt-4o-mini",
                status="error",
                answer="",
                citations=[],
                latency_ms=30000.0,
                input_tokens=None,
                output_tokens=None,
                error="The OpenAI API request timed out.",
            ),
        ],
    )

    parsed = AnswerResult.from_dict(backend_response.model_dump(mode="json"))

    assert parsed.question == backend_response.question
    assert parsed.query_was_masked is True
    assert len(parsed.model_results) == 2

    anthropic_result = parsed.model_results[0]
    assert anthropic_result.status == "success"
    assert len(anthropic_result.citations) == 1
    assert anthropic_result.citations[0].source_label == "S1"
    assert anthropic_result.input_tokens == 512

    openai_result = parsed.model_results[1]
    assert openai_result.status == "error"
    assert openai_result.error == "The OpenAI API request timed out."
    assert openai_result.input_tokens is None
    assert openai_result.output_tokens is None


def test_compare_response_is_fully_compatible_including_null_metrics():
    backend_response = CompareResponse(
        question="What is the overdraft fee?",
        query_was_masked=False,
        evidence_count=1,
        model_results=[
            ModelResultSchema(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                status="success",
                answer="The fee is $35 [S1].",
                citations=[],
                latency_ms=842.3,
                input_tokens=512,
                output_tokens=41,
                error=None,
            ),
            ModelResultSchema(
                provider="openai",
                model="gpt-4o-mini",
                status="success",
                answer="It costs $35 [S1].",
                citations=[],
                latency_ms=962.8,
                input_tokens=480,
                output_tokens=38,
                error=None,
            ),
        ],
        provider_metrics=[
            ProviderMetricsSchema(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                status="success",
                latency_ms=842.3,
                input_tokens=512,
                output_tokens=41,
                estimated_cost_usd=0.002181,
                valid_citation_count=1,
                citation_coverage=0.5,
                mean_citation_relevance=0.87,
                grounded_term_ratio=0.8,
                answer_length=8,
                evaluation_notes=["a note"],
            ),
            ProviderMetricsSchema(
                provider="openai",
                model="gpt-4o-mini",
                status="success",
                latency_ms=962.8,
                input_tokens=480,
                output_tokens=38,
                estimated_cost_usd=None,
                valid_citation_count=1,
                citation_coverage=0.5,
                mean_citation_relevance=0.87,
                grounded_term_ratio=None,
                answer_length=7,
                evaluation_notes=[],
            ),
        ],
        comparison=ComparisonSchema(
            answer_agreement_score=0.94,
            latency_difference_ms=-120.5,
            estimated_cost_difference_usd=None,
            comparison_status="both_successful",
            comparison_notes=["a comparison note"],
        ),
    )

    parsed = CompareResult.from_dict(backend_response.model_dump(mode="json"))

    assert len(parsed.provider_metrics) == 2
    assert parsed.provider_metrics[0].estimated_cost_usd == 0.002181
    assert parsed.provider_metrics[1].estimated_cost_usd is None
    assert parsed.provider_metrics[1].grounded_term_ratio is None
    assert parsed.comparison is not None
    assert parsed.comparison.comparison_status == "both_successful"
    assert parsed.comparison.answer_agreement_score == 0.94
    assert parsed.comparison.estimated_cost_difference_usd is None


def test_compare_response_with_neither_succeeded_and_null_agreement_score():
    backend_response = CompareResponse(
        question="Anything?",
        query_was_masked=False,
        evidence_count=0,
        model_results=[
            ModelResultSchema(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                status="insufficient_evidence",
                answer="The available evidence does not contain enough information to answer this question.",
                citations=[],
                latency_ms=0.0,
                input_tokens=None,
                output_tokens=None,
                error=None,
            ),
            ModelResultSchema(
                provider="openai",
                model="gpt-4o-mini",
                status="insufficient_evidence",
                answer="The available evidence does not contain enough information to answer this question.",
                citations=[],
                latency_ms=0.0,
                input_tokens=None,
                output_tokens=None,
                error=None,
            ),
        ],
        provider_metrics=[
            ProviderMetricsSchema(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                status="insufficient_evidence",
                latency_ms=0.0,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=None,
                valid_citation_count=0,
                citation_coverage=0.0,
                mean_citation_relevance=None,
                grounded_term_ratio=None,
                answer_length=0,
                evaluation_notes=[],
            ),
            ProviderMetricsSchema(
                provider="openai",
                model="gpt-4o-mini",
                status="insufficient_evidence",
                latency_ms=0.0,
                input_tokens=None,
                output_tokens=None,
                estimated_cost_usd=None,
                valid_citation_count=0,
                citation_coverage=0.0,
                mean_citation_relevance=None,
                grounded_term_ratio=None,
                answer_length=0,
                evaluation_notes=[],
            ),
        ],
        comparison=ComparisonSchema(
            answer_agreement_score=None,
            latency_difference_ms=0.0,
            estimated_cost_difference_usd=None,
            comparison_status="neither_succeeded",
            comparison_notes=["neither model succeeded"],
        ),
    )

    parsed = CompareResult.from_dict(backend_response.model_dump(mode="json"))

    assert parsed.comparison.answer_agreement_score is None
    assert parsed.comparison.comparison_status == "neither_succeeded"


# --- Frontend constants/limits must match the backend's real constraints -------------


def test_frontend_provider_labels_map_to_exactly_the_backend_allowed_providers():
    from pathlib import Path

    from app.schemas.answer import ALLOWED_PROVIDERS

    frontend_root = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app"
    ask_models_source = (frontend_root / "pages" / "ask_models.py").read_text(encoding="utf-8")

    for provider in ALLOWED_PROVIDERS:
        assert f'"{provider}"' in ask_models_source, (
            f"backend provider '{provider}' is not referenced in ask_models.py's provider_map"
        )


def test_frontend_top_k_slider_bounds_do_not_exceed_the_backend_max_top_k():
    from pathlib import Path

    from app.schemas.search import MAX_TOP_K

    frontend_root = Path(__file__).resolve().parents[2] / "frontend" / "streamlit_app" / "pages"
    for page_name in ("semantic_search.py", "ask_models.py", "compare_models.py"):
        source = (frontend_root / page_name).read_text(encoding="utf-8")
        # Every top_k slider in this app must cap at (never above) the
        # backend's MAX_TOP_K -- a frontend slider allowing a value the
        # backend would reject with a 422 is a broken control, not just
        # a cosmetic mismatch.
        assert f"max_value=50" in source, f"{page_name}: expected a top_k slider capped at {MAX_TOP_K}"
        assert MAX_TOP_K == 50, "MAX_TOP_K changed in the backend -- update the frontend sliders to match"


# --- Forward compatibility: an unknown extra field must never break parsing -----------


def test_unknown_extra_fields_do_not_break_document_upload_parsing():
    backend_response = DocumentUploadResponse(
        document_id="c" * 64,
        filename="policy.pdf",
        page_count=1,
        character_count=10,
        status="processed",
        preview="preview",
        chunk_count=1,
        pages_with_text=1,
        indexed_chunk_count=1,
        pii_detected=False,
        pii_entity_count=0,
        pii_categories=[],
    )
    payload = backend_response.model_dump(mode="json")
    payload["a_future_field_the_frontend_does_not_know_about"] = "some new value"

    parsed = DocumentUploadResult.from_dict(payload)

    assert parsed.document_id == backend_response.document_id
    assert not hasattr(parsed, "a_future_field_the_frontend_does_not_know_about")


def test_unknown_extra_fields_do_not_break_compare_response_parsing():
    backend_response = CompareResponse(
        question="fee?",
        query_was_masked=False,
        evidence_count=0,
        model_results=[],
        provider_metrics=[],
        comparison=ComparisonSchema(
            answer_agreement_score=None,
            latency_difference_ms=0.0,
            estimated_cost_difference_usd=None,
            comparison_status="neither_succeeded",
            comparison_notes=[],
        ),
    )
    payload = backend_response.model_dump(mode="json")
    payload["comparison"]["a_future_metric_field"] = 1.23
    payload["a_future_top_level_field"] = {"nested": "value"}

    parsed = CompareResult.from_dict(payload)

    assert parsed.comparison.comparison_status == "neither_succeeded"
