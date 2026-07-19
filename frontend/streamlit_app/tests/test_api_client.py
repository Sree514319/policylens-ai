"""Tests for `api_client.py`. Every HTTP interaction is mocked via
`httpx.MockTransport` (see `conftest.install_mock_transport`) -- this
file never makes a real network call.
"""

import httpx
import pytest

from streamlit_app.api_client import PolicyLensAPIClient
from streamlit_app.tests.sample_payloads import (
    SAMPLE_ANSWER_RESPONSE,
    SAMPLE_COMPARE_RESPONSE,
    SAMPLE_ERROR_RESPONSE,
    SAMPLE_HEALTH,
    SAMPLE_SEARCH_RESPONSE,
    SAMPLE_UPLOAD_RESPONSE,
)


def _client():
    return PolicyLensAPIClient(base_url="http://backend.test", request_timeout_seconds=5.0, connect_timeout_seconds=2.0)


# --- Success paths / exact field mapping ---------------------------------------------


def test_health_success(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, json=SAMPLE_HEALTH))

    result = _client().health()

    assert result.ok is True
    assert result.data.status == "ok"


def test_upload_document_success_maps_every_field(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(201, json=SAMPLE_UPLOAD_RESPONSE))

    result = _client().upload_document(filename="policy.pdf", file_bytes=b"%PDF-1.4 fake")

    assert result.ok is True
    data = result.data
    assert data.document_id == SAMPLE_UPLOAD_RESPONSE["document_id"]
    assert data.filename == "bank_policy.pdf"
    assert data.page_count == 3
    assert data.character_count == 4200
    assert data.status == "processed"
    assert data.chunk_count == 6
    assert data.pages_with_text == 3
    assert data.indexed_chunk_count == 6
    assert data.pii_detected is True
    assert data.pii_entity_count == 2
    assert data.pii_categories == ["EMAIL", "SSN"]


def test_search_success_maps_every_field(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE))

    result = _client().search(query="overdraft fee")

    assert result.ok is True
    assert result.data.query == "overdraft fee"
    assert result.data.query_was_masked is False
    assert result.data.result_count == 2
    assert len(result.data.results) == 2
    first = result.data.results[0]
    assert first.chunk_id == "chunk-1"
    assert first.source_filename == "bank_policy.pdf"
    assert first.page_number == 4
    assert first.relevance_score == 0.87


def test_ask_success_maps_every_field_including_nested_citations(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, json=SAMPLE_ANSWER_RESPONSE))

    result = _client().ask(question="What is the overdraft fee?")

    assert result.ok is True
    assert result.data.evidence_count == 2
    assert len(result.data.model_results) == 2
    anthropic_result = result.data.model_results[0]
    assert anthropic_result.provider == "anthropic"
    assert anthropic_result.status == "success"
    assert len(anthropic_result.citations) == 1
    assert anthropic_result.citations[0].source_label == "S1"
    openai_result = result.data.model_results[1]
    assert openai_result.status == "error"
    assert openai_result.error == "The OpenAI API request timed out."
    assert openai_result.input_tokens is None


def test_compare_success_maps_every_field_including_metrics_and_comparison(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, json=SAMPLE_COMPARE_RESPONSE))

    result = _client().compare(question="What is the overdraft fee?")

    assert result.ok is True
    data = result.data
    assert len(data.provider_metrics) == 2
    anthropic_metrics = data.provider_metrics[0]
    assert anthropic_metrics.estimated_cost_usd == 0.002181
    assert anthropic_metrics.grounded_term_ratio == 0.8
    openai_metrics = data.provider_metrics[1]
    assert openai_metrics.estimated_cost_usd is None
    assert openai_metrics.grounded_term_ratio is None
    assert data.comparison.comparison_status == "both_successful"
    assert data.comparison.answer_agreement_score == 0.94
    assert data.comparison.estimated_cost_difference_usd is None


# --- Request construction -------------------------------------------------------------


def test_search_request_body_includes_optional_fields_only_when_given(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        return httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE)

    install_mock_transport(handler)

    _client().search(query="fees")

    import json

    body = json.loads(call_log[0].content)
    assert body == {"query": "fees"}


def test_search_request_body_includes_document_id_and_top_k_when_given(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        return httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE)

    install_mock_transport(handler)

    _client().search(query="fees", document_id="doc-123", top_k=10)

    import json

    body = json.loads(call_log[0].content)
    assert body == {"query": "fees", "document_id": "doc-123", "top_k": 10}


def test_upload_sends_correct_multipart_request(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        return httpx.Response(201, json=SAMPLE_UPLOAD_RESPONSE)

    install_mock_transport(handler)

    _client().upload_document(filename="my policy.pdf", file_bytes=b"%PDF-1.4 raw bytes", content_type="application/pdf")

    request = call_log[0]
    assert request.method == "POST"
    assert request.url.path == "/api/v1/documents/upload"
    content_type_header = request.headers["content-type"]
    assert content_type_header.startswith("multipart/form-data")
    assert b'filename="my policy.pdf"' in request.content
    assert b"%PDF-1.4 raw bytes" in request.content
    assert b'name="file"' in request.content


def test_ask_request_body_shape(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        return httpx.Response(200, json=SAMPLE_ANSWER_RESPONSE)

    install_mock_transport(handler)

    _client().ask(question="fee?", document_id="doc-1", providers=["anthropic"], top_k=3)

    import json

    body = json.loads(call_log[0].content)
    assert body == {"question": "fee?", "document_id": "doc-1", "providers": ["anthropic"], "top_k": 3}


# --- Connection / timeout error handling ------------------------------------------------


def _raise(exc_class):
    def handler(request):
        raise exc_class("simulated")

    return handler


def test_health_connect_error_is_reported_as_connection_kind(install_mock_transport):
    install_mock_transport(_raise(httpx.ConnectError))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "connection"


def test_health_connect_timeout_is_reported_as_connection_kind_not_timeout(install_mock_transport):
    # httpx.ConnectTimeout is a TimeoutException subclass -- this is the
    # specific case that must NOT fall through to the timeout-retry path,
    # or an offline backend takes 3x as long to report as unreachable.
    install_mock_transport(_raise(httpx.ConnectTimeout))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "connection"


def test_health_connect_error_is_never_retried(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        raise httpx.ConnectError("simulated")

    install_mock_transport(handler)

    _client().health()

    assert len(call_log) == 1


def test_health_read_timeout_is_retried_up_to_the_bound(install_mock_transport, call_log, monkeypatch):
    monkeypatch.setattr("streamlit_app.api_client.time.sleep", lambda seconds: None)

    def handler(request):
        call_log.append(request)
        raise httpx.ReadTimeout("simulated")

    install_mock_transport(handler)

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "timeout"
    assert len(call_log) == 3  # _SAFE_REQUEST_MAX_ATTEMPTS


def test_health_read_timeout_then_success_recovers(install_mock_transport, call_log, monkeypatch):
    monkeypatch.setattr("streamlit_app.api_client.time.sleep", lambda seconds: None)

    def handler(request):
        call_log.append(request)
        if len(call_log) < 2:
            raise httpx.ReadTimeout("simulated")
        return httpx.Response(200, json=SAMPLE_HEALTH)

    install_mock_transport(handler)

    result = _client().health()

    assert result.ok is True
    assert len(call_log) == 2


def test_search_read_timeout_is_retried(install_mock_transport, call_log, monkeypatch):
    monkeypatch.setattr("streamlit_app.api_client.time.sleep", lambda seconds: None)

    def handler(request):
        call_log.append(request)
        raise httpx.ReadTimeout("simulated")

    install_mock_transport(handler)

    result = _client().search(query="fees")

    assert result.ok is False
    assert result.error.kind == "timeout"
    assert len(call_log) == 3


def test_upload_is_never_retried_on_connect_error(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        raise httpx.ConnectError("simulated")

    install_mock_transport(handler)

    result = _client().upload_document(filename="a.pdf", file_bytes=b"data")

    assert result.ok is False
    assert result.error.kind == "connection"
    assert len(call_log) == 1


def test_upload_is_never_retried_on_timeout(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        raise httpx.ReadTimeout("simulated")

    install_mock_transport(handler)

    result = _client().upload_document(filename="a.pdf", file_bytes=b"data")

    assert result.ok is False
    assert result.error.kind == "timeout"
    assert len(call_log) == 1


def test_ask_is_never_retried_on_timeout(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        raise httpx.ReadTimeout("simulated")

    install_mock_transport(handler)

    result = _client().ask(question="fee?")

    assert result.ok is False
    assert result.error.kind == "timeout"
    assert len(call_log) == 1


def test_compare_is_never_retried_on_timeout(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        raise httpx.ReadTimeout("simulated")

    install_mock_transport(handler)

    result = _client().compare(question="fee?")

    assert result.ok is False
    assert result.error.kind == "timeout"
    assert len(call_log) == 1


# --- Invalid / unexpected response bodies -----------------------------------------------


def test_invalid_json_response_is_reported_safely(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, content=b"not json at all"))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "invalid_response"
    assert "not json at all" not in result.error.message


def test_unexpected_json_shape_is_reported_safely(install_mock_transport):
    # Valid JSON, but missing every field `HealthStatus`/etc. expect --
    # exercised here via the upload parser, which requires many fields.
    install_mock_transport(lambda request: httpx.Response(201, json={"unexpected": "shape"}))

    result = _client().upload_document(filename="a.pdf", file_bytes=b"data")

    assert result.ok is False
    assert result.error.kind == "invalid_response"


# --- 4xx / 5xx error responses -----------------------------------------------------------


def test_4xx_response_surfaces_the_backend_detail_message(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(404, json=SAMPLE_ERROR_RESPONSE))

    result = _client().search(query="fees", document_id="unknown-doc")

    assert result.ok is False
    assert result.error.kind == "client_error"
    assert result.error.status_code == 404
    assert result.error.message == SAMPLE_ERROR_RESPONSE["detail"]


def test_5xx_response_is_reported_as_server_error(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(503, json={"detail": "The vector store is currently unavailable."}))

    result = _client().search(query="fees")

    assert result.ok is False
    assert result.error.kind == "server_error"
    assert result.error.status_code == 503


def test_error_response_with_no_body_falls_back_to_a_generic_message(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(500, content=b""))

    result = _client().health()

    assert result.ok is False
    assert "500" in result.error.message
    # Never a raw stack trace or exception repr.
    assert "Traceback" not in result.error.message
    assert "Exception" not in result.error.message


def test_error_response_with_malformed_body_falls_back_to_a_generic_message(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(422, content=b"<html>not json</html>"))

    result = _client().search(query="fees")

    assert result.ok is False
    assert "<html>" not in result.error.message


def test_non_json_500_from_a_proxy_or_load_balancer_never_exposes_the_raw_body(install_mock_transport):
    # A real-world scenario the backend itself would never produce, but a
    # reverse proxy/load balancer in front of it might: an HTML error
    # page with internal server names/paths baked in.
    raw_html_error = (
        b"<html><body><h1>502 Bad Gateway</h1>"
        b"<p>nginx/1.25.3 on internal-host-42.corp.local</p></body></html>"
    )
    install_mock_transport(lambda request: httpx.Response(502, content=raw_html_error))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "server_error"
    assert "internal-host-42" not in result.error.message
    assert "nginx" not in result.error.message
    assert "<html>" not in result.error.message
    assert "502" in result.error.message


# --- httpx.InvalidURL: not an HTTPError subclass, needs its own handling ---------------


def _malformed_base_url_client():
    # A control character in the base URL triggers `httpx.InvalidURL` at
    # request time -- a bare `Exception` subclass, NOT an `httpx.HTTPError`
    # subclass, so it is not caught by a plain `except httpx.HTTPError`.
    return PolicyLensAPIClient(
        base_url="http://backend.test/\x01bad", request_timeout_seconds=5.0, connect_timeout_seconds=2.0
    )


def test_invalid_url_on_health_is_a_safe_configuration_error():
    result = _malformed_base_url_client().health()
    assert result.ok is False
    assert result.error.kind == "connection"
    assert "POLICYLENS_API_BASE_URL" in result.error.message


def test_invalid_url_on_search_is_a_safe_configuration_error():
    result = _malformed_base_url_client().search(query="fees")
    assert result.ok is False
    assert result.error.kind == "connection"


def test_invalid_url_on_upload_is_a_safe_configuration_error():
    result = _malformed_base_url_client().upload_document(filename="a.pdf", file_bytes=b"data")
    assert result.ok is False
    assert result.error.kind == "connection"


def test_invalid_url_on_ask_is_a_safe_configuration_error():
    result = _malformed_base_url_client().ask(question="fee?")
    assert result.ok is False
    assert result.error.kind == "connection"


def test_invalid_url_on_compare_is_a_safe_configuration_error():
    result = _malformed_base_url_client().compare(question="fee?")
    assert result.ok is False
    assert result.error.kind == "connection"


# --- Malformed (non-dict) 2xx payloads: AttributeError must be caught, not crash -------


def test_health_with_a_list_payload_is_a_safe_invalid_response(install_mock_transport):
    # HealthStatus.from_dict calls `.get(...)` first -- on a list, that
    # raises AttributeError, not KeyError/TypeError.
    install_mock_transport(lambda request: httpx.Response(200, json=["unexpected", "shape"]))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "invalid_response"


def test_health_with_a_null_payload_is_a_safe_invalid_response(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, content=b"null"))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "invalid_response"


def test_health_with_a_string_payload_is_a_safe_invalid_response(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(200, content=b'"just a string"'))

    result = _client().health()

    assert result.ok is False
    assert result.error.kind == "invalid_response"


def test_compare_with_a_list_comparison_field_is_a_safe_invalid_response(install_mock_transport):
    # Comparison.from_dict also calls `.get(...)` first -- same failure
    # mode, reached via a nested field this time (CompareResult.from_dict
    # passes `data["comparison"]` straight through).
    malformed = {**SAMPLE_COMPARE_RESPONSE, "comparison": ["not", "a", "dict"]}
    install_mock_transport(lambda request: httpx.Response(200, json=malformed))

    result = _client().compare(question="fee?")

    assert result.ok is False
    assert result.error.kind == "invalid_response"


def test_upload_with_a_list_payload_is_a_safe_invalid_response(install_mock_transport):
    install_mock_transport(lambda request: httpx.Response(201, json=["unexpected"]))

    result = _client().upload_document(filename="a.pdf", file_bytes=b"data")

    assert result.ok is False
    assert result.error.kind == "invalid_response"


# --- URL construction: base URL variants must never drop /api/v1 or malform the path ---


@pytest.mark.parametrize(
    "base_url",
    [
        "http://backend.test",
        "http://backend.test/",
        "http://backend.test:8000",
        "http://backend.test:8000/",
        "https://backend.test",
    ],
)
def test_upload_path_is_correct_regardless_of_base_url_trailing_slash(install_mock_transport, call_log, base_url):
    def handler(request):
        call_log.append(request)
        return httpx.Response(201, json=SAMPLE_UPLOAD_RESPONSE)

    install_mock_transport(handler)

    client = PolicyLensAPIClient(base_url=base_url, request_timeout_seconds=5.0, connect_timeout_seconds=2.0)
    client.upload_document(filename="a.pdf", file_bytes=b"data")

    request_url = str(call_log[0].url)
    assert request_url.count("/api/v1/documents/upload") == 1
    assert "//api" not in request_url.replace("://", "")  # no doubled slashes from naive concatenation
    assert request_url.endswith("/api/v1/documents/upload")


def test_all_endpoint_paths_include_their_expected_segment(install_mock_transport, call_log):
    def handler(request):
        call_log.append(request)
        if "/health" in request.url.path:
            return httpx.Response(200, json=SAMPLE_HEALTH)
        if "/search" in request.url.path:
            return httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE)
        if "/answers" in request.url.path:
            return httpx.Response(200, json=SAMPLE_ANSWER_RESPONSE)
        if "/compare" in request.url.path:
            return httpx.Response(200, json=SAMPLE_COMPARE_RESPONSE)
        return httpx.Response(404, json={"detail": "unrouted in test"})

    install_mock_transport(handler)
    client = _client()

    client.health()
    client.search(query="fees")
    client.ask(question="fee?")
    client.compare(question="fee?")

    paths = [request.url.path for request in call_log]
    assert paths == ["/health", "/api/v1/search", "/api/v1/answers", "/api/v1/compare"]


# --- httpx clients are properly closed (no leaked connections/ResourceWarning) ---------


def test_many_sequential_requests_produce_no_resource_warnings(install_mock_transport, recwarn):
    install_mock_transport(lambda request: httpx.Response(200, json=SAMPLE_HEALTH))

    client = _client()
    for _ in range(20):
        result = client.health()
        assert result.ok is True

    resource_warnings = [w for w in recwarn.list if issubclass(w.category, ResourceWarning)]
    assert resource_warnings == []


# --- No logging of sensitive content -----------------------------------------------------


def test_api_client_module_has_no_logging_calls():
    # Checks for the actual code patterns (an import statement, a logger
    # call) rather than the bare word "logging", which this module's own
    # docstring uses in prose while describing this exact guarantee.
    import inspect

    import streamlit_app.api_client as api_client_module

    source = inspect.getsource(api_client_module)
    assert "import logging" not in source
    assert "logger." not in source
    assert ".log(" not in source
