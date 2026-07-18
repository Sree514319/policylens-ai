"""Tests for POST /api/v1/search."""

import pytest

from app.core.config import Settings, get_settings
from app.main import app
from app.schemas.search import MAX_TOP_K

UPLOAD_URL = "/api/v1/documents/upload"
SEARCH_URL = "/api/v1/search"


def _upload(client, pdf_bytes, filename="policy.pdf"):
    response = client.post(UPLOAD_URL, files={"file": (filename, pdf_bytes, "application/pdf")})
    assert response.status_code == 201
    return response.json()


def test_search_returns_results_for_indexed_document(client, valid_pdf_bytes):
    uploaded = _upload(client, valid_pdf_bytes)

    response = client.post(SEARCH_URL, json={"query": "Hello World page one"})

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] >= 1
    assert any(r["document_id"] == uploaded["document_id"] for r in body["results"])


def test_search_response_exact_key_set(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(SEARCH_URL, json={"query": "Hello World"})
    body = response.json()

    assert set(body.keys()) == {"query", "query_was_masked", "result_count", "results"}
    assert len(body["results"]) >= 1
    for result in body["results"]:
        assert set(result.keys()) == {
            "chunk_id",
            "document_id",
            "source_filename",
            "page_number",
            "excerpt",
            "relevance_score",
        }


def _build_pdf_with_wrapped_text(text):
    # `page.insert_text` (used by the shared `_build_pdf` helper) draws from
    # a single point with no word-wrap, so a long string silently runs off
    # the page and only the first line or so actually gets extracted.
    # `insert_textbox` wraps within a rectangle, so long text survives
    # extraction intact -- needed here to produce a single chunk longer
    # than EXCERPT_CHAR_LIMIT.
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_search_excerpt_never_carries_a_full_large_chunk(client):
    long_page_text = "Account terms and conditions apply broadly. " * 40
    pdf_bytes = _build_pdf_with_wrapped_text(long_page_text)
    uploaded = _upload(client, pdf_bytes, filename="long.pdf")
    assert uploaded["chunk_count"] >= 1
    assert uploaded["character_count"] > 303

    response = client.post(SEARCH_URL, json={"query": "account terms and conditions"})

    assert response.status_code == 200
    excerpt = response.json()["results"][0]["excerpt"]
    assert len(excerpt) <= 303  # EXCERPT_CHAR_LIMIT + "..."
    assert excerpt.endswith("...")
    assert excerpt != long_page_text.strip()


@pytest.mark.parametrize("query", ["", "   ", "\t\n  "])
def test_empty_or_whitespace_query_is_rejected(client, query):
    response = client.post(SEARCH_URL, json={"query": query})

    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [0, -1, 51, 1000])
def test_top_k_out_of_bounds_is_rejected(client, top_k):
    response = client.post(SEARCH_URL, json={"query": "overdraft fees", "top_k": top_k})

    assert response.status_code == 422


def test_top_k_within_bounds_is_accepted(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(SEARCH_URL, json={"query": "Hello World", "top_k": 1})

    assert response.status_code == 200
    assert len(response.json()["results"]) <= 1


def test_misconfigured_retrieval_top_k_default_is_clamped(client, vector_store, valid_pdf_bytes, monkeypatch):
    _upload(client, valid_pdf_bytes)
    app.dependency_overrides[get_settings] = lambda: Settings(retrieval_top_k=9999)

    captured = {}
    original_search = vector_store.search

    def _spy_search(*args, **kwargs):
        captured["top_k"] = kwargs.get("top_k")
        return original_search(*args, **kwargs)

    monkeypatch.setattr(vector_store, "search", _spy_search)

    response = client.post(SEARCH_URL, json={"query": "Hello World"})

    assert response.status_code == 200
    assert captured["top_k"] is not None
    assert captured["top_k"] <= MAX_TOP_K


def test_unknown_document_id_returns_404(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(
        SEARCH_URL,
        json={"query": "Hello World", "document_id": "0" * 64},
    )

    assert response.status_code == 404
    assert "detail" in response.json()


def test_document_scoped_search_via_api(client, valid_pdf_bytes):
    from tests.conftest import _build_pdf

    other_pdf_bytes = _build_pdf(["Completely unrelated content about routing numbers."])

    first = _upload(client, valid_pdf_bytes, filename="first.pdf")
    _upload(client, other_pdf_bytes, filename="second.pdf")

    response = client.post(
        SEARCH_URL,
        json={"query": "Hello World page one", "document_id": first["document_id"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert all(r["document_id"] == first["document_id"] for r in body["results"])


def test_search_before_any_upload_returns_empty_results(client):
    response = client.post(SEARCH_URL, json={"query": "anything at all"})

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 0
    assert body["results"] == []


def test_search_query_is_trimmed_and_echoed(client, valid_pdf_bytes):
    _upload(client, valid_pdf_bytes)

    response = client.post(SEARCH_URL, json={"query": "  Hello World  "})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Hello World"
    assert body["query_was_masked"] is False


# --- PII masking in the query --------------------------------------------------------


def test_query_containing_pii_is_masked_before_search(client):
    response = client.post(SEARCH_URL, json={"query": "What is the policy for SSN 123-45-6789?"})

    assert response.status_code == 200
    body = response.json()
    assert "123-45-6789" not in body["query"]
    assert "[SSN_REDACTED]" in body["query"]
    assert body["query_was_masked"] is True


def test_original_query_pii_never_appears_anywhere_in_the_response(client):
    response = client.post(SEARCH_URL, json={"query": "Email jane.doe@example.com about my account"})

    assert "jane.doe@example.com" not in response.text
    assert "[EMAIL_REDACTED]" in response.text


def test_pii_in_uploaded_filename_never_appears_in_search_results(client):
    from tests.conftest import _build_pdf

    pdf_bytes = _build_pdf(["Hello World, this is page one of the policy document."])
    upload_response = client.post(
        UPLOAD_URL, files={"file": ("Customer 555-123-4567 statement.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_response.status_code == 201

    response = client.post(SEARCH_URL, json={"query": "Hello World page one"})

    assert response.status_code == 200
    assert "555-123-4567" not in response.text
    body = response.json()
    assert any("[PHONE_REDACTED]" in r["source_filename"] for r in body["results"])


def test_pii_protection_disabled_leaves_query_unmasked(client):
    app.dependency_overrides[get_settings] = lambda: Settings(pii_protection_enabled=False)

    response = client.post(SEARCH_URL, json={"query": "What is the policy for SSN 123-45-6789?"})

    body = response.json()
    assert "123-45-6789" in body["query"]
    assert body["query_was_masked"] is False
