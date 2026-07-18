"""Shared pytest fixtures for the PolicyLens AI backend test suite."""

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm.providers import FakeLLMProvider, get_llm_provider_registry
from app.services.retrieval.embeddings import FakeEmbeddingProvider
from app.services.retrieval.vector_store import VectorStore, get_vector_store


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """A fresh, isolated directory for a test's own ChromaDB persistence.

    Never the project's real `data/vector_store/` directory, and never
    reused across tests -- pytest's `tmp_path` gives each test function
    its own throwaway directory.
    """

    return str(tmp_path / "chroma")


@pytest.fixture
def vector_store(tmp_chroma_dir):
    """A `VectorStore` backed by an isolated temp directory and the
    deterministic `FakeEmbeddingProvider` -- no model download, no
    network access.
    """

    return VectorStore(
        persist_directory=tmp_chroma_dir,
        collection_name="test_collection",
        embedding_provider=FakeEmbeddingProvider(),
    )


@pytest.fixture
def llm_providers():
    """Default fake LLM providers (deterministic, no network) for the `client`
    fixture. Individual tests override `get_llm_provider_registry` directly
    for success/error/timeout/etc. scenarios.
    """

    # citations: ["S1"] -- a "success" status requires at least one valid
    # citation when evidence was supplied; an empty list here would make
    # this an (incorrect) apparently-grounded-but-uncited default.
    return {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            model="fake-anthropic-model",
            response_json={"insufficient_evidence": False, "answer": "Default fake answer.", "citations": ["S1"]},
        ),
        "openai": FakeLLMProvider(
            name="openai",
            model="fake-openai-model",
            response_json={"insufficient_evidence": False, "answer": "Default fake answer.", "citations": ["S1"]},
        ),
    }


@pytest.fixture
def client(vector_store, llm_providers):
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_llm_provider_registry] = lambda: llm_providers
    return TestClient(app)


def _build_pdf(pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def valid_pdf_bytes():
    """A genuine, unencrypted, two-page PDF built entirely in memory."""

    return _build_pdf(
        [
            "Hello World, this is page one of the policy document.",
            "This is page two, with different content for testing.",
        ]
    )


@pytest.fixture
def encrypted_pdf_bytes():
    """A password-protected PDF, built and encrypted entirely in memory."""

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret content that requires a password.")
    data = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
        permissions=int(fitz.PDF_PERM_PRINT),
    )
    doc.close()
    return data


@pytest.fixture
def corrupted_pdf_bytes(valid_pdf_bytes):
    """Keeps the %PDF- signature but truncates the body so parsing fails."""

    return valid_pdf_bytes[:100]


@pytest.fixture(autouse=True)
def _reset_dependency_overrides():
    yield
    app.dependency_overrides.clear()
