"""Tests for POST /api/v1/documents/upload."""

import hashlib

from app.core.config import Settings, get_settings
from app.main import app

UPLOAD_URL = "/api/v1/documents/upload"


def test_valid_pdf_upload_returns_metadata(client, valid_pdf_bytes):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("policy.pdf", valid_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()

    assert body["filename"] == "policy.pdf"
    assert body["page_count"] == 2
    assert body["character_count"] > 0
    assert body["status"] == "processed"
    assert body["document_id"] == hashlib.sha256(valid_pdf_bytes).hexdigest()
    assert len(body["document_id"]) == 64
    assert isinstance(body["preview"], str)
    assert len(body["preview"]) <= 203  # 200 chars + "..."
    # The response must never contain the complete extracted text.
    assert "page two" not in body["preview"]

    # Phase 3: chunk/page summary counts are present...
    assert body["chunk_count"] == 2  # one short chunk per page at default chunk_size
    assert body["pages_with_text"] == 2

    # Phase 4: the same chunks were indexed into the vector store.
    assert body["indexed_chunk_count"] == body["chunk_count"]

    # ...but no field anywhere carries full chunk/page text.
    assert set(body.keys()) == {
        "document_id",
        "filename",
        "page_count",
        "character_count",
        "status",
        "preview",
        "chunk_count",
        "pages_with_text",
        "indexed_chunk_count",
    }
    assert "chunks" not in body
    assert "page two" not in str(body)


def test_reuploading_identical_pdf_does_not_duplicate_indexed_chunks(client, vector_store, valid_pdf_bytes):
    first = client.post(
        UPLOAD_URL,
        files={"file": ("policy.pdf", valid_pdf_bytes, "application/pdf")},
    )
    second = client.post(
        UPLOAD_URL,
        files={"file": ("policy-again.pdf", valid_pdf_bytes, "application/pdf")},
    )

    assert first.status_code == second.status_code == 201
    document_id = first.json()["document_id"]
    assert document_id == second.json()["document_id"]

    results = vector_store.search(query="Hello World page one", top_k=10, document_id=document_id)
    # Two chunks were produced by the PDF (one per page); re-uploading the
    # identical content must upsert in place, not double the count.
    assert len(results) == first.json()["chunk_count"]


def test_non_pdf_extension_rejected(client):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("notes.txt", b"just some plain text content", "text/plain")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_wrong_content_type_rejected(client, valid_pdf_bytes):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("policy.pdf", valid_pdf_bytes, "text/plain")},
    )

    assert response.status_code == 400


def test_invalid_pdf_signature_rejected(client):
    fake_bytes = b"This is definitely not a real pdf file, just padded plain bytes."

    response = client.post(
        UPLOAD_URL,
        files={"file": ("fake.pdf", fake_bytes, "application/pdf")},
    )

    assert response.status_code == 400


def test_empty_file_rejected(client):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_corrupted_pdf_rejected(client, corrupted_pdf_bytes):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("broken.pdf", corrupted_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 422


def test_encrypted_pdf_rejected(client, encrypted_pdf_bytes):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("secret.pdf", encrypted_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 422


def test_oversized_file_rejected(client):
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_size_mb=1)

    oversized_payload = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)
    response = client.post(
        UPLOAD_URL,
        files={"file": ("big.pdf", oversized_payload, "application/pdf")},
    )

    assert response.status_code == 413


def test_filename_is_sanitized(client, valid_pdf_bytes):
    response = client.post(
        UPLOAD_URL,
        files={"file": ("../../etc/weird name?<>.pdf", valid_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    filename = response.json()["filename"]

    assert "/" not in filename
    assert "\\" not in filename
    assert ".." not in filename
    assert filename.endswith(".pdf")


def test_document_id_is_stable_for_identical_content(client, valid_pdf_bytes):
    first = client.post(
        UPLOAD_URL,
        files={"file": ("copy1.pdf", valid_pdf_bytes, "application/pdf")},
    )
    second = client.post(
        UPLOAD_URL,
        files={"file": ("copy2.pdf", valid_pdf_bytes, "application/pdf")},
    )

    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["page_count"] == second.json()["page_count"] == 2
