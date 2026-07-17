"""Unit tests for the PDF ingestion service, independent of the API layer."""

import pytest

from app.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    InvalidContentTypeError,
    InvalidFileExtensionError,
    InvalidFileSignatureError,
)
from app.services.ingestion.pdf_processor import process_pdf, sanitize_filename


def test_process_pdf_preserves_page_number_and_source_filename(valid_pdf_bytes):
    result = process_pdf(valid_pdf_bytes, "policy.pdf", "application/pdf")

    assert result.page_count == 2
    assert len(result.pages) == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert all(page.source_filename == "policy.pdf" for page in result.pages)
    assert all(page.character_count == len(page.text) for page in result.pages)
    assert result.character_count == sum(page.character_count for page in result.pages)


def test_process_pdf_document_id_depends_only_on_content(valid_pdf_bytes):
    first = process_pdf(valid_pdf_bytes, "a.pdf", "application/pdf")
    second = process_pdf(valid_pdf_bytes, "totally-different-name.pdf", "application/pdf")

    assert first.document_id == second.document_id
    assert len(first.document_id) == 64


def test_process_pdf_rejects_empty_file():
    with pytest.raises(EmptyFileError):
        process_pdf(b"", "empty.pdf", "application/pdf")


def test_process_pdf_rejects_wrong_extension(valid_pdf_bytes):
    with pytest.raises(InvalidFileExtensionError):
        process_pdf(valid_pdf_bytes, "policy.txt", "application/pdf")


def test_process_pdf_rejects_wrong_content_type(valid_pdf_bytes):
    with pytest.raises(InvalidContentTypeError):
        process_pdf(valid_pdf_bytes, "policy.pdf", "application/octet-stream")


def test_process_pdf_rejects_invalid_signature():
    with pytest.raises(InvalidFileSignatureError):
        process_pdf(b"not a pdf at all", "fake.pdf", "application/pdf")


def test_process_pdf_rejects_corrupted_pdf(corrupted_pdf_bytes):
    with pytest.raises(CorruptedPDFError):
        process_pdf(corrupted_pdf_bytes, "broken.pdf", "application/pdf")


def test_process_pdf_rejects_encrypted_pdf(encrypted_pdf_bytes):
    with pytest.raises(EncryptedPDFError):
        process_pdf(encrypted_pdf_bytes, "secret.pdf", "application/pdf")


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("policy.pdf", "policy.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
        ("weird name?<>.pdf", "weird name___.pdf"),
        ("", "document.pdf"),
    ],
)
def test_sanitize_filename(raw_name, expected):
    assert sanitize_filename(raw_name) == expected


def test_sanitize_filename_truncates_long_names():
    long_name = ("a" * 300) + ".pdf"
    sanitized = sanitize_filename(long_name)

    assert len(sanitized) <= 150
    assert sanitized.endswith(".pdf")
