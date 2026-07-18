"""Tests for PII masking orchestration (`app.services.privacy.masking`)."""

import pytest

from app.core.exceptions import PIIConfigurationError, PIIProcessingError
from app.services.ingestion.pdf_processor import ExtractedDocument, PageContent
from app.services.ingestion.text_chunker import chunk_document
from app.services.privacy.detectors import CATEGORY_EMAIL, CATEGORY_SSN, FakePIIDetector, LocalRegexPIIDetector, PIIEntity
from app.services.privacy.masking import apply_masking, mask_extracted_document, mask_query, validate_pii_configuration


# --- apply_masking -----------------------------------------------------------------


def test_apply_masking_replaces_each_entity_with_its_placeholder():
    text = "SSN 123-45-6789 on file."
    entities = [PIIEntity(category=CATEGORY_SSN, start=4, end=15, placeholder="[SSN_REDACTED]")]

    assert apply_masking(text, entities) == "SSN [SSN_REDACTED] on file."


def test_apply_masking_handles_multiple_entities_in_order():
    text = "a@example.com and 123-45-6789"
    entities = [
        PIIEntity(category=CATEGORY_EMAIL, start=0, end=13, placeholder="[EMAIL_REDACTED]"),
        PIIEntity(category=CATEGORY_SSN, start=18, end=29, placeholder="[SSN_REDACTED]"),
    ]

    assert apply_masking(text, entities) == "[EMAIL_REDACTED] and [SSN_REDACTED]"


def test_apply_masking_with_no_entities_returns_text_unchanged():
    assert apply_masking("nothing sensitive here", []) == "nothing sensitive here"


def test_apply_masking_never_contains_the_original_value():
    text = "Card 4111 1111 1111 1111 on file."
    entities = LocalRegexPIIDetector().detect(text)
    masked = apply_masking(text, entities)

    assert "4111 1111 1111 1111" not in masked
    assert "[CARD_REDACTED]" in masked


# --- mask_query --------------------------------------------------------------------


def test_mask_query_masks_and_reports_true_when_pii_present():
    masked, was_masked = mask_query("My SSN is 123-45-6789", LocalRegexPIIDetector())

    assert was_masked is True
    assert "123-45-6789" not in masked
    assert "[SSN_REDACTED]" in masked


def test_mask_query_reports_false_when_no_pii_present():
    masked, was_masked = mask_query("What is the overdraft fee?", LocalRegexPIIDetector())

    assert was_masked is False
    assert masked == "What is the overdraft fee?"


def test_mask_query_wraps_detector_failure_without_leaking_the_query_text():
    secret_query = "super-secret-query-marker-99887"
    detector = FakePIIDetector(raise_exception=RuntimeError(secret_query))

    with pytest.raises(PIIProcessingError) as exc_info:
        mask_query(secret_query, detector)

    assert secret_query not in exc_info.value.detail


# --- mask_extracted_document ---------------------------------------------------------


def _document(pages_text):
    pages = [
        PageContent(page_number=i + 1, source_filename="policy.pdf", text=text, character_count=len(text))
        for i, text in enumerate(pages_text)
    ]
    return ExtractedDocument(
        document_id="doc-hash",
        filename="policy.pdf",
        page_count=len(pages),
        character_count=sum(p.character_count for p in pages),
        preview=pages[0].text[:50] if pages else "",
        pages=pages,
    )


def test_mask_extracted_document_masks_every_page():
    document = _document(["SSN: 123-45-6789 on file.", "Contact a@example.com for help."])

    summary = mask_extracted_document(document, LocalRegexPIIDetector())

    assert "123-45-6789" not in document.pages[0].text
    assert "[SSN_REDACTED]" in document.pages[0].text
    assert "a@example.com" not in document.pages[1].text
    assert "[EMAIL_REDACTED]" in document.pages[1].text
    assert summary.detected is True
    assert summary.entity_count == 2
    assert summary.categories == ["EMAIL", "SSN"]


def test_mask_extracted_document_updates_character_count_and_preview():
    document = _document(["SSN: 123-45-6789 is quite a bit longer than the placeholder."])
    original_preview = document.preview

    mask_extracted_document(document, LocalRegexPIIDetector())

    assert document.character_count == len(document.pages[0].text)
    assert "123-45-6789" not in document.preview
    assert document.preview != original_preview


def test_mask_extracted_document_with_no_pii_reports_not_detected():
    document = _document(["Ordinary policy text with no sensitive content at all."])

    summary = mask_extracted_document(document, LocalRegexPIIDetector())

    assert summary.detected is False
    assert summary.entity_count == 0
    assert summary.categories == []
    assert document.pages[0].text == "Ordinary policy text with no sensitive content at all."


def test_mask_extracted_document_preserves_page_number_and_filename():
    document = _document(["Page one text.", "Page two text."])

    mask_extracted_document(document, LocalRegexPIIDetector())

    assert [p.page_number for p in document.pages] == [1, 2]
    assert all(p.source_filename == "policy.pdf" for p in document.pages)


# --- Filename masking ----------------------------------------------------------------


def _document_with_filename(filename, pages_text=("Ordinary page text.",)):
    pages = [
        PageContent(page_number=i + 1, source_filename=filename, text=text, character_count=len(text))
        for i, text in enumerate(pages_text)
    ]
    return ExtractedDocument(
        document_id="doc-hash",
        filename=filename,
        page_count=len(pages),
        character_count=sum(p.character_count for p in pages),
        preview=pages[0].text[:50] if pages else "",
        pages=pages,
    )


def test_mask_extracted_document_masks_pii_in_the_filename():
    document = _document_with_filename("jane.doe@example.com report.pdf")

    summary = mask_extracted_document(document, LocalRegexPIIDetector())

    assert document.filename == "[EMAIL_REDACTED] report.pdf"
    assert "jane.doe@example.com" not in document.filename
    assert "EMAIL" in summary.categories


def test_mask_extracted_document_propagates_the_masked_filename_to_every_page():
    document = _document_with_filename(
        "SSN-123-45-6789.pdf", pages_text=["Page one text.", "Page two text."]
    )

    mask_extracted_document(document, LocalRegexPIIDetector())

    assert document.filename == "SSN-[SSN_REDACTED].pdf"
    assert all(p.source_filename == "SSN-[SSN_REDACTED].pdf" for p in document.pages)
    assert all("123-45-6789" not in p.source_filename for p in document.pages)


def test_mask_extracted_document_with_no_pii_in_filename_leaves_it_unchanged():
    document = _document_with_filename("policy.pdf")

    mask_extracted_document(document, LocalRegexPIIDetector())

    assert document.filename == "policy.pdf"


def test_mask_extracted_document_filename_failure_never_leaks_the_filename():
    secret_filename = "super-secret-filename-marker-77331.pdf"
    document = _document_with_filename(secret_filename)
    detector = FakePIIDetector(raise_exception=RuntimeError(secret_filename))

    with pytest.raises(PIIProcessingError) as exc_info:
        mask_extracted_document(document, detector)

    assert secret_filename not in exc_info.value.detail


# --- Idempotency: masking an already-masked document is a no-op ----------------------


def test_masking_an_already_masked_document_again_changes_nothing():
    document = _document(["Card 4111 1111 1111 1111 on file, SSN 123-45-6789 too."])
    detector = LocalRegexPIIDetector()

    mask_extracted_document(document, detector)
    once_masked_text = document.pages[0].text
    once_masked_filename = document.filename

    mask_extracted_document(document, detector)

    assert document.pages[0].text == once_masked_text
    assert document.filename == once_masked_filename


def test_mask_extracted_document_handles_empty_pages_safely():
    document = _document(["", "   "])

    summary = mask_extracted_document(document, LocalRegexPIIDetector())

    assert summary.detected is False
    assert document.pages[0].text == ""


def test_mask_extracted_document_wraps_detector_failure_without_leaking_page_text():
    secret_page_text = "super-secret-page-marker-55221"
    document = _document([secret_page_text])
    detector = FakePIIDetector(raise_exception=RuntimeError(secret_page_text))

    with pytest.raises(PIIProcessingError) as exc_info:
        mask_extracted_document(document, detector)

    assert secret_page_text not in exc_info.value.detail


# --- Masking happens before chunking (only masked text is ever chunked) ----------------


def test_chunking_only_ever_sees_masked_text():
    document = _document(["Customer SSN 123-45-6789 must be protected in this policy clause."])

    mask_extracted_document(document, LocalRegexPIIDetector())
    chunks = chunk_document(document, chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 1
    assert "123-45-6789" not in chunks[0].text
    assert "[SSN_REDACTED]" in chunks[0].text


# --- Configuration validation --------------------------------------------------------


def test_validate_pii_configuration_accepts_a_real_version_when_enabled():
    validate_pii_configuration(True, "v1")  # must not raise


def test_validate_pii_configuration_rejects_empty_version_when_enabled():
    with pytest.raises(PIIConfigurationError):
        validate_pii_configuration(True, "")


def test_validate_pii_configuration_rejects_whitespace_only_version_when_enabled():
    with pytest.raises(PIIConfigurationError):
        validate_pii_configuration(True, "   ")


def test_validate_pii_configuration_allows_empty_version_when_disabled():
    validate_pii_configuration(False, "")  # must not raise -- protection is off
