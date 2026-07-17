"""Unit tests for the deterministic text chunking service."""

import pytest

from app.core.exceptions import InvalidChunkConfigurationError
from app.services.ingestion.pdf_processor import PageContent, process_pdf
from app.services.ingestion.text_chunker import (
    _normalize_whitespace,
    chunk_document,
    chunk_pages,
)


def _page(text, page_number=1, source_filename="policy.pdf"):
    return PageContent(
        page_number=page_number,
        source_filename=source_filename,
        text=text,
        character_count=len(text),
    )


def test_short_text_produces_single_chunk():
    page = _page("Short policy clause about overdraft fees.")

    chunks = chunk_pages([page], "doc-1", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Short policy clause about overdraft fees."
    assert chunk.start_character == 0
    assert chunk.end_character == len(chunk.text)
    assert chunk.chunk_index == 0
    assert chunk.page_number == 1


def test_long_text_produces_multiple_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(200))  # well over chunk_size
    page = _page(text)

    chunks = chunk_pages([page], "doc-1", chunk_size=50, chunk_overlap=10, min_chunk_length=5)

    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:]):
        # Consecutive chunks on the same page must overlap: the next chunk
        # starts before the current one ends.
        assert following.start_character < current.end_character
        assert following.chunk_index == current.chunk_index + 1
        assert current.character_count <= 50


def test_chunk_ids_are_deterministic_for_identical_input():
    text = " ".join(f"clause-{i}" for i in range(120))
    page = _page(text)

    first_run = chunk_pages([page], "doc-1", chunk_size=40, chunk_overlap=8, min_chunk_length=5)
    second_run = chunk_pages([page], "doc-1", chunk_size=40, chunk_overlap=8, min_chunk_length=5)

    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]
    assert [c.text for c in first_run] == [c.text for c in second_run]
    assert len(first_run) > 1


def test_chunk_id_changes_if_document_id_differs():
    page = _page("Fees may apply to overdrawn accounts.")

    chunks_a = chunk_pages([page], "doc-a", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)
    chunks_b = chunk_pages([page], "doc-b", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert chunks_a[0].chunk_id != chunks_b[0].chunk_id
    # Content is identical, so the content_hash (independent of document_id)
    # must still match.
    assert chunks_a[0].content_hash == chunks_b[0].content_hash


def test_chunk_metadata_matches_source_document_and_page():
    pages = [
        _page("Page one content about savings accounts.", page_number=1, source_filename="terms.pdf"),
        _page("Page two content about checking accounts.", page_number=2, source_filename="terms.pdf"),
    ]

    chunks = chunk_pages(pages, "doc-42", chunk_size=1000, chunk_overlap=50, min_chunk_length=10)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    for chunk in chunks:
        assert chunk.document_id == "doc-42"
        assert chunk.source_filename == "terms.pdf"


def test_empty_and_whitespace_only_pages_produce_no_chunks():
    pages = [
        _page("", page_number=1),
        _page("   \n\t  ", page_number=2),
        _page("Real content on the third page.", page_number=3),
    ]

    chunks = chunk_pages(pages, "doc-1", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0


def test_whitespace_is_normalized_without_altering_financial_values():
    raw = "Interest   rate:\n\n10.5%\tper annum.\r\nMinimum balance: $1,000.00."
    page = _page(raw)

    chunks = chunk_pages([page], "doc-1", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Interest rate: 10.5% per annum. Minimum balance: $1,000.00."
    assert _normalize_whitespace(raw) == chunks[0].text


def test_chunks_are_cut_on_word_boundaries_not_mid_word():
    text = " ".join(f"clause{i}" for i in range(80))
    normalized = _normalize_whitespace(text)
    page = _page(text)

    chunks = chunk_pages([page], "doc-1", chunk_size=30, chunk_overlap=6, min_chunk_length=5)

    for chunk in chunks:
        if chunk.end_character < len(normalized):
            # The character immediately after the cut must be the space
            # that separates two words -- i.e. the cut itself lands
            # exactly on a word boundary rather than inside a token.
            assert normalized[chunk.end_character] == " "
        assert not chunk.text.startswith(" ")
        assert not chunk.text.endswith(" ")


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "min_chunk_length"),
    [
        (0, 0, 0),
        (-10, 0, 0),
        (100, -1, 0),
        (100, 100, 0),  # overlap == chunk_size
        (100, 150, 0),  # overlap > chunk_size
        (100, 0, -1),
        (100, 0, 200),  # min_chunk_length > chunk_size
    ],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size, chunk_overlap, min_chunk_length):
    page = _page("Some policy text.")

    with pytest.raises(InvalidChunkConfigurationError):
        chunk_pages(
            [page],
            "doc-1",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_length=min_chunk_length,
        )


def test_minimum_chunk_length_merges_small_trailing_remainder():
    # 100 words of ~6-7 chars each; chunk_size/overlap chosen so the sliding
    # window leaves a small remainder at the end of the page.
    text = " ".join(f"item{i:02d}" for i in range(60))
    normalized = _normalize_whitespace(text)
    page = _page(text)

    chunks_no_min = chunk_pages([page], "doc-1", chunk_size=25, chunk_overlap=5, min_chunk_length=0)
    chunks_with_min = chunk_pages([page], "doc-1", chunk_size=25, chunk_overlap=5, min_chunk_length=15)

    assert len(chunks_with_min) <= len(chunks_no_min)
    assert all(c.character_count >= 15 or c.end_character == len(normalized) for c in chunks_with_min)
    # No content is lost: the last chunk still reaches the end of the text
    # and there are no gaps between consecutive chunks.
    assert chunks_with_min[-1].end_character == len(normalized)
    for current, following in zip(chunks_with_min, chunks_with_min[1:]):
        assert following.start_character <= current.end_character


def test_word_boundary_snap_does_not_cascade_into_duplicate_chunks():
    # Regression test: a long word ("klmnopqrst", 10 chars) sitting right
    # after a short first word used to make the word-boundary snap repeatedly
    # fall back to the same early space on every iteration (since the next
    # window still contained it), producing a cascade of near-duplicate,
    # near-empty chunks instead of real forward progress.
    text = "abcdefghij klmnopqrst " + " ".join(f"tail{i}" for i in range(20))
    normalized = _normalize_whitespace(text)
    page = _page(text)

    chunks = chunk_pages([page], "doc-1", chunk_size=10, chunk_overlap=3, min_chunk_length=0)

    # No chunk should be a tiny fragment of another (the bug produced 1-3
    # character chunks that were pure substrings of the previous chunk).
    assert all(chunk.character_count >= 4 for chunk in chunks)

    # No character span outside the two-page-independent, page-local text
    # should be covered more than `chunk_overlap` characters redundantly:
    # consecutive chunks' overlap must never exceed chunk_size (a loose
    # upper bound that the old cascade badly violated -- (7,10),(8,10),
    # (9,10) all overlapped almost entirely with the first chunk).
    for current, following in zip(chunks, chunks[1:]):
        overlap = current.end_character - following.start_character
        assert 0 <= overlap <= 10

    # Full coverage, no gaps.
    assert chunks[0].start_character == 0
    assert chunks[-1].end_character == len(normalized)
    for current, following in zip(chunks, chunks[1:]):
        assert following.start_character <= current.end_character


def test_single_token_longer_than_chunk_size_hard_cuts_cleanly():
    long_token = "X" * 250  # one unbroken "word" (e.g. an account number)
    text = f"Account reference {long_token} on file."
    normalized = _normalize_whitespace(text)
    page = _page(text)

    chunks = chunk_pages([page], "doc-1", chunk_size=50, chunk_overlap=10, min_chunk_length=5)

    assert len(chunks) > 1
    # Every character of the source text is covered by at least one chunk.
    assert chunks[0].start_character == 0
    assert chunks[-1].end_character == len(normalized)
    for current, following in zip(chunks, chunks[1:]):
        assert following.start_character <= current.end_character
        # The forced hard cut through the long token must not regress
        # backwards or stall: each chunk makes real forward progress.
        assert following.start_character > current.start_character
    for chunk in chunks:
        assert normalized[chunk.start_character:chunk.end_character] == chunk.text


def test_overlap_close_to_but_smaller_than_chunk_size():
    text = " ".join(f"clause{i}" for i in range(100))
    normalized = _normalize_whitespace(text)
    page = _page(text)

    chunk_size = 40
    chunk_overlap = 39  # as close to chunk_size as the validator allows
    chunks = chunk_pages([page], "doc-1", chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_length=5)

    assert len(chunks) > 1
    assert chunks[0].start_character == 0
    assert chunks[-1].end_character == len(normalized)
    for current, following in zip(chunks, chunks[1:]):
        # Forward progress is still guaranteed even with a near-maximal
        # overlap, and no chunk regresses behind the previous one's start.
        assert following.start_character > current.start_character
        assert following.chunk_index == current.chunk_index + 1


def test_unicode_text_is_chunked_correctly():
    text = (
        "Politique de confidentialité : les frais de découvert s'élèvent à 25 € "
        "par transaction. 個人情報の取り扱いについて — 手数料は¥500です。 "
        "Überziehungsgebühr beträgt €30 pro Monat. 🏦 Emoji and multilingual text "
        "must not break chunk boundaries or character counts."
    )
    page = _page(text)

    chunks = chunk_pages([page], "doc-1", chunk_size=60, chunk_overlap=15, min_chunk_length=5)

    normalized = _normalize_whitespace(text)
    assert len(chunks) > 1
    assert chunks[0].start_character == 0
    assert chunks[-1].end_character == len(normalized)
    for chunk in chunks:
        assert normalized[chunk.start_character:chunk.end_character] == chunk.text
        assert chunk.character_count == len(chunk.text)
        # chunk_id/content_hash must be computable (UTF-8 safe) without error.
        assert len(chunk.chunk_id) == 64
        assert len(chunk.content_hash) == 64


def test_financial_symbols_and_percentages_are_never_altered():
    raw = "APR: 19.99%.\tOverdraft fee:  $35.00 . Balance:\n£1,234.56 or €1,000.00. Rate change: -0.25%."
    page = _page(raw)

    chunks = chunk_pages([page], "doc-1", chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 1
    text = chunks[0].text
    for token in ["19.99%", "$35.00", "£1,234.56", "€1,000.00", "-0.25%"]:
        assert token in text


def test_deterministic_results_across_repeated_runs_multi_page():
    pages = [
        _page("Savings account overdraft terms and 12.5% penalty rate.", page_number=1),
        _page("Checking account monthly fee of $8.00 unless waived.", page_number=2),
        _page("Closing an account: final statement issued within 30 days.", page_number=3),
    ]

    runs = [
        chunk_pages(pages, "doc-stable", chunk_size=35, chunk_overlap=8, min_chunk_length=5)
        for _ in range(5)
    ]

    baseline = runs[0]
    for run in runs[1:]:
        assert len(run) == len(baseline)
        for a, b in zip(baseline, run):
            assert a.chunk_id == b.chunk_id
            assert a.content_hash == b.content_hash
            assert a.text == b.text
            assert a.page_number == b.page_number
            assert a.chunk_index == b.chunk_index
            assert a.start_character == b.start_character
            assert a.end_character == b.end_character


def test_multi_page_pdf_chunking_preserves_page_numbers():
    from tests.conftest import _build_pdf  # reuse the in-memory PDF builder

    pdf_bytes = _build_pdf(
        [
            "Page one: savings account terms and overdraft policy details.",
            "Page two: checking account terms and monthly fee schedule.",
            "Page three: closing an account and final statement rules.",
        ]
    )

    extracted = process_pdf(pdf_bytes, "multi.pdf", "application/pdf")
    chunks = chunk_document(extracted, chunk_size=1000, chunk_overlap=100, min_chunk_length=10)

    assert len(chunks) == 3
    assert [c.page_number for c in chunks] == [1, 2, 3]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.document_id == extracted.document_id for c in chunks)
    assert all(c.source_filename == extracted.filename for c in chunks)
