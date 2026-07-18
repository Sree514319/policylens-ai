"""Tests for VectorStore's PII_REDACTION_VERSION enforcement.

Refusing to open a collection built under a different (or missing) privacy
version -- whenever PII protection is currently enabled -- prevents
silently mixing raw/differently-masked chunks with chunks masked under the
current rules. See `VectorStore._verify_collection_configuration`.
"""

import pytest

from app.core.exceptions import PrivacyVersionMismatchError
from app.services.retrieval.embeddings import FakeEmbeddingProvider
from app.services.retrieval.vector_store import VectorStore
from tests.test_vector_store import _chunk


def _store(tmp_chroma_dir, collection_name="privacy_test", **overrides):
    kwargs = dict(
        persist_directory=tmp_chroma_dir,
        collection_name=collection_name,
        embedding_provider=FakeEmbeddingProvider(),
        pii_protection_enabled=True,
        pii_redaction_version="v1",
    )
    kwargs.update(overrides)
    return VectorStore(**kwargs)


def test_fresh_collection_records_the_current_pii_redaction_version(tmp_chroma_dir):
    store = _store(tmp_chroma_dir, pii_redaction_version="v1")
    store.upsert_chunks([_chunk("account terms", chunk_index=0)])

    assert store._collection.metadata.get("pii_redaction_version") == "v1"


def test_reopening_with_the_same_version_does_not_raise(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_redaction_version="v1")

    # Must not raise -- legitimate reuse of an already-current collection.
    _store(tmp_chroma_dir, pii_redaction_version="v1")


def test_reopening_with_a_different_version_raises(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_redaction_version="v1")

    with pytest.raises(PrivacyVersionMismatchError):
        _store(tmp_chroma_dir, pii_redaction_version="v2")


def test_reopening_a_collection_with_no_recorded_version_is_refused_when_protection_enabled(tmp_chroma_dir):
    # Simulate a collection built before PII protection existed (or while
    # it was disabled): no version was ever recorded.
    _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version=None)

    with pytest.raises(PrivacyVersionMismatchError):
        _store(tmp_chroma_dir, pii_protection_enabled=True, pii_redaction_version="v1")


def test_missing_version_error_includes_the_migration_instruction(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version=None)

    with pytest.raises(PrivacyVersionMismatchError) as exc_info:
        _store(tmp_chroma_dir, pii_protection_enabled=True, pii_redaction_version="v1")

    detail = exc_info.value.detail
    assert "delete" in detail.lower()
    assert "vector store" in detail.lower() or "vector_store" in detail.lower()


def test_legacy_collection_opens_fine_when_protection_is_disabled(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version=None)

    # Protection is off now too -- no version check should even run.
    store = _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version=None)
    assert store is not None


def test_mismatched_version_does_not_raise_when_protection_is_disabled(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_protection_enabled=True, pii_redaction_version="v1")

    # Even though the stored version differs conceptually from "v2", the
    # check is skipped entirely once protection is off -- the operator has
    # explicitly accepted the risk.
    store = _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version="v2")
    assert store is not None


def test_consistent_collection_still_indexes_and_searches_normally(tmp_chroma_dir):
    store = _store(tmp_chroma_dir, pii_redaction_version="v1")
    store.upsert_chunks([_chunk("[SSN_REDACTED] appears in this masked chunk.", chunk_index=0)])

    results = store.search(query="masked chunk", top_k=5)

    assert len(results) == 1
    assert "[SSN_REDACTED]" in results[0].excerpt


def test_mismatch_is_rejected_before_any_data_could_be_written_or_read(tmp_chroma_dir):
    # A collection built under "v1" already holds one chunk.
    v1_store = _store(tmp_chroma_dir, pii_redaction_version="v1")
    v1_store.upsert_chunks([_chunk("original v1 content", chunk_index=0)])

    # Reopening under "v2" must fail in the constructor itself -- before
    # any caller could possibly reach `.search()` or `.upsert_chunks()` on
    # the new instance (there is no instance; construction raised).
    with pytest.raises(PrivacyVersionMismatchError):
        _store(tmp_chroma_dir, pii_redaction_version="v2")

    # The original v1 data is untouched -- reopening correctly under "v1"
    # still finds exactly what was written, proving the failed "v2" attempt
    # never got far enough to read or overwrite anything.
    reopened = _store(tmp_chroma_dir, pii_redaction_version="v1")
    results = reopened.search(query="original v1 content", top_k=5)
    assert len(results) == 1


def test_error_never_includes_actual_indexed_document_text(tmp_chroma_dir):
    _store(tmp_chroma_dir, pii_protection_enabled=False, pii_redaction_version=None)

    with pytest.raises(PrivacyVersionMismatchError) as exc_info:
        _store(tmp_chroma_dir, pii_protection_enabled=True, pii_redaction_version="v1")

    # The error is purely about configuration -- it never needed access to
    # (and never includes) any indexed chunk content.
    assert "SSN" not in exc_info.value.detail
    assert "[" not in exc_info.value.detail  # no accidental placeholder/content leakage
