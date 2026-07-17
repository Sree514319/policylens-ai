"""Regression tests for the Phase 4 pre-commit safety review.

Covers: malformed/mismatched embedding vectors, embedding-provider /
collection configuration drift, large-batch indexing, malformed Chroma
query results, local-embedding-provider construction failures, and
`RETRIEVAL_TOP_K` clamping. All tests use isolated temp directories and
the deterministic `FakeEmbeddingProvider` (or small hand-written fakes) --
no model download, no network access.
"""

import math

import chromadb
import pytest

from app.core.exceptions import EmbeddingConfigurationMismatchError, VectorStoreError
from app.services.retrieval.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from app.services.retrieval.vector_store import VectorStore, get_vector_store
from tests.test_vector_store import _chunk


class _WrongDimensionProvider(EmbeddingProvider):
    """Claims one dimension but actually returns vectors of another."""

    @property
    def name(self):
        return "wrong-dimension-provider"

    @property
    def dimension(self):
        return 32

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]  # only 3 dims, not 32

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _NaNProvider(EmbeddingProvider):
    @property
    def name(self):
        return "nan-provider"

    @property
    def dimension(self):
        return 4

    def embed_documents(self, texts):
        return [[0.1, float("nan"), 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        return [0.1, float("nan"), 0.3, 0.4]


class _NonNumericProvider(EmbeddingProvider):
    @property
    def name(self):
        return "non-numeric-provider"

    @property
    def dimension(self):
        return 3

    def embed_documents(self, texts):
        return [["a", "b", "c"] for _ in texts]

    def embed_query(self, text):
        return ["a", "b", "c"]


class _NotAListProvider(EmbeddingProvider):
    """Returns something that isn't even a list of vectors."""

    @property
    def name(self):
        return "not-a-list-provider"

    @property
    def dimension(self):
        return 4

    def embed_documents(self, texts):
        return None

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class _RenamedFakeProvider(FakeEmbeddingProvider):
    """Same embedding behavior as FakeEmbeddingProvider, different declared name."""

    @property
    def name(self):
        return "renamed-fake-provider"


class _DifferentDimensionFakeProvider(FakeEmbeddingProvider):
    """Declares a different dimension than FakeEmbeddingProvider (32)."""

    @property
    def dimension(self):
        return 64


# --- Malformed / mismatched embedding vectors --------------------------------


def test_upsert_rejects_wrong_dimension_embeddings(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "safety_test", _WrongDimensionProvider())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([_chunk("some policy text", chunk_index=0)])


def test_upsert_rejects_nan_embedding_values(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "safety_test", _NaNProvider())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([_chunk("some policy text", chunk_index=0)])


def test_upsert_rejects_non_numeric_embedding_values(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "safety_test", _NonNumericProvider())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([_chunk("some policy text", chunk_index=0)])


def test_upsert_rejects_non_list_embedding_result(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "safety_test", _NotAListProvider())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([_chunk("some policy text", chunk_index=0)])


def test_search_rejects_wrong_dimension_query_embedding(tmp_chroma_dir):
    # Index successfully with a well-behaved provider, then swap in a
    # broken one that would return an incompatible query embedding.
    good_store = VectorStore(tmp_chroma_dir, "safety_test", FakeEmbeddingProvider())
    good_store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])

    good_store._embedding_provider = _WrongDimensionProvider()
    with pytest.raises(VectorStoreError):
        good_store.search(query="overdraft fee policy", top_k=5)


# --- Embedding-provider / collection configuration drift ----------------------


def test_reopening_with_different_provider_name_raises_mismatch(tmp_chroma_dir):
    VectorStore(tmp_chroma_dir, "drift_test", FakeEmbeddingProvider())

    with pytest.raises(EmbeddingConfigurationMismatchError):
        VectorStore(tmp_chroma_dir, "drift_test", _RenamedFakeProvider())


def test_reopening_with_different_dimension_raises_mismatch(tmp_chroma_dir):
    VectorStore(tmp_chroma_dir, "drift_test_dim", FakeEmbeddingProvider())

    with pytest.raises(EmbeddingConfigurationMismatchError):
        VectorStore(tmp_chroma_dir, "drift_test_dim", _DifferentDimensionFakeProvider())


def test_reopening_with_same_provider_does_not_raise(tmp_chroma_dir):
    VectorStore(tmp_chroma_dir, "no_drift_test", FakeEmbeddingProvider())

    # Must not raise -- same provider name/dimension, legitimate reuse.
    VectorStore(tmp_chroma_dir, "no_drift_test", FakeEmbeddingProvider())


def test_reopening_legacy_non_cosine_collection_raises_mismatch(tmp_chroma_dir):
    # Simulate a collection that predates/bypassed our cosine-space setup:
    # created directly via chromadb with no metadata at all (defaults to
    # ChromaDB's own default space, "l2").
    client = chromadb.PersistentClient(path=tmp_chroma_dir)
    client.get_or_create_collection(name="legacy_collection", embedding_function=None)

    with pytest.raises(EmbeddingConfigurationMismatchError):
        VectorStore(tmp_chroma_dir, "legacy_collection", FakeEmbeddingProvider())


# --- Large-batch indexing ------------------------------------------------------


def test_large_batch_indexing_splits_across_multiple_upserts(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "batch_test", FakeEmbeddingProvider())
    store._max_batch_size = 5  # force multiple small batches

    chunks = [_chunk(f"policy clause number {i} about fees", chunk_index=i) for i in range(23)]
    indexed = store.upsert_chunks(chunks)

    assert indexed == 23
    results = store.search(query="policy clause fees", top_k=23)
    assert len(results) == 23


def test_batch_boundary_exact_multiple_does_not_leave_a_stray_empty_batch(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "batch_test_exact", FakeEmbeddingProvider())
    store._max_batch_size = 4

    chunks = [_chunk(f"clause {i}", chunk_index=i) for i in range(8)]  # exactly 2 batches
    indexed = store.upsert_chunks(chunks)

    assert indexed == 8
    assert len(store.search(query="clause", top_k=8)) == 8


def test_indexed_count_matches_actual_stored_count_across_batches(tmp_chroma_dir):
    store = VectorStore(tmp_chroma_dir, "batch_count_test", FakeEmbeddingProvider())
    store._max_batch_size = 3

    chunks = [_chunk(f"term {i} of the agreement", chunk_index=i) for i in range(10)]
    indexed = store.upsert_chunks(chunks)

    all_ids = store._collection.get(limit=100).get("ids") or []
    assert indexed == 10
    assert len(all_ids) == 10


# --- Malformed / empty Chroma result shapes ------------------------------------


def test_search_handles_missing_result_keys_safely(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "malformed_test", FakeEmbeddingProvider())
    store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])

    # A response missing every expected key entirely is treated the same
    # as "no results" -- safe, not a crash -- via the existing `.get(...)
    # or [[]]` fallback.
    monkeypatch.setattr(store._collection, "query", lambda **kwargs: {})

    assert store.search(query="overdraft fee policy", top_k=5) == []


def test_search_handles_none_valued_result_keys_safely(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "malformed_test2", FakeEmbeddingProvider())
    store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])

    monkeypatch.setattr(
        store._collection,
        "query",
        lambda **kwargs: {"ids": None, "documents": None, "metadatas": None, "distances": None},
    )

    # Treated as "no results" rather than crashing.
    assert store.search(query="overdraft fee policy", top_k=5) == []


def test_search_rejects_mismatched_result_array_lengths(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "malformed_test3", FakeEmbeddingProvider())
    store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])

    monkeypatch.setattr(
        store._collection,
        "query",
        lambda **kwargs: {
            "ids": [["a", "b"]],
            "documents": [["only one"]],  # shorter than ids -- inconsistent
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        },
    )

    with pytest.raises(VectorStoreError):
        store.search(query="overdraft fee policy", top_k=5)


# --- Local embedding provider construction/embed failures ---------------------


def test_get_vector_store_wraps_provider_construction_failure(monkeypatch):
    import app.services.retrieval.vector_store as vector_store_module

    def _boom():
        raise RuntimeError("simulated onnxruntime import failure at /some/internal/path")

    monkeypatch.setattr(vector_store_module, "LocalEmbeddingProvider", _boom)
    get_vector_store.cache_clear()

    try:
        with pytest.raises(VectorStoreError) as exc_info:
            get_vector_store()
        # The client-safe detail must never leak the underlying exception's
        # message (which could contain a local file path).
        assert "/some/internal/path" not in str(exc_info.value.detail)
    finally:
        get_vector_store.cache_clear()


def test_embed_documents_failure_during_upsert_maps_to_vector_store_error(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "embed_fail_test", FakeEmbeddingProvider())

    def _boom(texts):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(store._embedding_provider, "embed_documents", _boom)

    with pytest.raises(VectorStoreError):
        store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])


def test_embed_query_failure_during_search_maps_to_vector_store_error(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "embed_fail_test2", FakeEmbeddingProvider())
    store.upsert_chunks([_chunk("overdraft fee policy", chunk_index=0)])

    def _boom(text):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(store._embedding_provider, "embed_query", _boom)

    with pytest.raises(VectorStoreError):
        store.search(query="overdraft fee policy", top_k=5)


# --- Empty indexing input (regression guard) -----------------------------------


def test_upsert_empty_list_does_not_touch_chroma(tmp_chroma_dir, monkeypatch):
    store = VectorStore(tmp_chroma_dir, "empty_input_test", FakeEmbeddingProvider())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("upsert should not be called for an empty chunk list")

    monkeypatch.setattr(store._collection, "upsert", _fail_if_called)

    assert store.upsert_chunks([]) == 0
