"""Tests for the ChromaDB-backed VectorStore service.

All tests use the `vector_store` / `tmp_chroma_dir` fixtures (isolated
temp directory + `FakeEmbeddingProvider`) from conftest.py -- never the
project's real `data/vector_store/` directory, and never a network call.
"""

import hashlib

import pytest

from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.services.ingestion.text_chunker import Chunk
from app.services.retrieval.embeddings import FakeEmbeddingProvider
from app.services.retrieval.vector_store import EXCERPT_CHAR_LIMIT, VectorStore, _distance_to_relevance_score


def _chunk(
    text,
    document_id="doc-1",
    chunk_index=0,
    page_number=1,
    source_filename="policy.pdf",
):
    chunk_id = hashlib.sha256(f"{document_id}:{page_number}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        page_number=page_number,
        source_filename=source_filename,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


# --- Indexing -----------------------------------------------------------


def test_upsert_chunks_indexes_and_returns_count(vector_store):
    chunks = [_chunk("Overdraft fees are $35 per occurrence.", chunk_index=0)]

    indexed = vector_store.upsert_chunks(chunks)

    assert indexed == 1
    assert vector_store.document_exists("doc-1") is True


def test_upsert_empty_list_is_a_noop(vector_store):
    assert vector_store.upsert_chunks([]) == 0


def test_upsert_is_idempotent_for_identical_chunk_ids(vector_store):
    chunks = [_chunk("Overdraft fees are $35 per occurrence.", chunk_index=0)]

    first_count = vector_store.upsert_chunks(chunks)
    second_count = vector_store.upsert_chunks(chunks)  # re-index identical content

    assert first_count == 1
    assert second_count == 1
    results = vector_store.search(query="overdraft fees", top_k=10)
    assert len(results) == 1  # not duplicated


def test_reindexing_updates_content_instead_of_duplicating(vector_store):
    original = _chunk("Overdraft fees are $35 per occurrence.", chunk_index=0)
    vector_store.upsert_chunks([original])

    # Same chunk_id (same document_id/page/index), but the "content" behind
    # it is re-processed with different text -- simulates a corrected
    # re-upload. This should replace, not duplicate, the stored entry.
    updated = _chunk("Overdraft fees are $40 per occurrence.", chunk_index=0)
    updated.chunk_id = original.chunk_id
    vector_store.upsert_chunks([updated])

    results = vector_store.search(query="overdraft fees", top_k=10)
    assert len(results) == 1
    assert "$40" in results[0].excerpt


# --- Metadata round-trip --------------------------------------------------


def test_search_result_metadata_matches_indexed_chunk(vector_store):
    chunk = _chunk(
        "Minimum balance requirements apply to all savings accounts.",
        document_id="doc-xyz",
        chunk_index=3,
        page_number=7,
        source_filename="savings-terms.pdf",
    )
    vector_store.upsert_chunks([chunk])

    results = vector_store.search(query="minimum balance requirements", top_k=5)

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == chunk.chunk_id
    assert result.document_id == "doc-xyz"
    assert result.page_number == 7
    assert result.source_filename == "savings-terms.pdf"


# --- Ordering / relevance --------------------------------------------------


def test_results_are_ordered_by_relevance(vector_store):
    vector_store.upsert_chunks(
        [
            _chunk("overdraft fee schedule and penalty amounts", chunk_index=0),
            _chunk("checking account routing number instructions", chunk_index=1),
            _chunk("overdraft fee policy overview", chunk_index=2),
        ]
    )

    results = vector_store.search(query="overdraft fee policy", top_k=3)

    assert len(results) == 3
    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The two chunks sharing vocabulary with the query should outrank the
    # unrelated one, which must therefore rank last.
    assert results[-1].excerpt == "checking account routing number instructions"


def test_distance_to_relevance_score_formula():
    assert _distance_to_relevance_score(0.0) == 1.0
    assert _distance_to_relevance_score(2.0) == 0.0
    assert _distance_to_relevance_score(1.0) == 0.5
    # Defensive clamping for floating-point drift beyond the theoretical range.
    assert _distance_to_relevance_score(-0.001) == 1.0
    assert _distance_to_relevance_score(2.5) == 0.0


# --- Document-scoped vs cross-document search -------------------------------


def test_document_scoped_search_only_returns_that_document(vector_store):
    vector_store.upsert_chunks(
        [
            _chunk("overdraft policy for account A", document_id="doc-a", chunk_index=0),
            _chunk("overdraft policy for account B", document_id="doc-b", chunk_index=0),
        ]
    )

    results = vector_store.search(query="overdraft policy", top_k=10, document_id="doc-a")

    assert len(results) == 1
    assert results[0].document_id == "doc-a"


def test_cross_document_search_returns_matches_from_multiple_documents(vector_store):
    vector_store.upsert_chunks(
        [
            _chunk("overdraft policy for account A", document_id="doc-a", chunk_index=0),
            _chunk("overdraft policy for account B", document_id="doc-b", chunk_index=0),
        ]
    )

    results = vector_store.search(query="overdraft policy", top_k=10)

    document_ids = {r.document_id for r in results}
    assert document_ids == {"doc-a", "doc-b"}


def test_unknown_document_id_raises_not_found(vector_store):
    vector_store.upsert_chunks([_chunk("some content", document_id="doc-a")])

    with pytest.raises(DocumentNotFoundError):
        vector_store.search(query="some content", top_k=5, document_id="doc-does-not-exist")


# --- top_k ------------------------------------------------------------------


def test_top_k_limits_result_count(vector_store):
    vector_store.upsert_chunks(
        [_chunk(f"policy clause number {i} about fees", chunk_index=i) for i in range(10)]
    )

    results = vector_store.search(query="policy clause fees", top_k=3)

    assert len(results) == 3


# --- Minimum relevance score filtering --------------------------------------


def test_min_relevance_score_filters_out_low_scoring_results(vector_store):
    vector_store.upsert_chunks(
        [
            _chunk("overdraft fee policy overview", chunk_index=0),
            _chunk("checking account routing number instructions", chunk_index=1),
        ]
    )

    unfiltered = vector_store.search(query="overdraft fee policy", top_k=10, min_relevance_score=0.0)
    assert len(unfiltered) == 2

    # A threshold between the two scores keeps only the closer match.
    cutoff = (unfiltered[0].relevance_score + unfiltered[1].relevance_score) / 2
    filtered = vector_store.search(query="overdraft fee policy", top_k=10, min_relevance_score=cutoff)

    assert len(filtered) == 1
    assert filtered[0].relevance_score >= cutoff


def test_impossibly_high_min_relevance_score_returns_no_results(vector_store):
    vector_store.upsert_chunks([_chunk("overdraft fee policy overview", chunk_index=0)])

    results = vector_store.search(query="overdraft fee policy", top_k=10, min_relevance_score=1.01)

    assert results == []


# --- Excerpt capping ---------------------------------------------------------


def test_excerpt_is_capped_and_never_the_full_long_chunk(vector_store):
    long_text = "Account terms and conditions. " * 30  # well over EXCERPT_CHAR_LIMIT
    assert len(long_text) > EXCERPT_CHAR_LIMIT
    vector_store.upsert_chunks([_chunk(long_text, chunk_index=0)])

    results = vector_store.search(query="account terms and conditions", top_k=1)

    assert len(results) == 1
    assert len(results[0].excerpt) <= EXCERPT_CHAR_LIMIT + 3  # + "..."
    assert results[0].excerpt != long_text
    assert results[0].excerpt.endswith("...")


def test_short_chunk_excerpt_is_not_truncated(vector_store):
    short_text = "Short clause."
    vector_store.upsert_chunks([_chunk(short_text, chunk_index=0)])

    results = vector_store.search(query="short clause", top_k=1)

    assert results[0].excerpt == short_text


# --- Persistence --------------------------------------------------------------


def test_data_persists_across_service_reinstantiation(tmp_chroma_dir):
    first_instance = VectorStore(
        persist_directory=tmp_chroma_dir,
        collection_name="persist_test",
        embedding_provider=FakeEmbeddingProvider(),
    )
    first_instance.upsert_chunks([_chunk("overdraft fee policy overview", chunk_index=0)])

    # A brand-new VectorStore pointed at the SAME directory/collection name
    # must see data written by the first instance -- proving persistence
    # to disk, not just in-process memory.
    second_instance = VectorStore(
        persist_directory=tmp_chroma_dir,
        collection_name="persist_test",
        embedding_provider=FakeEmbeddingProvider(),
    )
    results = second_instance.search(query="overdraft fee policy", top_k=5)

    assert len(results) == 1
    assert results[0].document_id == "doc-1"


# --- Unavailable / corrupted vector store ------------------------------------


def test_vector_store_unavailable_path_raises_typed_error(tmp_path):
    # Point the persistence directory at an existing *file* rather than a
    # directory -- ChromaDB cannot open a SQLite store there and raises;
    # VectorStore must translate that into a client-safe typed error.
    blocked_path = tmp_path / "not_a_directory.txt"
    blocked_path.write_text("this occupies the path ChromaDB needs as a directory")

    with pytest.raises(VectorStoreError):
        VectorStore(
            persist_directory=str(blocked_path),
            collection_name="test_collection",
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_search_wraps_unexpected_chroma_failures(vector_store, monkeypatch):
    vector_store.upsert_chunks([_chunk("overdraft fee policy overview", chunk_index=0)])

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated Chroma failure")

    monkeypatch.setattr(vector_store._collection, "query", _boom)

    with pytest.raises(VectorStoreError):
        vector_store.search(query="overdraft fee policy", top_k=5)
