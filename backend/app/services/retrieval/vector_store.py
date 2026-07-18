"""ChromaDB-backed vector store for citation-ready chunk retrieval.

This is the only module in the codebase that imports ``chromadb`` or
touches its objects. Every other layer (API routes, tests) interacts
through the plain dataclasses defined here (`SearchResult`) -- raw
ChromaDB types (query results, collections, clients) never cross this
boundary, per the "never expose raw ChromaDB internals through the API"
requirement.

Embeddings are always computed by an injected `EmbeddingProvider` and
passed to Chroma explicitly (`embeddings=...` / `query_embeddings=...`).
The collection itself is created with ``embedding_function=None``, so
nothing here can silently trigger a model download or network call from
inside ChromaDB -- if a caller ever forgot to supply embeddings, it would
fail loudly instead.

Privacy note: chunk *text* (already stripped of the original PDF bytes,
which are never persisted -- see `pdf_processor`) is stored alongside its
embedding in the configured persistence directory so it can be returned
as a search excerpt later. That directory (`data/vector_store/` by
default) is git-ignored; nothing indexed here is ever committed to the
repository.
"""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import chromadb

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    EmbeddingConfigurationMismatchError,
    PrivacyVersionMismatchError,
    VectorStoreError,
)
from app.services.ingestion.text_chunker import Chunk
from app.services.privacy.masking import validate_pii_configuration
from app.services.retrieval.embeddings import EmbeddingProvider, LocalEmbeddingProvider

logger = logging.getLogger(__name__)

EXCERPT_CHAR_LIMIT = 300
_HNSW_SPACE_KEY = "hnsw:space"
_PROVIDER_NAME_KEY = "embedding_provider_name"
_PROVIDER_DIMENSION_KEY = "embedding_dimension"
_PII_REDACTION_VERSION_KEY = "pii_redaction_version"
_EXPECTED_HNSW_SPACE = "cosine"

_MIGRATION_INSTRUCTION = (
    "Delete the local vector store directory (CHROMA_PERSIST_DIRECTORY, "
    "'data/vector_store' by default -- it is git-ignored, so this only "
    "affects your local index) and re-upload your documents so they are "
    "indexed under the current PII_REDACTION_VERSION."
)

# Conservative fallback if the installed Chroma client ever lacks
# get_max_batch_size() (older/alternate backends). The real, version-
# specific limit is queried at init time whenever available.
_DEFAULT_MAX_BATCH_SIZE = 2000


@dataclass
class SearchResult:
    """A single citation-ready search hit. Never carries the full chunk text."""

    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int
    excerpt: str
    relevance_score: float


def _distance_to_relevance_score(distance: float) -> float:
    """Map a ChromaDB cosine distance to a normalized [0.0, 1.0] relevance score.

    The collection is configured for cosine space, where
    ``distance = 1 - cosine_similarity``. Since cosine similarity ranges
    over [-1, 1], distance ranges over [0, 2]: 0 means identical direction
    (best possible match), 2 means diametrically opposite (worst possible
    match). We linearly remap that onto [0.0, 1.0]:

        relevance_score = 1 - (distance / 2)

    so 1.0 is the best possible match and 0.0 is the worst. The result is
    clamped defensively in case of floating-point drift at the extremes.
    """

    score = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, score))


def _build_excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHAR_LIMIT:
        return text
    return text[:EXCERPT_CHAR_LIMIT] + "..."


def _validate_embedding_batch(vectors: object, expected_dimension: int) -> None:
    """Guard against a malformed or dimension-mismatched embedding provider result.

    A provider that silently returns the wrong shape (empty vectors, the
    wrong dimension, non-numeric values, or NaNs) would otherwise either
    crash inside ChromaDB with an opaque error or -- worse -- get accepted
    and corrupt similarity search. This fails fast with a clear, typed,
    client-safe error instead.
    """

    if not isinstance(vectors, list):
        raise VectorStoreError("The embedding provider returned an invalid result.")

    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or len(vector) != expected_dimension:
            raise VectorStoreError("The embedding provider returned a vector of unexpected dimension.")
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(value):
                raise VectorStoreError("The embedding provider returned a malformed vector.")


class VectorStore:
    """Wraps a single persistent Chroma collection for chunk storage and search."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        pii_protection_enabled: bool = True,
        pii_redaction_version: Optional[str] = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._pii_protection_enabled = pii_protection_enabled
        self._pii_redaction_version = pii_redaction_version

        # ChromaDB's metadata values must be str/int/float/bool -- None is
        # rejected outright (a TypeError from the Rust bindings), so the
        # key is only included when there's an actual version to record.
        # A collection created with no version (PII protection disabled at
        # the time) simply has no entry here, which `metadata.get(...)`
        # in `_verify_collection_configuration` already treats as "missing".
        creation_metadata = {
            _HNSW_SPACE_KEY: _EXPECTED_HNSW_SPACE,
            _PROVIDER_NAME_KEY: embedding_provider.name,
            _PROVIDER_DIMENSION_KEY: embedding_provider.dimension,
        }
        if pii_redaction_version is not None:
            creation_metadata[_PII_REDACTION_VERSION_KEY] = pii_redaction_version

        try:
            self._client = chromadb.PersistentClient(path=persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata=creation_metadata,
                embedding_function=None,
            )
            self._max_batch_size = self._client.get_max_batch_size()
        except Exception as exc:
            logger.error("Failed to initialize the vector store.")
            raise VectorStoreError("The vector store is currently unavailable.") from exc

        self._verify_collection_configuration()

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """The `EmbeddingProvider` this store indexes/searches with.

        Exposed so other services (e.g. model-comparison's answer-agreement
        score) can embed arbitrary text with the exact same provider
        instance already in use, instead of constructing a second one.
        """

        return self._embedding_provider

    def _verify_collection_configuration(self) -> None:
        """Reject silently mixing incompatible embeddings into one collection.

        ChromaDB's `get_or_create_collection` ignores the `metadata`
        argument when the collection already exists -- the space and any
        extra keys set at *first creation* stick permanently. So the
        metadata read back here reflects either the values this instance
        just set (brand-new collection) or whatever a prior instance set
        (existing collection). Comparing that against the *current*
        provider catches two real drift scenarios: EMBEDDING_MODEL_NAME
        changed since this collection was first created, or the
        collection predates/bypassed cosine-space configuration.

        The active HNSW space is read from `collection.configuration`
        (not `collection.metadata`) because Chroma always populates it --
        defaulting to "l2" -- even when no metadata was ever supplied at
        creation, whereas `.metadata` is `None` in that case. Using
        `.metadata` alone would silently skip exactly the legacy/foreign
        collection this check exists to catch.

        A collection with no tracking metadata at all for provider
        name/dimension (pre-Phase-4 data) cannot be verified on those two
        fields and is allowed through rather than hard-failing on data
        this code never wrote.
        """

        configuration = self._collection.configuration or {}
        actual_space = (configuration.get("hnsw") or {}).get("space")
        if actual_space != _EXPECTED_HNSW_SPACE:
            raise EmbeddingConfigurationMismatchError(
                "The vector store collection is not configured for cosine distance."
            )

        metadata = self._collection.metadata or {}
        stored_provider_name = metadata.get(_PROVIDER_NAME_KEY)
        if stored_provider_name is not None and stored_provider_name != self._embedding_provider.name:
            raise EmbeddingConfigurationMismatchError(
                "The configured embedding provider does not match the one this collection was built with."
            )

        stored_dimension = metadata.get(_PROVIDER_DIMENSION_KEY)
        if stored_dimension is not None and stored_dimension != self._embedding_provider.dimension:
            raise EmbeddingConfigurationMismatchError(
                "The configured embedding dimension does not match the one this collection was built with."
            )

        # Stricter than the embedding-provider check above: a *missing*
        # PII_REDACTION_VERSION is refused, not allowed through. Unlike
        # embedding drift (where legacy data is merely unverifiable), a
        # missing version here could mean this collection holds entirely
        # unmasked chunks -- silently accepting it while PII protection is
        # enabled would risk mixing raw and masked chunks in one
        # collection with no way to tell them apart later.
        if self._pii_protection_enabled:
            stored_pii_version = metadata.get(_PII_REDACTION_VERSION_KEY)
            if stored_pii_version != self._pii_redaction_version:
                raise PrivacyVersionMismatchError(
                    "This vector store collection was indexed under a different or missing "
                    f"PII_REDACTION_VERSION (expected '{self._pii_redaction_version}'). "
                    f"{_MIGRATION_INSTRUCTION}"
                )

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """Index (or re-index) chunks. Upserting identical chunk_ids is idempotent.

        Large chunk lists are inserted in batches capped at the backend's
        max batch size (ChromaDB rejects a single call exceeding it).
        """

        if not chunks:
            return 0

        batch_size = self._max_batch_size or _DEFAULT_MAX_BATCH_SIZE

        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                embeddings = self._embedding_provider.embed_documents([chunk.text for chunk in batch])
                _validate_embedding_batch(embeddings, self._embedding_provider.dimension)

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in batch],
                    metadatas=[
                        {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "source_filename": chunk.source_filename,
                            "start_character": chunk.start_character,
                            "end_character": chunk.end_character,
                            "content_hash": chunk.content_hash,
                        }
                        for chunk in batch
                    ],
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.error("Failed to index %d chunk(s) into the vector store.", len(chunks))
            raise VectorStoreError("Failed to index the document's chunks.") from exc

        return len(chunks)

    def document_exists(self, document_id: str) -> bool:
        try:
            result = self._collection.get(where={"document_id": document_id}, limit=1)
        except Exception as exc:
            logger.error("Failed to query the vector store for a document's existence.")
            raise VectorStoreError("The vector store is currently unavailable.") from exc

        return len(result.get("ids") or []) > 0

    def search(
        self,
        query: str,
        top_k: int,
        document_id: Optional[str] = None,
        min_relevance_score: float = 0.0,
    ) -> List[SearchResult]:
        """Run a semantic search, optionally scoped to a single document.

        Raises `DocumentNotFoundError` if `document_id` is given but no
        chunks are indexed under it. Raises `VectorStoreError` if the
        underlying store is unavailable or returns a malformed result.
        Results below `min_relevance_score` are filtered out and results
        are ordered by relevance (closest first), as returned by Chroma's
        HNSW index.
        """

        if document_id is not None and not self.document_exists(document_id):
            raise DocumentNotFoundError(f"No indexed document found for document_id '{document_id}'.")

        try:
            query_embedding = self._embedding_provider.embed_query(query)
            _validate_embedding_batch([query_embedding], self._embedding_provider.dimension)

            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": document_id} if document_id else None,
                include=["documents", "metadatas", "distances"],
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.error("Vector store search failed.")
            raise VectorStoreError("The vector store is currently unavailable.") from exc

        try:
            ids = (raw.get("ids") or [[]])[0]
            documents = (raw.get("documents") or [[]])[0]
            metadatas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]
        except (IndexError, TypeError) as exc:
            logger.error("Vector store returned a malformed query result.")
            raise VectorStoreError("The vector store returned an unexpected result.") from exc

        if not len(ids) == len(documents) == len(metadatas) == len(distances):
            # A real ChromaDB response never has mismatched parallel arrays;
            # if it ever did, silently zip()-truncating could drop or
            # misalign results. Fail loudly instead.
            logger.error("Vector store returned inconsistent result array lengths.")
            raise VectorStoreError("The vector store returned an unexpected result.")

        results: List[SearchResult] = []
        for chunk_id, document_text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = _distance_to_relevance_score(distance)
            if score < min_relevance_score:
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    source_filename=metadata.get("source_filename", ""),
                    page_number=metadata.get("page_number", 0),
                    excerpt=_build_excerpt(document_text),
                    relevance_score=score,
                )
            )

        return results


@lru_cache
def get_vector_store() -> VectorStore:
    """Process-wide singleton `VectorStore`, wired to the production embedding
    provider and configured persistence directory/collection name.

    Tests override this dependency (via `app.dependency_overrides`) with a
    `VectorStore` pointed at an isolated temporary directory and the
    deterministic `FakeEmbeddingProvider`, so this factory is never invoked
    -- and no model download or network access ever happens -- during
    the test suite.
    """

    settings = get_settings()
    validate_pii_configuration(settings.pii_protection_enabled, settings.pii_redaction_version)

    try:
        embedding_provider = LocalEmbeddingProvider()
    except Exception as exc:
        # Covers a missing/broken local dependency (onnxruntime, tokenizers)
        # or a failed model load -- never let the raw exception (which may
        # include local file paths) escape past this generic message.
        logger.error("Failed to initialize the local embedding provider.")
        raise VectorStoreError("The vector store is currently unavailable.") from exc

    return VectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedding_provider=embedding_provider,
        pii_protection_enabled=settings.pii_protection_enabled,
        pii_redaction_version=settings.pii_redaction_version,
    )
