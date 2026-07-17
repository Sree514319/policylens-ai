"""Tests for the embedding-provider abstraction.

`LocalEmbeddingProvider` is only ever *constructed* here, never called --
per its own docstring, construction merely imports onnxruntime/tokenizers
(already-installed ChromaDB dependencies) and never touches the network.
The actual embedding call would lazily download a model on first use,
which the test suite must never trigger.
"""

import math

from app.services.retrieval.embeddings import EmbeddingProvider, FakeEmbeddingProvider, LocalEmbeddingProvider


def test_fake_embedding_provider_is_deterministic():
    provider = FakeEmbeddingProvider()

    first = provider.embed_query("Overdraft fees apply to negative balances.")
    second = provider.embed_query("Overdraft fees apply to negative balances.")

    assert first == second
    assert len(first) == provider.dimension


def test_fake_embedding_provider_batches_match_single_calls():
    provider = FakeEmbeddingProvider()
    texts = ["Savings account terms.", "Checking account fees.", "Overdraft policy."]

    batch = provider.embed_documents(texts)
    singles = [provider.embed_query(text) for text in texts]

    assert batch == singles


def test_fake_embedding_provider_vectors_are_unit_normalized():
    provider = FakeEmbeddingProvider()

    vector = provider.embed_query("Interest rate is 12.5% APR.")
    magnitude = math.sqrt(sum(v * v for v in vector))

    assert math.isclose(magnitude, 1.0, rel_tol=1e-6)


def test_fake_embedding_provider_handles_empty_text():
    provider = FakeEmbeddingProvider()

    vector = provider.embed_query("")

    assert len(vector) == provider.dimension
    # An all-whitespace/empty vector is still deterministic.
    assert vector == provider.embed_query("")


def test_fake_embedding_provider_empty_batch_returns_empty_list():
    provider = FakeEmbeddingProvider()

    assert provider.embed_documents([]) == []


def test_fake_embedding_provider_shared_vocabulary_increases_similarity():
    provider = FakeEmbeddingProvider()

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # vectors are already unit-normalized

    query = provider.embed_query("overdraft fee policy")
    related = provider.embed_query("overdraft fee schedule")
    unrelated = provider.embed_query("checking account routing number")

    assert cosine(query, related) > cosine(query, unrelated)


def test_local_embedding_provider_is_constructible_without_network_access():
    provider = LocalEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 384
    assert provider.name == "chromadb-default-onnx-minilm-l6-v2"
