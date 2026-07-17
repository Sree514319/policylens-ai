"""Embedding-provider abstraction so vector storage is not tied to one model.

`VectorStore` depends only on the `EmbeddingProvider` interface, never on a
concrete embedding implementation. Two providers are available:

- `LocalEmbeddingProvider`: production use. Wraps ChromaDB's bundled local
  ONNX model (all-MiniLM-L6-v2). Runs entirely on-device; the model is
  downloaded once (cached under ~/.cache/chroma) on first use, and nothing
  is ever sent to OpenAI, Anthropic, or any other third-party API. No API
  key is required.
- `FakeEmbeddingProvider`: test use only. A deterministic, dependency-free,
  hash-based bag-of-words embedding. It captures no real semantic meaning,
  but is stable (same text -> same vector, always) and fast, so tests can
  exercise indexing, upsert idempotency, filtering, and relative ranking
  without any model download or network access.
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors for storage in / query against Chroma."""

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, stable identifier for this provider/model."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The length of every vector this provider produces."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """Production provider: ChromaDB's bundled local ONNX MiniLM model.

    Constructing this class never touches the network -- it only imports
    the ``onnxruntime``/``tokenizers`` packages that ship as ChromaDB
    dependencies. The one-time model download happens lazily, inside
    ChromaDB's own embedding function, the first time text is actually
    embedded.
    """

    _DIMENSION = 384

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    @property
    def name(self) -> str:
        return "chromadb-default-onnx-minilm-l6-v2"

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [[float(x) for x in vector] for vector in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider for tests only.

    Each token is hashed (SHA-256) into a fixed-length vector and token
    vectors are summed and L2-normalized -- a crude bag-of-words scheme.
    It is stable and reproducible, and texts sharing more vocabulary end
    up more cosine-similar than texts sharing none, which is enough for
    tests to exercise ranking and filtering behavior without a real model.
    """

    _DIMENSION = 32

    @property
    def name(self) -> str:
        return "fake-deterministic-hash-embedding"

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        tokens = text.lower().split() or [""]
        vector = [0.0] * self._DIMENSION

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self._DIMENSION):
                vector[i] += digest[i % len(digest)] / 255.0

        magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / magnitude for v in vector]
