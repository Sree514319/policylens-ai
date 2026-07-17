"""Typed application exceptions mapped to safe HTTP responses.

Every exception carries a client-safe ``detail`` message only. Internal
details (file paths, stack traces, parser internals) must never be placed
in ``detail`` since it is returned directly to the caller.
"""


class PolicyLensError(Exception):
    """Base class for all expected, handled application errors."""

    status_code: int = 400

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidFileExtensionError(PolicyLensError):
    status_code = 400


class InvalidContentTypeError(PolicyLensError):
    status_code = 400


class InvalidFileSignatureError(PolicyLensError):
    status_code = 400


class EmptyFileError(PolicyLensError):
    status_code = 400


class FileTooLargeError(PolicyLensError):
    status_code = 413


class CorruptedPDFError(PolicyLensError):
    status_code = 422


class EncryptedPDFError(PolicyLensError):
    status_code = 422


class InvalidChunkConfigurationError(PolicyLensError):
    """Raised when chunk_size/chunk_overlap/min_chunk_length are inconsistent.

    This reflects bad server configuration (env vars), not a bad client
    request, so it maps to 500 rather than a 4xx status.
    """

    status_code = 500


class VectorStoreError(PolicyLensError):
    """Raised when the ChromaDB-backed vector store is unavailable or
    returns a corrupted/unexpected result. Maps to 503 (Service
    Unavailable) since this reflects a dependency-availability problem,
    not a malformed client request.
    """

    status_code = 503


class DocumentNotFoundError(PolicyLensError):
    """Raised when a search request scopes to a document_id that has no
    indexed chunks in the vector store.
    """

    status_code = 404


class EmbeddingConfigurationMismatchError(PolicyLensError):
    """Raised when the active embedding provider does not match the
    provider/dimension recorded in an existing collection (e.g.
    EMBEDDING_MODEL_NAME was changed, or the collection's distance space
    is not cosine as expected). This is a server misconfiguration -- not
    a bad request -- so it maps to 500, matching
    `InvalidChunkConfigurationError`. Silently continuing would mix
    incompatible embeddings in the same collection and produce meaningless
    similarity scores.
    """

    status_code = 500
