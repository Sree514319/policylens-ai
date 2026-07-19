"""Typed HTTP client for the PolicyLens AI backend.

The frontend talks to FastAPI exclusively over HTTP through this module --
nothing here, or anywhere else under ``frontend/``, imports backend
service or schema code directly (see the module docstrings in
``app.services.*`` for what those actually do; this module only knows
the wire format: JSON in, JSON out).

Every public method returns an ``APIResult``: none of them raise for an
*expected* failure mode (connection refused, timeout, malformed JSON, a
4xx/5xx response) -- callers always get back a safe, typed result to
render. ``APIError.message`` is always a short, user-facing string; it
never contains a raw stack trace, exception repr, or other backend
internals. Only genuinely unexpected programming errors (a bad call
signature, etc.) would still raise normally.

Privacy: this module never logs uploaded file bytes, questions, answers,
or citation text -- it has no logging calls at all. Whatever a caller
does with the returned data (e.g. write it to `st.session_state`) is that
caller's responsibility -- see `session_state.py`.

Forward compatibility: every `from_dict` below extracts only the fields
it knows about (by name); an unrecognized *additional* field in a
response is silently ignored rather than rejected, so a future backend
release adding a new field never breaks this client. A *missing*
required field, or a field of the wrong shape (e.g. a list where a dict
was expected), is handled explicitly -- see `_parse_response`'s
`KeyError`/`TypeError`/`ValueError`/`AttributeError` handling -- as a
safe `"invalid_response"` result, never an uncaught exception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

import httpx

_HEALTH_PATH = "/health"
_UPLOAD_PATH = "/api/v1/documents/upload"
_SEARCH_PATH = "/api/v1/search"
_ANSWERS_PATH = "/api/v1/answers"
_COMPARE_PATH = "/api/v1/compare"

# Bounded retries only for safe (read-only, no side effect, not billed to a
# third party) requests: health checks and semantic search. Upload, ask,
# and compare are never retried automatically -- see each method's
# docstring for why.
_SAFE_REQUEST_MAX_ATTEMPTS = 3
_SAFE_REQUEST_RETRY_BACKOFF_SECONDS = 0.5

T = TypeVar("T")


# --- Errors and the generic result wrapper ------------------------------------------


@dataclass(frozen=True)
class APIError:
    """A safe-to-display error.

    `kind` is a small, fixed vocabulary a caller can branch on without
    string-matching `message`:
      - "connection": couldn't reach the backend at all (refused, DNS, etc.)
      - "timeout": the backend didn't respond within the configured timeout
      - "invalid_response": a 2xx response whose body wasn't the JSON
        shape this client expected
      - "client_error": a 4xx response (bad request, not found, etc.)
      - "server_error": a 5xx response
    """

    kind: str
    message: str
    status_code: Optional[int] = None


@dataclass(frozen=True)
class APIResult(Generic[T]):
    """Exactly one of `data` (on success) or `error` (on failure) is set."""

    ok: bool
    data: Optional[T] = None
    error: Optional[APIError] = None

    @classmethod
    def success(cls, data: T) -> "APIResult[T]":
        return cls(ok=True, data=data, error=None)

    @classmethod
    def failure(cls, error: APIError) -> "APIResult[T]":
        return cls(ok=False, data=None, error=error)


def _connection_error() -> APIError:
    return APIError(kind="connection", message="Could not connect to the PolicyLens backend. Is it running?")


def _timeout_error() -> APIError:
    return APIError(kind="timeout", message="The backend did not respond in time. Please try again.")


def _network_error() -> APIError:
    return APIError(kind="connection", message="A network error occurred while contacting the backend.")


def _configuration_error() -> APIError:
    return APIError(
        kind="connection",
        message="The configured backend URL is invalid. Check POLICYLENS_API_BASE_URL.",
    )


# --- Response dataclasses, mirroring backend schemas field-for-field ----------------


@dataclass(frozen=True)
class HealthStatus:
    status: str

    @classmethod
    def from_dict(cls, data: dict) -> "HealthStatus":
        return cls(status=str(data.get("status", "")))


@dataclass(frozen=True)
class DocumentUploadResult:
    document_id: str
    filename: str
    page_count: int
    character_count: int
    status: str
    preview: str
    chunk_count: int
    pages_with_text: int
    indexed_chunk_count: int
    pii_detected: bool
    pii_entity_count: int
    pii_categories: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentUploadResult":
        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            page_count=data["page_count"],
            character_count=data["character_count"],
            status=data["status"],
            preview=data["preview"],
            chunk_count=data["chunk_count"],
            pages_with_text=data["pages_with_text"],
            indexed_chunk_count=data["indexed_chunk_count"],
            pii_detected=data["pii_detected"],
            pii_entity_count=data["pii_entity_count"],
            pii_categories=list(data.get("pii_categories") or []),
        )


@dataclass(frozen=True)
class SearchResultItem:
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int
    excerpt: str
    relevance_score: float

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResultItem":
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_filename=data["source_filename"],
            page_number=data["page_number"],
            excerpt=data["excerpt"],
            relevance_score=data["relevance_score"],
        )


@dataclass(frozen=True)
class SearchResult:
    query: str
    query_was_masked: bool
    result_count: int
    results: List[SearchResultItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        return cls(
            query=data["query"],
            query_was_masked=data["query_was_masked"],
            result_count=data["result_count"],
            results=[SearchResultItem.from_dict(item) for item in data.get("results") or []],
        )


@dataclass(frozen=True)
class Citation:
    source_label: str
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int
    excerpt: str
    relevance_score: float

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        return cls(
            source_label=data["source_label"],
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            source_filename=data["source_filename"],
            page_number=data["page_number"],
            excerpt=data["excerpt"],
            relevance_score=data["relevance_score"],
        )


@dataclass(frozen=True)
class ModelResult:
    provider: str
    model: str
    status: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResult":
        return cls(
            provider=data["provider"],
            model=data["model"],
            status=data["status"],
            answer=data["answer"],
            citations=[Citation.from_dict(item) for item in data.get("citations") or []],
            latency_ms=data["latency_ms"],
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class AnswerResult:
    question: str
    query_was_masked: bool
    evidence_count: int
    model_results: List[ModelResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerResult":
        return cls(
            question=data["question"],
            query_was_masked=data["query_was_masked"],
            evidence_count=data["evidence_count"],
            model_results=[ModelResult.from_dict(item) for item in data.get("model_results") or []],
        )


@dataclass(frozen=True)
class ProviderMetrics:
    provider: str
    model: str
    status: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    valid_citation_count: int
    citation_coverage: float
    mean_citation_relevance: Optional[float]
    grounded_term_ratio: Optional[float]
    answer_length: int
    evaluation_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderMetrics":
        return cls(
            provider=data["provider"],
            model=data["model"],
            status=data["status"],
            latency_ms=data["latency_ms"],
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            estimated_cost_usd=data.get("estimated_cost_usd"),
            valid_citation_count=data["valid_citation_count"],
            citation_coverage=data["citation_coverage"],
            mean_citation_relevance=data.get("mean_citation_relevance"),
            grounded_term_ratio=data.get("grounded_term_ratio"),
            answer_length=data["answer_length"],
            evaluation_notes=list(data.get("evaluation_notes") or []),
        )


@dataclass(frozen=True)
class Comparison:
    answer_agreement_score: Optional[float]
    latency_difference_ms: float
    estimated_cost_difference_usd: Optional[float]
    comparison_status: str
    comparison_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Comparison":
        return cls(
            answer_agreement_score=data.get("answer_agreement_score"),
            latency_difference_ms=data["latency_difference_ms"],
            estimated_cost_difference_usd=data.get("estimated_cost_difference_usd"),
            comparison_status=data["comparison_status"],
            comparison_notes=list(data.get("comparison_notes") or []),
        )


@dataclass(frozen=True)
class CompareResult:
    question: str
    query_was_masked: bool
    evidence_count: int
    model_results: List[ModelResult] = field(default_factory=list)
    provider_metrics: List[ProviderMetrics] = field(default_factory=list)
    comparison: Optional[Comparison] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CompareResult":
        return cls(
            question=data["question"],
            query_was_masked=data["query_was_masked"],
            evidence_count=data["evidence_count"],
            model_results=[ModelResult.from_dict(item) for item in data.get("model_results") or []],
            provider_metrics=[ProviderMetrics.from_dict(item) for item in data.get("provider_metrics") or []],
            comparison=Comparison.from_dict(data["comparison"]),
        )


# --- The client itself ---------------------------------------------------------------


class PolicyLensAPIClient:
    """A short-lived HTTP client for one backend base URL.

    Cheap to construct (no persistent connection is opened until a
    request is made); a fresh ``httpx.Client`` is used per call rather
    than held open across Streamlit reruns, keeping this class free of
    shared mutable state that would complicate both Streamlit's rerun
    model and testing.
    """

    def __init__(self, base_url: str, request_timeout_seconds: float, connect_timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=request_timeout_seconds,
            write=request_timeout_seconds,
            pool=request_timeout_seconds,
        )

    # -- Safe, retried requests --------------------------------------------------

    def health(self) -> APIResult[HealthStatus]:
        """`GET /health`. Retried on timeout (bounded) -- never on a hard
        connection refusal, since retrying that immediately rarely helps."""

        def _do() -> httpx.Response:
            with httpx.Client(timeout=self._timeout) as client:
                return client.get(f"{self._base_url}{_HEALTH_PATH}")

        return self._request_with_bounded_retries(_do, HealthStatus.from_dict)

    def search(
        self, query: str, document_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> APIResult[SearchResult]:
        """`POST /api/v1/search`. Read-only and side-effect-free -- safe to
        retry on timeout (bounded)."""

        body: Dict[str, Any] = {"query": query}
        if document_id:
            body["document_id"] = document_id
        if top_k is not None:
            body["top_k"] = top_k

        def _do() -> httpx.Response:
            with httpx.Client(timeout=self._timeout) as client:
                return client.post(f"{self._base_url}{_SEARCH_PATH}", json=body)

        return self._request_with_bounded_retries(_do, SearchResult.from_dict)

    # -- Requests that are never retried automatically ----------------------------

    def upload_document(
        self, filename: str, file_bytes: bytes, content_type: str = "application/pdf"
    ) -> APIResult[DocumentUploadResult]:
        """`POST /api/v1/documents/upload`. Never retried, under any
        failure mode (including timeout) -- a retried upload could index
        the same document twice under different in-flight conditions, or
        silently double a large transfer. A failed upload always requires
        an explicit, user-initiated retry."""

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}{_UPLOAD_PATH}",
                    files={"file": (filename, file_bytes, content_type)},
                )
        except httpx.InvalidURL:
            # Not an `httpx.HTTPError` subclass (it's a bare `Exception`),
            # so it needs its own clause -- a malformed configured base
            # URL (e.g. containing a control character) must still
            # produce a safe, typed error instead of an uncaught crash.
            return APIResult.failure(_configuration_error())
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # A failed/timed-out TCP handshake means "couldn't reach the
            # backend at all" -- distinct from a read timeout below, where
            # the connection succeeded but the response didn't arrive.
            return APIResult.failure(_connection_error())
        except httpx.TimeoutException:
            return APIResult.failure(
                APIError(kind="timeout", message="The upload timed out. Try a smaller file or check your connection.")
            )
        except httpx.HTTPError:
            return APIResult.failure(_network_error())

        return self._parse_response(response, DocumentUploadResult.from_dict)

    def ask(
        self,
        question: str,
        document_id: Optional[str] = None,
        providers: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> APIResult[AnswerResult]:
        """`POST /api/v1/answers`. Not retried -- a lost response after a
        successful server-side call could otherwise cause a duplicate,
        billed request to Anthropic/OpenAI."""

        body: Dict[str, Any] = {"question": question}
        if document_id:
            body["document_id"] = document_id
        if providers:
            body["providers"] = providers
        if top_k is not None:
            body["top_k"] = top_k

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}{_ANSWERS_PATH}", json=body)
        except httpx.InvalidURL:
            return APIResult.failure(_configuration_error())
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return APIResult.failure(_connection_error())
        except httpx.TimeoutException:
            return APIResult.failure(
                APIError(kind="timeout", message="The request timed out waiting for a model response.")
            )
        except httpx.HTTPError:
            return APIResult.failure(_network_error())

        return self._parse_response(response, AnswerResult.from_dict)

    def compare(
        self, question: str, document_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> APIResult[CompareResult]:
        """`POST /api/v1/compare`. Not retried, for the same reason as
        `ask()` -- this also triggers real Anthropic/OpenAI calls
        server-side when external calls are enabled."""

        body: Dict[str, Any] = {"question": question}
        if document_id:
            body["document_id"] = document_id
        if top_k is not None:
            body["top_k"] = top_k

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}{_COMPARE_PATH}", json=body)
        except httpx.InvalidURL:
            return APIResult.failure(_configuration_error())
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return APIResult.failure(_connection_error())
        except httpx.TimeoutException:
            return APIResult.failure(
                APIError(kind="timeout", message="The request timed out waiting for a model response.")
            )
        except httpx.HTTPError:
            return APIResult.failure(_network_error())

        return self._parse_response(response, CompareResult.from_dict)

    # -- Shared internals ----------------------------------------------------------

    def _request_with_bounded_retries(self, make_request, parse) -> APIResult:
        last_timeout_error: Optional[APIError] = None

        for attempt in range(1, _SAFE_REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = make_request()
            except httpx.InvalidURL:
                return APIResult.failure(_configuration_error())
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Never retried: if the TCP handshake itself failed or
                # timed out, the backend simply isn't reachable right now
                # -- retrying immediately hits the same wall and would
                # make an offline backend take 3x as long to report as
                # unreachable. `httpx.ConnectTimeout` is a `TimeoutException`
                # subclass, so it must be caught here, before the generic
                # `except httpx.TimeoutException` below, or it would
                # silently fall into (and exhaust) the retry loop instead.
                return APIResult.failure(_connection_error())
            except httpx.TimeoutException:
                # A *read* (or write/pool) timeout means the connection was
                # established but the response didn't arrive in time --
                # plausibly transient server-side slowness, worth a bounded
                # retry for these side-effect-free requests.
                last_timeout_error = _timeout_error()
                if attempt < _SAFE_REQUEST_MAX_ATTEMPTS:
                    time.sleep(_SAFE_REQUEST_RETRY_BACKOFF_SECONDS * attempt)
                    continue
                return APIResult.failure(last_timeout_error)
            except httpx.HTTPError:
                return APIResult.failure(_network_error())

            return self._parse_response(response, parse)

        # Unreachable in practice (the loop always returns), but keeps the
        # function's control flow explicit rather than implicitly falling
        # off the end.
        return APIResult.failure(last_timeout_error or _timeout_error())

    @staticmethod
    def _parse_response(response: httpx.Response, parse) -> APIResult:
        if response.status_code // 100 == 2:
            try:
                payload = response.json()
            except ValueError:
                return APIResult.failure(
                    APIError(
                        kind="invalid_response",
                        message="The backend returned a response that could not be read.",
                        status_code=response.status_code,
                    )
                )
            try:
                return APIResult.success(parse(payload))
            except (KeyError, TypeError, ValueError, AttributeError):
                # `AttributeError` covers `from_dict` implementations that
                # call `.get(...)` on the payload before any bracket
                # access (e.g. `HealthStatus`, `Comparison`) -- if the
                # backend (or a misbehaving proxy in front of it) ever
                # returns valid JSON that isn't a dict at that point
                # (a bare list/string/number/null), `.get` raises
                # `AttributeError`, not `KeyError`/`TypeError`, and must
                # still be handled as a safe "unexpected shape" result
                # rather than an uncaught crash.
                return APIResult.failure(
                    APIError(
                        kind="invalid_response",
                        message="The backend returned a response in an unexpected shape.",
                        status_code=response.status_code,
                    )
                )

        detail = PolicyLensAPIClient._safe_extract_detail(response)
        kind = "server_error" if response.status_code >= 500 else "client_error"
        return APIResult.failure(APIError(kind=kind, message=detail, status_code=response.status_code))

    @staticmethod
    def _safe_extract_detail(response: httpx.Response) -> str:
        # The backend's error contract (`ErrorResponse`) is always
        # `{"detail": "<client-safe message>"}` -- that message is already
        # designed to be shown to a user, so it's safe to surface directly.
        # Anything else (malformed body, non-dict JSON, no body at all)
        # falls back to a generic, status-code-only message -- never the
        # raw response text.
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail

        return f"The backend returned an error (status {response.status_code})."


def get_api_client(config=None) -> PolicyLensAPIClient:
    """Build a client from frontend configuration.

    Pages call this fresh each run (Streamlit re-executes the script on
    every interaction) rather than caching an instance at import time --
    that keeps this factory trivially monkeypatchable in tests and avoids
    any risk of a stale base URL surviving a config change.
    """

    if config is None:
        from streamlit_app.config import load_config

        config = load_config()

    return PolicyLensAPIClient(
        base_url=config.api_base_url,
        request_timeout_seconds=config.request_timeout_seconds,
        connect_timeout_seconds=config.connect_timeout_seconds,
    )
