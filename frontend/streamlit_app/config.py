"""Frontend-only configuration.

Deliberately independent of the backend's ``app.core.config.Settings`` --
the frontend talks to the backend exclusively over HTTP (see
``api_client.py``) and must never import backend service/config code
directly. Reads the same root ``.env`` file the backend uses (via
``python-dotenv``) so one file configures both processes, but only reads
the two environment variables that are actually this frontend's concern.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0

_ENV_LOADED = False
_URL_SCHEME_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def _ensure_env_loaded() -> None:
    # `load_dotenv()` searches the current directory and its parents for a
    # `.env` file -- safe to call more than once (idempotent), but guarded
    # here so repeated Streamlit reruns don't re-read the file every time.
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _parse_positive_float(raw: str, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _validate_base_url(raw: str) -> str:
    """Reject a base URL that would later crash `httpx` with an
    `InvalidURL` (or simply never connect), falling back to the safe
    default instead of passing a malformed value through unvalidated.

    Bounds: must be non-empty, must start with an explicit ``http://``
    or ``https://`` scheme (so an operator typo like "localhost:8000"
    fails fast and obviously here rather than as a confusing connection
    error later), and must not contain control characters (which is
    exactly what triggers `httpx.InvalidURL`).
    """

    candidate = raw.strip()
    if not candidate:
        return DEFAULT_API_BASE_URL
    if not _URL_SCHEME_PATTERN.match(candidate):
        return DEFAULT_API_BASE_URL
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in candidate):
        return DEFAULT_API_BASE_URL
    return candidate.rstrip("/")


@dataclass(frozen=True)
class FrontendConfig:
    """Immutable, process-wide frontend configuration."""

    api_base_url: str
    request_timeout_seconds: float
    connect_timeout_seconds: float


def load_config() -> FrontendConfig:
    """Read frontend configuration from the environment.

    Never raises: a missing or malformed value silently falls back to a
    safe default rather than crashing the whole app over a bad `.env`.
    """

    _ensure_env_loaded()

    raw_base_url = os.environ.get("POLICYLENS_API_BASE_URL", "")
    base_url = _validate_base_url(raw_base_url)

    raw_timeout = os.environ.get("FRONTEND_REQUEST_TIMEOUT_SECONDS", "")
    timeout = _parse_positive_float(raw_timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
    connect_timeout = min(DEFAULT_CONNECT_TIMEOUT_SECONDS, timeout)

    return FrontendConfig(
        api_base_url=base_url,
        request_timeout_seconds=timeout,
        connect_timeout_seconds=connect_timeout,
    )
