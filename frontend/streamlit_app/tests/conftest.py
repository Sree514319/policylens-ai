"""Shared fixtures for frontend tests.

No test in this package ever makes a real network call: every HTTP
interaction goes through `httpx.MockTransport`, installed via
`install_mock_transport` below.
"""

import httpx
import pytest


@pytest.fixture
def install_mock_transport(monkeypatch):
    """Returns a function; call it with `handler(request) -> httpx.Response`
    to make every `httpx.Client(...)` constructed for the rest of this
    test use that handler instead of touching the network.
    """

    def _install(handler):
        real_client_cls = httpx.Client

        def _mock_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client_cls(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", _mock_client)

    return _install


@pytest.fixture
def call_log():
    """A plain list a mock handler can append to, so a test can assert on
    call count (e.g. "upload was never retried") and inspect each
    request actually sent."""

    return []
