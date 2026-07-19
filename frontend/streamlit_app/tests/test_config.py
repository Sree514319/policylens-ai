"""Tests for `config.py`'s environment parsing and base-URL validation."""

import pytest

from streamlit_app.config import DEFAULT_API_BASE_URL, DEFAULT_REQUEST_TIMEOUT_SECONDS, load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("POLICYLENS_API_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", raising=False)


def test_defaults_when_unset():
    config = load_config()
    assert config.api_base_url == DEFAULT_API_BASE_URL
    assert config.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_valid_http_url_is_used_as_is(monkeypatch):
    monkeypatch.setenv("POLICYLENS_API_BASE_URL", "http://myhost:9000")
    assert load_config().api_base_url == "http://myhost:9000"


def test_valid_https_url_is_used_as_is(monkeypatch):
    monkeypatch.setenv("POLICYLENS_API_BASE_URL", "https://myhost:9443")
    assert load_config().api_base_url == "https://myhost:9443"


def test_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("POLICYLENS_API_BASE_URL", "http://myhost:9000/")
    assert load_config().api_base_url == "http://myhost:9000"


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "localhost:8000",  # missing scheme
        "myhost.example.com",  # missing scheme
        "ftp://myhost:9000",  # wrong scheme
        "ws://myhost:9000",  # wrong scheme
        "http://myhost:9000/\x01evil",  # control character
        "http://myhost\x7f:9000",  # DEL character
    ],
)
def test_invalid_base_url_falls_back_to_default(monkeypatch, bad_value):
    monkeypatch.setenv("POLICYLENS_API_BASE_URL", bad_value)
    assert load_config().api_base_url == DEFAULT_API_BASE_URL


def test_scheme_check_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("POLICYLENS_API_BASE_URL", "HTTP://myhost:9000")
    assert load_config().api_base_url == "HTTP://myhost:9000"


def test_valid_timeout_is_used(monkeypatch):
    monkeypatch.setenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", "45")
    config = load_config()
    assert config.request_timeout_seconds == 45.0


@pytest.mark.parametrize("bad_value", ["", "not a number", "-5", "0"])
def test_invalid_timeout_falls_back_to_default(monkeypatch, bad_value):
    monkeypatch.setenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", bad_value)
    assert load_config().request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_connect_timeout_is_capped_by_request_timeout(monkeypatch):
    monkeypatch.setenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", "2")
    config = load_config()
    assert config.connect_timeout_seconds <= config.request_timeout_seconds
    assert config.connect_timeout_seconds <= 2.0
