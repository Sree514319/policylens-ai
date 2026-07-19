"""Streamlit `AppTest`-based integration/smoke tests.

Every HTTP call these pages would make is mocked via
`install_mock_transport` (see conftest.py) -- these tests never touch a
real network, and never call a live LLM provider.
"""

import httpx
from streamlit.testing.v1 import AppTest

from streamlit_app.tests.sample_payloads import (
    SAMPLE_ANSWER_RESPONSE,
    SAMPLE_COMPARE_RESPONSE,
    SAMPLE_HEALTH,
    SAMPLE_SEARCH_RESPONSE,
    SAMPLE_UPLOAD_RESPONSE,
)

PAGES_DIR = "frontend/streamlit_app/pages"


def _json_handler(payload, status_code=200):
    return lambda request: httpx.Response(status_code, json=payload)


# --- Full-app smoke test ---------------------------------------------------------------


def test_app_entrypoint_loads_without_exception(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_HEALTH))

    at = AppTest.from_file("frontend/streamlit_app/app.py")
    at.run(timeout=30)

    assert at.exception == []


# --- Backend-offline UI ------------------------------------------------------------------


def test_home_page_handles_backend_offline_without_crashing(install_mock_transport):
    def handler(request):
        raise httpx.ConnectError("simulated: nothing listening")

    install_mock_transport(handler)

    at = AppTest.from_file(f"{PAGES_DIR}/home.py")
    at.run(timeout=15)

    assert at.exception == []
    assert any("unreachable" in w.value.lower() for w in at.warning)


def test_home_page_shows_connected_status_when_backend_is_healthy(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_HEALTH))

    at = AppTest.from_file(f"{PAGES_DIR}/home.py")
    at.run(timeout=15)

    assert at.exception == []
    assert any("connected" in s.value.lower() for s in at.success)


def test_other_pages_load_without_a_backend_at_all(install_mock_transport):
    # Upload/Search/Ask/Compare/About don't call the backend until a user
    # submits a form -- they must all still render on initial load even
    # with the backend completely unreachable.
    def handler(request):
        raise httpx.ConnectError("simulated: nothing listening")

    install_mock_transport(handler)

    for page in ["upload_document", "semantic_search", "ask_models", "compare_models", "about_limitations"]:
        at = AppTest.from_file(f"{PAGES_DIR}/{page}.py")
        at.run(timeout=15)
        assert at.exception == [], f"{page} raised an exception while the backend was offline"


# --- Upload: session-state safety -------------------------------------------------------


def test_successful_upload_updates_session_state_with_safe_metadata_only(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_UPLOAD_RESPONSE, status_code=201))

    at = AppTest.from_file(f"{PAGES_DIR}/upload_document.py")
    at.run(timeout=15)

    at.file_uploader[0].upload("bank_policy.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")
    at.run(timeout=15)

    # Click the "Upload and process" button (only button on this page at
    # this point in the flow).
    assert len(at.button) >= 1
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    active_document = at.session_state["policylens_active_document"]
    assert active_document is not None
    assert active_document.document_id == SAMPLE_UPLOAD_RESPONSE["document_id"]
    assert active_document.pii_detected is True
    assert active_document.pii_categories == ["EMAIL", "SSN"]


def test_raw_pdf_bytes_are_never_retained_in_session_state_after_upload(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_UPLOAD_RESPONSE, status_code=201))

    at = AppTest.from_file(f"{PAGES_DIR}/upload_document.py")
    at.run(timeout=15)
    at.file_uploader[0].upload("bank_policy.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)

    # Walk every value stored in session_state -- none of them may be (or
    # contain) the raw uploaded bytes, and no key may hold a bytes object
    # at all.
    for key, value in at.session_state.filtered_state.items():
        assert not isinstance(value, (bytes, bytearray)), f"session_state[{key!r}] holds raw bytes"
        assert b"%PDF-1.4 fake pdf content" not in repr(value).encode("utf-8", errors="ignore")


def test_upload_failure_shows_a_safe_error_and_does_not_set_active_document(install_mock_transport):
    install_mock_transport(_json_handler({"detail": "The uploaded file could not be read as a valid PDF."}, status_code=422))

    at = AppTest.from_file(f"{PAGES_DIR}/upload_document.py")
    at.run(timeout=15)
    at.file_uploader[0].upload("broken.pdf", b"not a real pdf", "application/pdf")
    at.run(timeout=15)
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert at.session_state["policylens_active_document"] is None
    assert any("could not be read" in e.value for e in at.error)


# --- Search: masked-query display, results, empty/error states --------------------------


def test_search_renders_masked_query_notice_when_pii_was_masked(install_mock_transport):
    masked_response = {**SAMPLE_SEARCH_RESPONSE, "query": "SSN [SSN_REDACTED]", "query_was_masked": True}
    install_mock_transport(_json_handler(masked_response))

    at = AppTest.from_file(f"{PAGES_DIR}/semantic_search.py")
    at.run(timeout=15)
    at.text_input[0].set_value("SSN 123-45-6789")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("masked" in info.value.lower() for info in at.info)
    assert any("[SSN_REDACTED]" in t.value for t in at.text)


def test_search_renders_no_results_state(install_mock_transport):
    empty_response = {"query": "nothing matches", "query_was_masked": False, "result_count": 0, "results": []}
    install_mock_transport(_json_handler(empty_response))

    at = AppTest.from_file(f"{PAGES_DIR}/semantic_search.py")
    at.run(timeout=15)
    at.text_input[0].set_value("nothing matches")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("no matching results" in info.value.lower() for info in at.info)


def test_search_renders_error_state_safely(install_mock_transport):
    install_mock_transport(_json_handler({"detail": "The vector store is currently unavailable."}, status_code=503))

    at = AppTest.from_file(f"{PAGES_DIR}/semantic_search.py")
    at.run(timeout=15)
    at.text_input[0].set_value("anything")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("currently unavailable" in e.value for e in at.error)


# --- Ask Models: success / insufficient / error / partial failure -----------------------


def test_ask_models_renders_mixed_success_and_error_results(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_ANSWER_RESPONSE))

    at = AppTest.from_file(f"{PAGES_DIR}/ask_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the overdraft fee?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    # The successful Anthropic answer's text is present verbatim.
    assert any("$35 per occurrence" in t.value for t in at.text)
    # The failed OpenAI result's safe error message is shown.
    assert any("timed out" in e.value.lower() for e in at.error)
    # A partial-failure summary is shown since statuses differ.
    assert any("did not" in info.value for info in at.info)


def test_ask_models_insufficient_evidence_state(install_mock_transport):
    insufficient_response = {
        "question": "What is the meaning of life?",
        "query_was_masked": False,
        "evidence_count": 0,
        "model_results": [
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "status": "insufficient_evidence",
                "answer": "The available evidence does not contain enough information to answer this question.",
                "citations": [],
                "latency_ms": 0.0,
                "input_tokens": None,
                "output_tokens": None,
                "error": None,
            }
        ],
    }
    install_mock_transport(_json_handler(insufficient_response))

    at = AppTest.from_file(f"{PAGES_DIR}/ask_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the meaning of life?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("does not contain enough information" in w.value for w in at.warning)


def test_ask_models_explains_when_external_calls_are_disabled(install_mock_transport):
    disabled_response = {
        "question": "What is the overdraft fee?",
        "query_was_masked": False,
        "evidence_count": 1,
        "model_results": [
            {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "status": "error",
                "answer": "",
                "citations": [],
                "latency_ms": 0.0,
                "input_tokens": None,
                "output_tokens": None,
                "error": "External LLM calls are disabled by server configuration (ALLOW_EXTERNAL_LLM_CALLS=false).",
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "status": "error",
                "answer": "",
                "citations": [],
                "latency_ms": 0.0,
                "input_tokens": None,
                "output_tokens": None,
                "error": "External LLM calls are disabled by server configuration (ALLOW_EXTERNAL_LLM_CALLS=false).",
            },
        ],
    }
    install_mock_transport(_json_handler(disabled_response))

    at = AppTest.from_file(f"{PAGES_DIR}/ask_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the overdraft fee?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("ALLOW_EXTERNAL_LLM_CALLS" in info.value for info in at.info)


# --- Compare Models: no winner/accuracy language, null metrics --------------------------


_FORBIDDEN_WORDS = ["winner", "accuracy", "won", "best model", "correct answer"]


def test_compare_page_never_renders_winner_or_accuracy_language(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_COMPARE_RESPONSE))

    at = AppTest.from_file(f"{PAGES_DIR}/compare_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the overdraft fee?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []

    # `st.caption` is excluded here: this page's own authored disclaimer
    # ("there is no overall 'winner' and no accuracy score") deliberately
    # uses the word "winner" *in negation* to state the no-winner policy
    # -- captions on this page never contain dynamic provider/model
    # content, only our own static UI copy or backend diagnostic notes
    # (see `render_comparison_summary`/`render_provider_metrics`). The
    # scan below covers every element that *could* carry model- or
    # data-driven text: answers, citations, metric values/labels, and
    # comparison notes.
    all_rendered_text = []
    for collection in (at.markdown, at.text, at.info, at.success, at.warning, at.error, at.subheader, at.title):
        all_rendered_text.extend(element.value for element in collection)
    for metric in at.metric:
        all_rendered_text.append(str(metric.label))
        all_rendered_text.append(str(metric.value))

    joined = " ".join(all_rendered_text).lower()
    for forbidden in _FORBIDDEN_WORDS:
        assert forbidden not in joined, f"found forbidden word {forbidden!r} on the Compare Models page"

    # The one legitimate, expected use of "winner" on this page is the
    # explicit negation in the static disclaimer caption -- confirmed
    # here rather than just silently excluded, so a regression that
    # removes the disclaimer (or that starts declaring a real winner
    # somewhere unexpected) would still be caught.
    caption_text = " ".join(c.value.lower() for c in at.caption)
    assert 'no overall "winner"' in caption_text


def test_compare_page_handles_null_metrics_without_fake_zeroes(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_COMPARE_RESPONSE))

    at = AppTest.from_file(f"{PAGES_DIR}/compare_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the overdraft fee?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    # openai's estimated_cost_usd and grounded_term_ratio are null in the
    # sample payload -- they must render as "Not available", never as a
    # fabricated "$0.00" or "0%".
    metric_values = [str(metric.value) for metric in at.metric]
    assert "Not available" in metric_values


def test_compare_page_shows_comparison_status_and_notes(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_COMPARE_RESPONSE))

    at = AppTest.from_file(f"{PAGES_DIR}/compare_models.py")
    at.run(timeout=15)
    at.text_area[0].set_value("What is the overdraft fee?")
    at.button[0].click().run(timeout=15)

    assert at.exception == []
    assert any("both models answered successfully" in m.value.lower() for m in at.markdown)


# --- Clear session ------------------------------------------------------------------------


def test_clear_session_button_resets_document_state(install_mock_transport):
    install_mock_transport(_json_handler(SAMPLE_HEALTH))

    from streamlit_app.api_client import DocumentUploadResult

    stub_document = DocumentUploadResult.from_dict(SAMPLE_UPLOAD_RESPONSE)

    at = AppTest.from_file("frontend/streamlit_app/app.py")
    # Seed state as if a document was already active from a prior page visit.
    at.session_state["policylens_active_document"] = stub_document
    at.session_state["policylens_document_history"] = [stub_document]
    at.run(timeout=30)

    assert at.session_state["policylens_active_document"] is not None

    clear_button = next(button for button in at.button if "Clear session" in button.label)
    clear_button.click().run(timeout=30)

    assert at.exception == []
    assert at.session_state["policylens_active_document"] is None
    assert at.session_state["policylens_document_history"] == []


def test_successful_upload_resets_the_file_uploader_widget(install_mock_transport):
    # A successful upload bumps the session generation (see
    # session_state.set_active_document), which changes the
    # file_uploader's key -- Streamlit then treats it as a brand-new
    # widget with no file selected, rather than continuing to show the
    # just-uploaded file as still "selected" indefinitely.
    install_mock_transport(_json_handler(SAMPLE_UPLOAD_RESPONSE, status_code=201))

    at = AppTest.from_file(f"{PAGES_DIR}/upload_document.py")
    at.run(timeout=15)
    at.file_uploader[0].upload("bank_policy.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")
    at.run(timeout=15)
    assert at.file_uploader[0].value is not None

    at.button[0].click().run(timeout=15)
    assert at.exception == []

    # A further rerun (e.g. the user interacting with anything else)
    # re-executes the page with the new generation's widget key.
    at.run(timeout=15)
    assert at.file_uploader[0].value is None


def test_uploading_a_second_different_document_resets_a_manually_chosen_search_scope(install_mock_transport):
    # Prove the generation bump actually overrides a *manual* prior
    # selection, not just the widget's initial default (which would
    # trivially "point at the new document" on first render regardless).
    from streamlit_app.api_client import DocumentUploadResult

    first_document = DocumentUploadResult.from_dict(SAMPLE_UPLOAD_RESPONSE)
    second_payload = {**SAMPLE_UPLOAD_RESPONSE, "document_id": "b" * 64, "filename": "second.pdf"}

    install_mock_transport(_json_handler(SAMPLE_SEARCH_RESPONSE))

    at = AppTest.from_file(f"{PAGES_DIR}/semantic_search.py")
    at.session_state["policylens_active_document"] = first_document
    at.session_state["policylens_document_history"] = [first_document]
    at.run(timeout=15)

    # Manually select "All documents" (index 0), overriding the default
    # (which would point at `first_document`).
    at.selectbox[0].select_index(0).run(timeout=15)
    assert at.selectbox[0].value == "All documents"

    # Now a second document becomes active (simulating a fresh upload on
    # the Upload Document page during this same session).
    from streamlit_app.session_state import set_active_document

    import streamlit as st

    st.session_state = at.session_state
    set_active_document(DocumentUploadResult.from_dict(second_payload))

    at.run(timeout=15)

    # The scope selector must reflect the *new* active document by
    # default again -- not still show the user's earlier "All documents"
    # choice, and not silently still refer to the first document either.
    assert "second.pdf" in at.selectbox[0].value


# --- No sensitive Streamlit caching --------------------------------------------------


def test_no_page_uses_cache_data_or_cache_resource():
    from pathlib import Path

    frontend_root = Path(__file__).resolve().parents[1]
    for python_file in frontend_root.rglob("*.py"):
        if "tests" in python_file.parts:
            continue
        source = python_file.read_text(encoding="utf-8")
        assert "st.cache_data" not in source, f"{python_file} uses st.cache_data"
        assert "st.cache_resource" not in source, f"{python_file} uses st.cache_resource"
        assert "@cache" not in source, f"{python_file} uses a bare @cache decorator"


# --- Installed Streamlit API compatibility --------------------------------------------


def test_installed_streamlit_supports_every_api_this_app_uses():
    import inspect

    import streamlit as st

    assert hasattr(st, "navigation"), "st.navigation is required (multipage nav in app.py)"
    assert hasattr(st, "Page"), "st.Page is required (multipage nav in app.py)"

    container_params = inspect.signature(st.container).parameters
    assert "border" in container_params, "st.container(border=True) is used in components/render.py"

    button_params = inspect.signature(st.button).parameters
    assert "width" in button_params, "st.button(width=...) is used in app.py's sidebar"

    chart_params = inspect.signature(st.bar_chart).parameters
    assert "x_label" in chart_params and "y_label" in chart_params, (
        "st.bar_chart(x_label=..., y_label=...) is used in pages/compare_models.py"
    )

    form_submit_params = inspect.signature(st.form_submit_button).parameters
    assert "type" in form_submit_params


def test_app_entrypoint_runs_end_to_end_on_the_installed_streamlit_version(install_mock_transport):
    # A broader, behavioral companion to the signature-only check above:
    # if any used API were actually incompatible, the full app wouldn't
    # load without raising.
    install_mock_transport(_json_handler(SAMPLE_HEALTH))

    at = AppTest.from_file("frontend/streamlit_app/app.py")
    at.run(timeout=30)

    assert at.exception == []


# --- Runtime imports from the repository root (matches the documented run command) ----


def test_page_paths_resolve_relative_to_the_entrypoint_not_the_process_cwd(install_mock_transport, tmp_path, monkeypatch):
    # st.Page("pages/home.py", ...) in app.py must resolve relative to
    # app.py's own directory (per Streamlit's documented behavior), not
    # the interpreter's current working directory -- otherwise the
    # documented run command (`streamlit run frontend/streamlit_app/app.py`
    # from the repository root) would only work by coincidence of CWD.
    install_mock_transport(_json_handler(SAMPLE_HEALTH))

    monkeypatch.chdir(tmp_path)  # deliberately NOT the repository root

    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    app_path = repo_root / "frontend" / "streamlit_app" / "app.py"

    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    assert at.exception == []


def test_streamlit_run_from_the_repository_root_serves_the_app(monkeypatch):
    # A genuine subprocess check mirroring the documented command exactly
    # (`.venv/Scripts/python.exe -m streamlit run frontend/streamlit_app/app.py`
    # from the repo root) -- proves the whole process (not just AppTest's
    # in-process script execution) can locate and serve every page.
    import socket
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/streamlit_app/app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        last_error = None
        response_text = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2) as response:
                    response_text = response.read().decode("utf-8", errors="ignore")
                break
            except Exception as exc:  # noqa: BLE001 -- polling until the server is up
                last_error = exc
                time.sleep(1)

        assert response_text is not None, f"streamlit run never became reachable: {last_error}"
        assert "streamlit" in response_text.lower()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
