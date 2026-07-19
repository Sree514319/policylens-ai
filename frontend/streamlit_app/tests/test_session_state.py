"""Tests for `session_state.py`.

`st.session_state` supports plain dict-style access
(`in`/`__getitem__`/`__setitem__`), which is all `session_state.py` ever
uses -- so a plain `dict` monkeypatched in for `streamlit.session_state`
is a faithful, fast, Streamlit-runtime-free stand-in for these tests.
"""

import streamlit as st

from streamlit_app import session_state as ss
from streamlit_app.api_client import DocumentUploadResult


def _document(document_id="doc-1", filename="a.pdf", **overrides):
    defaults = dict(
        document_id=document_id,
        filename=filename,
        page_count=1,
        character_count=100,
        status="processed",
        preview="Some masked preview text.",
        chunk_count=1,
        pages_with_text=1,
        indexed_chunk_count=1,
        pii_detected=False,
        pii_entity_count=0,
        pii_categories=[],
    )
    defaults.update(overrides)
    return DocumentUploadResult(**defaults)


def _fake_session_state(monkeypatch):
    fake_state = {}
    monkeypatch.setattr(st, "session_state", fake_state)
    return fake_state


def test_init_session_state_sets_safe_defaults(monkeypatch):
    _fake_session_state(monkeypatch)

    ss.init_session_state()

    assert ss.get_active_document() is None
    assert ss.get_document_history() == []


def test_init_session_state_does_not_overwrite_existing_values(monkeypatch):
    fake_state = _fake_session_state(monkeypatch)
    ss.init_session_state()
    document = _document()
    ss.set_active_document(document)

    ss.init_session_state()  # called again, as every page does

    assert ss.get_active_document() is document


def test_set_active_document_replaces_not_merges(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    first = _document(document_id="doc-1", filename="first.pdf")
    second = _document(document_id="doc-2", filename="second.pdf")

    ss.set_active_document(first)
    assert ss.get_active_document().document_id == "doc-1"

    ss.set_active_document(second)

    # Switching documents must fully replace the active one -- never keep
    # using the previous document_id alongside or instead of the new one.
    active = ss.get_active_document()
    assert active.document_id == "doc-2"
    assert active.filename == "second.pdf"


def test_set_active_document_records_history_most_recent_first(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    ss.set_active_document(_document(document_id="doc-1"))
    ss.set_active_document(_document(document_id="doc-2"))

    history = ss.get_document_history()
    assert [doc.document_id for doc in history] == ["doc-2", "doc-1"]


def test_set_active_document_deduplicates_history_by_document_id(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    ss.set_active_document(_document(document_id="doc-1", filename="v1.pdf"))
    ss.set_active_document(_document(document_id="doc-2"))
    # Re-selecting doc-1 (e.g. re-uploading identical content) must not
    # create a second, stale history entry.
    ss.set_active_document(_document(document_id="doc-1", filename="v1-reuploaded.pdf"))

    history = ss.get_document_history()
    assert [doc.document_id for doc in history] == ["doc-1", "doc-2"]
    assert history[0].filename == "v1-reuploaded.pdf"


def test_document_history_is_capped(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    for i in range(ss._MAX_DOCUMENT_HISTORY + 5):
        ss.set_active_document(_document(document_id=f"doc-{i}"))

    assert len(ss.get_document_history()) == ss._MAX_DOCUMENT_HISTORY
    # Most recent survives the cap.
    assert ss.get_document_history()[0].document_id == f"doc-{ss._MAX_DOCUMENT_HISTORY + 4}"


def test_clear_active_document_leaves_history_intact(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document(document_id="doc-1"))

    ss.clear_active_document()

    assert ss.get_active_document() is None
    assert len(ss.get_document_history()) == 1


def test_clear_session_removes_active_document_and_history(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document(document_id="doc-1"))
    ss.set_active_document(_document(document_id="doc-2"))

    ss.clear_session()

    assert ss.get_active_document() is None
    assert ss.get_document_history() == []


def test_get_document_history_returns_a_copy_not_the_live_list(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document(document_id="doc-1"))

    history = ss.get_document_history()
    history.append(_document(document_id="doc-injected"))

    # Mutating the returned list must not affect what's actually stored.
    assert len(ss.get_document_history()) == 1


# --- Privacy: only safe, already-masked metadata is ever stored ----------------------


def test_active_document_never_carries_raw_pdf_bytes_or_full_text(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    document = _document(preview="Masked preview, capped at ~200 chars.")
    ss.set_active_document(document)

    stored = ss.get_active_document()
    # DocumentUploadResult has no field for raw bytes or full text at
    # all -- this locks in that invariant at the type level, not just by
    # checking values.
    stored_fields = set(vars(stored).keys())
    assert "raw_bytes" not in stored_fields
    assert "file_bytes" not in stored_fields
    assert "full_text" not in stored_fields
    assert stored_fields == {
        "document_id",
        "filename",
        "page_count",
        "character_count",
        "status",
        "preview",
        "chunk_count",
        "pages_with_text",
        "indexed_chunk_count",
        "pii_detected",
        "pii_entity_count",
        "pii_categories",
    }


def test_session_state_module_never_stores_raw_bytes_type(monkeypatch):
    # Defensive: even if a caller tried to sneak raw bytes in as the
    # "document", set_active_document only ever assigns it verbatim into
    # session_state (it doesn't accept or extract bytes) -- this test
    # documents that the only accepted shape is DocumentUploadResult by
    # asserting the stored object is never a `bytes` instance.
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document())

    assert not isinstance(ss.get_active_document(), (bytes, bytearray))


# --- Session-generation widget-key mechanism ("Clear session" / document switching) ---


def test_widget_key_is_stable_within_the_same_generation(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    assert ss.widget_key("pdf_uploader") == ss.widget_key("pdf_uploader")


def test_widget_key_differs_across_generations(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    key_before = ss.widget_key("pdf_uploader")
    ss.clear_session()
    key_after = ss.widget_key("pdf_uploader")

    assert key_before != key_after


def test_clear_session_bumps_the_generation_exactly_once(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    generation_before = ss.get_session_generation()

    ss.clear_session()

    assert ss.get_session_generation() == generation_before + 1


def test_switching_to_a_different_document_bumps_the_generation(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document(document_id="doc-1"))
    generation_after_first = ss.get_session_generation()

    ss.set_active_document(_document(document_id="doc-2"))

    assert ss.get_session_generation() == generation_after_first + 1


def test_reuploading_the_same_document_does_not_bump_the_generation(monkeypatch):
    # Re-uploading identical content (same document_id) is not a
    # "switch" -- document-scoped widgets showing that same document
    # don't need to reset.
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    ss.set_active_document(_document(document_id="doc-1", filename="v1.pdf"))
    generation_after_first = ss.get_session_generation()

    ss.set_active_document(_document(document_id="doc-1", filename="v1.pdf"))

    assert ss.get_session_generation() == generation_after_first


def test_first_upload_of_the_session_bumps_the_generation(monkeypatch):
    # Going from "no active document" to "an active document" is also a
    # meaningful switch -- e.g. a scope selector that defaulted to "All
    # documents" should reset to reflect the newly-available document.
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    generation_before = ss.get_session_generation()

    ss.set_active_document(_document(document_id="doc-1"))

    assert ss.get_session_generation() == generation_before + 1


def test_clear_active_document_bumps_generation_only_if_something_was_active(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()
    generation_before = ss.get_session_generation()

    # Nothing was active -- no-op, no generation bump.
    ss.clear_active_document()
    assert ss.get_session_generation() == generation_before

    ss.set_active_document(_document(document_id="doc-1"))
    generation_with_active_doc = ss.get_session_generation()

    ss.clear_active_document()
    assert ss.get_session_generation() == generation_with_active_doc + 1


def test_clear_session_resets_the_shared_generation_widget_keys_rely_on(monkeypatch):
    _fake_session_state(monkeypatch)
    ss.init_session_state()

    key_before = ss.widget_key("any_widget_name")
    ss.clear_session()
    key_after = ss.widget_key("any_widget_name")

    assert key_before != key_after


def test_every_page_actually_uses_widget_key_for_its_stateful_inputs():
    # A regression guard at the source level: it's not enough for
    # widget_key() itself to work correctly (see the tests above) -- each
    # page must actually call it for its file_uploader/text_input/
    # text_area/selectbox/multiselect/slider, or "Clear session" silently
    # fails to reset that specific widget. Checks the real page files,
    # not a hand-maintained list, so a page added later without wiring
    # this in is caught here too.
    from pathlib import Path

    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    stateful_widget_calls = ("st.file_uploader(", "st.text_input(", "st.text_area(", "st.selectbox(", "st.multiselect(", "st.slider(")

    for page_file in pages_dir.glob("*.py"):
        source = page_file.read_text(encoding="utf-8")
        widget_call_count = sum(source.count(call) for call in stateful_widget_calls)
        widget_key_call_count = source.count("widget_key(")
        if widget_call_count == 0:
            continue
        assert widget_key_call_count >= widget_call_count, (
            f"{page_file.name} creates {widget_call_count} stateful input widget(s) but only "
            f"{widget_key_call_count} use widget_key() -- 'Clear session'/document-switching "
            "would not reset the rest."
        )
