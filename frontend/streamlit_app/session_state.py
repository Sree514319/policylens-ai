"""Session-state helpers: what's safe to keep across Streamlit reruns.

Only ever stores document IDs and the already-privacy-safe metadata the
backend returns for an upload (masked preview, counts, PII category
names -- see `api_client.DocumentUploadResult`, which by construction
never carries raw or unmasked text). Deliberately does NOT store:

- raw PDF bytes (never assigned to `st.session_state` anywhere in this
  app -- an uploaded file's bytes live only in the local variable a page
  passes to `api_client.upload_document`, for the duration of that one
  request)
- question/answer/search *response data*: each page's search/ask/compare
  API result lives only within the Streamlit run that produced it; a
  widget interaction or page navigation clears it rather than persisting
  it.

Accurate-behavior note (this app does NOT overclaim widget-level
privacy): the above covers data this module explicitly writes to
`st.session_state`. It does NOT cover Streamlit's OWN automatic
per-widget state -- a submitted question/query normally remains visible
in its `st.text_input`/`st.text_area` box (and held server-side in that
widget's Streamlit-managed state) for the rest of the browser session,
same as any ordinary web form field, until the user changes it, the
widget's key changes, or "Clear session" is used. `widget_key()` below
exists specifically to make "Clear session" (and switching the active
document) able to reset that residual widget state too, by giving
document-scoped/session-scoped widgets a key that changes when either
happens -- Streamlit has no single built-in "reset every widget" API.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from streamlit_app.api_client import DocumentUploadResult

_ACTIVE_DOCUMENT_KEY = "policylens_active_document"
_DOCUMENT_HISTORY_KEY = "policylens_document_history"
_SESSION_GENERATION_KEY = "policylens_session_generation"
_MAX_DOCUMENT_HISTORY = 20


def init_session_state() -> None:
    """Ensure every key this app uses exists, with a safe default.

    Call once near the top of every page. Idempotent and cheap -- never
    overwrites a value that's already set.
    """

    if _ACTIVE_DOCUMENT_KEY not in st.session_state:
        st.session_state[_ACTIVE_DOCUMENT_KEY] = None
    if _DOCUMENT_HISTORY_KEY not in st.session_state:
        st.session_state[_DOCUMENT_HISTORY_KEY] = []
    if _SESSION_GENERATION_KEY not in st.session_state:
        st.session_state[_SESSION_GENERATION_KEY] = 0


def get_session_generation() -> int:
    init_session_state()
    return st.session_state[_SESSION_GENERATION_KEY]


def widget_key(name: str) -> str:
    """A widget key namespaced by the current "session generation".

    Pass this as `key=` to any widget whose stored value should be
    wiped when the user clicks "Clear session", or when the active
    document changes (both bump the generation counter -- see
    `set_active_document`/`clear_session` below). Streamlit has no
    single API to reset arbitrary widget state directly; giving the
    widget a brand-new key each generation makes Streamlit treat it as
    a fresh, never-before-seen widget, which is the standard workaround.
    """

    return f"{name}_{get_session_generation()}"


def _bump_session_generation() -> None:
    init_session_state()
    st.session_state[_SESSION_GENERATION_KEY] = st.session_state[_SESSION_GENERATION_KEY] + 1


def get_active_document() -> Optional[DocumentUploadResult]:
    init_session_state()
    return st.session_state[_ACTIVE_DOCUMENT_KEY]


def get_document_history() -> List[DocumentUploadResult]:
    init_session_state()
    return list(st.session_state[_DOCUMENT_HISTORY_KEY])


def set_active_document(document: DocumentUploadResult) -> None:
    """Record a newly uploaded (or newly selected) document as the active
    one -- REPLACING whatever was active before.

    This is a plain assignment, never a merge or partial update, so
    switching documents can never accidentally keep using a stale
    document_id left over from a previous selection.

    Also recorded in this session's document history (deduplicated by
    `document_id`, most-recent-first, capped) so other pages can offer a
    "pick a previously uploaded document" selector without the user
    re-entering an ID by hand.

    If this actually changes which document is active (a genuine switch,
    not e.g. re-uploading identical content), the session generation is
    bumped -- any widget keyed via `widget_key()` (the document-scope
    selectors on Search/Ask/Compare, and the file uploader itself) resets
    to its fresh default instead of continuing to show a choice/value
    that referred to the previous document.
    """

    init_session_state()
    previous = st.session_state[_ACTIVE_DOCUMENT_KEY]
    st.session_state[_ACTIVE_DOCUMENT_KEY] = document

    history = [doc for doc in st.session_state[_DOCUMENT_HISTORY_KEY] if doc.document_id != document.document_id]
    history.insert(0, document)
    st.session_state[_DOCUMENT_HISTORY_KEY] = history[:_MAX_DOCUMENT_HISTORY]

    if previous is None or previous.document_id != document.document_id:
        _bump_session_generation()


def clear_active_document() -> None:
    """Explicitly deselect the active document (e.g. "search across all
    documents") without touching the upload history."""

    init_session_state()
    if st.session_state[_ACTIVE_DOCUMENT_KEY] is not None:
        st.session_state[_ACTIVE_DOCUMENT_KEY] = None
        _bump_session_generation()


def clear_session() -> None:
    """The app's "Clear session" action.

    Drops every document reference this browser session has
    accumulated, AND bumps the session generation so every
    `widget_key()`-keyed widget (the file uploader, and the
    document-scope selectors on Search/Ask/Compare) resets to its
    fresh default on the next rerun -- a plain document/history reset
    alone would leave an already-selected file or a previously-chosen
    scope option still showing in those widgets, since Streamlit
    widgets keep their own value independently of what this module
    writes to `st.session_state` (see this module's docstring).

    This cannot and does not delete anything from the backend's vector
    store -- it only forgets what this session remembers locally.
    """

    init_session_state()
    st.session_state[_ACTIVE_DOCUMENT_KEY] = None
    st.session_state[_DOCUMENT_HISTORY_KEY] = []
    _bump_session_generation()
