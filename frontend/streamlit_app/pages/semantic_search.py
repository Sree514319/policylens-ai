"""Semantic Search page.

Search *results* (the API response) are only ever held in a local
variable for the run that produced them -- they are not written to
`st.session_state`, so navigating away or triggering an unrelated widget
clears them rather than accumulating a search history.

Accurate-behavior note: the submitted query *text* itself is a different
matter -- like any web form, the text input widget below keeps showing
what was typed (and Streamlit holds that value server-side) until it's
changed, its key changes, or "Clear session" is used. See
`session_state.py`'s module docstring for why, and `widget_key()` for
how "Clear session"/switching documents resets it here.
"""

import sys
from pathlib import Path

# Streamlit execs each page as its own script -- this page needs the
# same `streamlit_app`-importability fix as `app.py` independently (see
# the full explanation there), since it can run on its own (e.g. under
# test) without `app.py` having run first in the same process.
_here = Path(__file__).resolve()
for _candidate in (_here.parent, *_here.parents):
    if (_candidate / "streamlit_app").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import streamlit as st  # noqa: E402

from streamlit_app.api_client import get_api_client  # noqa: E402
from streamlit_app.components.render import (  # noqa: E402
    render_api_error,
    render_masked_query_notice,
    render_search_result_card,
)
from streamlit_app.session_state import (  # noqa: E402
    get_active_document,
    get_document_history,
    init_session_state,
    widget_key,
)

init_session_state()

st.title("🔍 Semantic Search")
st.caption("Search indexed document chunks by meaning, not just keywords.")

active_document = get_active_document()
history = get_document_history()

with st.form("semantic_search_form"):
    query = st.text_input("Search query", placeholder="e.g. overdraft fee schedule", key=widget_key("search_query"))

    scope_options = {"All documents": None}
    for document in history:
        scope_options[f"{document.filename} ({document.document_id[:8]}...)"] = document.document_id
    default_index = 0
    if active_document is not None:
        for index, (label, doc_id) in enumerate(scope_options.items()):
            if doc_id == active_document.document_id:
                default_index = index
                break
    scope_label = st.selectbox(
        "Scope", options=list(scope_options.keys()), index=default_index, key=widget_key("search_scope")
    )

    top_k = st.slider(
        "Number of results (top_k)", min_value=1, max_value=50, value=5, key=widget_key("search_top_k")
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a search query first.")
    else:
        client = get_api_client()
        with st.spinner("Searching..."):
            result = client.search(query=query, document_id=scope_options[scope_label], top_k=top_k)

        if result.ok:
            render_masked_query_notice(result.data.query, result.data.query_was_masked)
            st.divider()
            if result.data.result_count == 0:
                st.info("No matching results. Try a different query, or upload a document first.")
            else:
                st.caption(f"{result.data.result_count} result(s), ranked by relevance.")
                for index, item in enumerate(result.data.results, start=1):
                    render_search_result_card(item, index)
        else:
            render_api_error(result.error)
