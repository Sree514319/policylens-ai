"""Upload Document page.

Raw PDF bytes only ever exist in a local variable for the duration of
the upload button's click handler below -- they are passed straight to
`api_client.upload_document` and never assigned to `st.session_state`,
cached, or logged. Only the backend's already-masked, already-summarized
`DocumentUploadResult` is ever stored (see `session_state.py`).
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
from streamlit_app.components.render import render_api_error, render_document_summary  # noqa: E402
from streamlit_app.session_state import (  # noqa: E402
    get_active_document,
    init_session_state,
    set_active_document,
    widget_key,
)

init_session_state()

st.title("📤 Upload Document")
st.caption("PDF only. Text is extracted in memory, masked for PII, then chunked and indexed.")

st.info(
    "🔒 Use only **synthetic or public sample** banking policy documents. "
    "Local PII masking runs automatically, but it is best-effort — see About & Limitations."
)

# Keyed by the current session generation so "Clear session" (and a
# successful upload switching the active document) resets this widget --
# otherwise a previously-selected file would keep showing as "selected"
# indefinitely, since Streamlit widgets retain their own value
# independently of anything this app writes to st.session_state.
uploaded_file = st.file_uploader(
    "Choose a PDF file", type=["pdf"], accept_multiple_files=False, key=widget_key("pdf_uploader")
)

if uploaded_file is not None:
    # Read once, into a local variable reused for both the size display
    # and the upload call below -- avoids a redundant second read of the
    # same buffer. This local `file_bytes` is the only reference this
    # application's own code ever holds to the raw content; it goes out
    # of scope (eligible for garbage collection) once this script run
    # ends, and is never assigned to `st.session_state`, cached, or
    # logged. Streamlit's own file_uploader widget necessarily keeps the
    # bytes in its internal state for as long as the widget continues to
    # show this file as selected -- this app cannot control that, but the
    # generation-keyed widget above means a successful upload (or "Clear
    # session") resets the widget and releases that reference promptly.
    file_bytes = uploaded_file.getvalue()
    size_kb = len(file_bytes) / 1024
    st.markdown(f"**Selected file:** {uploaded_file.name} ({size_kb:,.1f} KB)")

    if st.button("Upload and process", type="primary"):
        client = get_api_client()
        with st.spinner("Uploading and processing — extracting text, masking PII, chunking, indexing..."):
            result = client.upload_document(
                filename=uploaded_file.name,
                file_bytes=file_bytes,
                content_type=uploaded_file.type or "application/pdf",
            )

        if result.ok:
            set_active_document(result.data)
            st.success(f"✅ Uploaded and indexed: {result.data.filename}")
        else:
            render_api_error(result.error)

active_document = get_active_document()
if active_document is not None:
    st.divider()
    st.subheader("Active document")
    st.caption("This document is now the default scope on Search, Ask Models, and Compare Models.")
    render_document_summary(active_document)
