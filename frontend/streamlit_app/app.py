"""PolicyLens AI -- Streamlit frontend entrypoint.

Run with:
    streamlit run frontend/streamlit_app/app.py

This script only wires up page configuration, session-state
initialization, and navigation -- all actual page content lives in
`pages/*.py`, and all backend communication goes through
`api_client.py` (HTTP only; this app never imports backend service code).
"""

import sys
from pathlib import Path

# Streamlit's script runner never adds this file's own directory (let
# alone its parents) to `sys.path` -- it just execs the target file more
# or less as-is. That means `streamlit_app`, the package this very file
# belongs to, is NOT importable by default under
# `streamlit run frontend/streamlit_app/app.py`, and the
# `from streamlit_app...` imports below would fail with
# `ModuleNotFoundError: No module named 'streamlit_app'`.
#
# Fix: walk up from this file to find the directory that *contains* the
# `streamlit_app` package (i.e. `frontend/`) and add it to `sys.path`,
# before any `streamlit_app` import runs. Uses `__file__`, never the
# process's current working directory, so this works regardless of
# where `streamlit run`/`pytest` was invoked from, on any OS.
#
# Every page in `pages/` carries this exact same fix independently (not
# just a "run once in app.py" fix) -- each page is its own script that
# Streamlit can execute on its own (e.g. under test, via
# `AppTest.from_file("pages/some_page.py")`), so each needs its own
# guarantee that `streamlit_app` is importable.
_here = Path(__file__).resolve()
for _candidate in (_here.parent, *_here.parents):
    if (_candidate / "streamlit_app").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import streamlit as st  # noqa: E402

from streamlit_app.session_state import clear_session, init_session_state  # noqa: E402
from streamlit_app.styling import configure_page  # noqa: E402

configure_page()
init_session_state()

with st.sidebar:
    st.markdown("## PolicyLens AI")
    st.caption("Multi-model financial document intelligence — portfolio project.")
    st.divider()
    if st.button("🧹 Clear session", help="Forget every document this session has uploaded or selected. Does not delete anything from the backend.", width="stretch"):
        clear_session()
        st.success("Session cleared.")
        st.rerun()

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/upload_document.py", title="Upload Document", icon="📤"),
    st.Page("pages/semantic_search.py", title="Semantic Search", icon="🔍"),
    st.Page("pages/ask_models.py", title="Ask Models", icon="🤖"),
    st.Page("pages/compare_models.py", title="Compare Models", icon="⚖️"),
    st.Page("pages/about_limitations.py", title="About & Limitations", icon="ℹ️"),
]

navigation = st.navigation(pages)
navigation.run()
