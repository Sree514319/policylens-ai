"""PolicyLens AI -- Streamlit frontend entrypoint.

Run with:
    streamlit run frontend/streamlit_app/app.py

This script only wires up page configuration, session-state
initialization, and navigation -- all actual page content lives in
`pages/*.py`, and all backend communication goes through
`api_client.py` (HTTP only; this app never imports backend service code).
"""

import streamlit as st

from streamlit_app.session_state import clear_session, init_session_state
from streamlit_app.styling import configure_page

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
