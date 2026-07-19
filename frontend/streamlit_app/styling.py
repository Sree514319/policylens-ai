"""Shared, static visual constants and the app's one CSS injection.

Deliberately minimal: this app leans on Streamlit's built-in, already
-accessible components (`st.info`/`st.success`/`st.warning`/`st.error`,
`st.container(border=True)`, `st.metric`, `st.columns`) for a consistent
look rather than custom HTML/CSS, and avoids animation entirely.

The one call in this file that opts into raw HTML rendering is safe
because its argument is a hardcoded, static string with no interpolated
user/document/model content whatsoever -- never do that with any value
that came from an upload, a query, or a provider response (see
`components/render.py`, which never opts into raw HTML rendering).
"""

import streamlit as st

PAGE_TITLE = "PolicyLens AI"
PAGE_ICON = "📄"

STATUS_ICON = {"success": "✅", "insufficient_evidence": "⚠️", "error": "❌"}

# A restrained, professional financial/AI palette: deep slate-blue primary,
# a single accent, and standard semantic colors -- reused only for the one
# static CSS block below and never interpolated with dynamic content.
_PRIMARY_COLOR = "#1E3A5F"
_ACCENT_COLOR = "#2563EB"

_STATIC_CSS = f"""
<style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }}
    h1, h2, h3 {{
        color: {_PRIMARY_COLOR};
    }}
    a {{
        color: {_ACCENT_COLOR};
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 600;
    }}
</style>
"""


def configure_page() -> None:
    """Call exactly once, first thing, in `app.py` -- `st.set_page_config`
    is only valid as the very first Streamlit command in a run. Sets a
    responsive wide layout and the browser tab title/icon; injects the
    static CSS above. Individual pages set their own on-page `st.title`
    but must NOT call this again."""

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_STATIC_CSS, unsafe_allow_html=True)
