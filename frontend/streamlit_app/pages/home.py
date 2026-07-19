"""Home page: project overview, backend health, and privacy notice."""

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
from streamlit_app.components.render import render_backend_status  # noqa: E402
from streamlit_app.session_state import init_session_state  # noqa: E402

init_session_state()

st.title("📄 PolicyLens AI")
st.caption("A multi-model financial document intelligence assistant — portfolio project.")

st.markdown(
    """
PolicyLens AI lets you upload a banking policy PDF, ask questions about
it in natural language, and get **cited** answers — generated
independently by both **Anthropic Claude** and **OpenAI GPT** — alongside
a transparent, metric-by-metric comparison of grounding, latency, and
cost. It never produces a single "accuracy" score or declares a winner.
"""
)

st.divider()
st.subheader("Backend status")

client = get_api_client()
with st.spinner("Checking backend connection..."):
    health_result = client.health()

if health_result.ok:
    render_backend_status(True, f"status: {health_result.data.status}")
else:
    render_backend_status(False, health_result.error.message)
    st.caption(
        "Every page still loads without a backend connection, but uploading, "
        "searching, and asking/comparing models require it. Start the backend with:"
    )
    st.code(
        ".venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --reload",
        language="bash",
    )

st.divider()
st.subheader("⚠️ Privacy notice")
st.warning(
    "This is a **portfolio project, not a compliance product**. Local, regex-based "
    "PII masking runs on every document and query, but it is best-effort — it does "
    "**not** reliably detect personal names or free-form addresses, and it makes "
    "**no HIPAA, PCI-DSS, GDPR, or GLBA compliance claim**. Only upload **synthetic "
    "or public sample documents** — never real customer or other sensitive data. "
    "See the About & Limitations page for details."
)

st.divider()
st.subheader("How it works")
st.markdown(
    """
1. **Upload Document** — upload a PDF; it's validated, text-extracted in
   memory, PII-masked, chunked, and indexed for search. Only the masked
   preview and summary counts are ever shown or stored.
2. **Semantic Search** — run a keyword-free search over your indexed
   document(s) and see ranked, citation-ready excerpts.
3. **Ask Models** — ask a grounded question; Claude and OpenAI each
   answer independently, citing only the retrieved evidence.
4. **Compare Models** — ask the same question and see both models'
   answers, citations, and evaluation metrics side by side.
5. **About & Limitations** — architecture, PII coverage, and every
   limitation of the evaluation metrics, spelled out plainly.
"""
)
