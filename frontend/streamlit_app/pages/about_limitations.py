"""About & Limitations page: architecture summary and every disclosed limitation."""

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

from streamlit_app.session_state import init_session_state  # noqa: E402

init_session_state()

st.title("ℹ️ About & Limitations")

st.subheader("Architecture summary")
st.markdown(
    """
FastAPI backend + local ChromaDB vector store + a provider-neutral LLM
abstraction (Anthropic Claude / OpenAI GPT) + local, regex-based PII
masking + a transparent evaluation/comparison layer. This Streamlit app
is a pure HTTP client of that backend — it never imports backend service
code directly, holds no API keys, and makes no direct calls to Anthropic
or OpenAI itself.

**Pipeline:** PDF upload → in-memory text extraction → PII masking →
deterministic chunking → local embedding + indexing → semantic
search / grounded RAG answering → optional Claude-vs-OpenAI comparison.
"""
)

st.divider()
st.subheader("🔒 PII detection: coverage and limitations")
st.markdown(
    """
A local, regex- and checksum-based detector masks common US financial
identifiers **before** any text is chunked, indexed, displayed, or sent
to a third-party model:

- US Social Security Numbers (with SSA-invalid-group rejection)
- Credit/debit card numbers (Luhn-validated)
- Email addresses
- US phone numbers
- IPv4 addresses
- Dates of birth (only when clearly labeled, and calendar-valid)
- Bank account numbers (only with account-number context)
- US routing numbers (context- and checksum-validated)

**This is best-effort, not a compliance product.** It does **not**
reliably detect personal names, free-form postal addresses, or most
non-US identifiers. Detection does not cross a page boundary. Only
synthetic or public sample documents should ever be uploaded.
"""
)

st.divider()
st.subheader("🌐 External-provider privacy")
st.warning(
    "Real Anthropic/OpenAI calls are **off by default** on the backend "
    "(`ALLOW_EXTERNAL_LLM_CALLS=false`) and must be explicitly enabled by "
    "whoever runs it. When enabled, only the **masked** question and **masked** "
    "retrieved evidence excerpts are sent — never the full document, and never "
    "an unmasked question. This frontend never talks to Anthropic/OpenAI directly "
    "and never sees or stores an API key."
)

st.divider()
st.subheader("📊 Evaluation metrics: limitations")
st.markdown(
    """
- **`grounded_term_ratio` is a lexical overlap heuristic, not
  fact-checking.** It measures whether the answer's vocabulary appears in
  its cited excerpts — not whether its claims are actually true.
- **`answer_agreement_score` is semantic similarity, not correctness.**
  Two similar-sounding answers can both be wrong; two differently-worded
  answers can both be right.
- **There is no accuracy, precision, recall, or "% correct" score
  anywhere in this application.** No labeled, human-verified answer
  dataset exists in this project.
- **`estimated_cost_usd` depends entirely on operator-configured
  pricing.** No vendor price list is hardcoded; if configured, it can go
  stale as real pricing changes.
- The comparison page **never** declares an overall winner — every
  metric is reported (or shown as unavailable) individually.
"""
)

st.divider()
st.subheader("⚠️ Compliance disclaimer")
st.error(
    "PolicyLens AI is a **technical portfolio demonstration**. It is **not** "
    "HIPAA, PCI-DSS, GDPR, or GLBA compliant, makes **no regulatory compliance "
    "claim of any kind**, and must not be used with real customer or other "
    "sensitive data. Model outputs are informational only and are not financial, "
    "legal, or compliance advice."
)
