"""Ask Models page: independent grounded answers from each selected provider.

Accurate-behavior note: like any web form, the question text box below
keeps showing what was typed until it's changed, its key changes, or
"Clear session" is used -- see `session_state.py`'s module docstring.
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
    render_model_result,
    render_partial_failure_notice,
)
from streamlit_app.session_state import (  # noqa: E402
    get_active_document,
    get_document_history,
    init_session_state,
    widget_key,
)

_EXTERNAL_CALLS_DISABLED_MARKER = "ALLOW_EXTERNAL_LLM_CALLS"

init_session_state()

st.title("🤖 Ask Models")
st.caption("Ask a grounded question. Each selected model answers independently, citing only retrieved evidence.")

active_document = get_active_document()
history = get_document_history()

with st.form("ask_models_form"):
    question = st.text_area(
        "Question", placeholder="e.g. What is the overdraft fee?", height=100, key=widget_key("ask_question")
    )

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
        "Document scope", options=list(scope_options.keys()), index=default_index, key=widget_key("ask_scope")
    )

    provider_labels = st.multiselect(
        "Providers",
        options=["Claude (Anthropic)", "OpenAI"],
        default=["Claude (Anthropic)", "OpenAI"],
        key=widget_key("ask_providers"),
    )
    top_k = st.slider(
        "Evidence chunks to retrieve (top_k)", min_value=1, max_value=50, value=5, key=widget_key("ask_top_k")
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif not provider_labels:
        st.warning("Select at least one provider.")
    else:
        provider_map = {"Claude (Anthropic)": "anthropic", "OpenAI": "openai"}
        providers = [provider_map[label] for label in provider_labels]

        client = get_api_client()
        with st.spinner("Retrieving evidence and asking selected model(s)..."):
            result = client.ask(
                question=question, document_id=scope_options[scope_label], providers=providers, top_k=top_k
            )

        if result.ok:
            render_masked_query_notice(result.data.question, result.data.query_was_masked)
            st.caption(f"Evidence chunks supplied: {result.data.evidence_count}")
            st.divider()

            if all(
                _EXTERNAL_CALLS_DISABLED_MARKER in (r.error or "") for r in result.data.model_results
            ) and result.data.model_results:
                st.info(
                    "ℹ️ External LLM calls are disabled on this backend "
                    f"({_EXTERNAL_CALLS_DISABLED_MARKER}=false). An operator must opt in "
                    "explicitly before real Anthropic/OpenAI requests are made — see About & Limitations."
                )

            render_partial_failure_notice(result.data.model_results)

            for model_result in result.data.model_results:
                render_model_result(model_result)
        else:
            render_api_error(result.error)
