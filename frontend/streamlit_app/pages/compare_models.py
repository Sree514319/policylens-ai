"""Compare Models page: transparent, metric-by-metric Claude-vs-OpenAI comparison.

Deliberately never renders a "winner" or a combined accuracy score --
see `components/render.render_comparison_summary` and the explicit
caption on this page.

Accurate-behavior note: like any web form, the question text box below
keeps showing what was typed until it's changed, its key changes, or
"Clear session" is used -- see `session_state.py`'s module docstring.
"""

import streamlit as st

from streamlit_app.api_client import get_api_client
from streamlit_app.components.formatting import format_latency
from streamlit_app.components.render import (
    render_api_error,
    render_comparison_summary,
    render_masked_query_notice,
    render_model_result,
    render_provider_metrics,
)
from streamlit_app.session_state import get_active_document, get_document_history, init_session_state, widget_key

init_session_state()

st.title("⚖️ Compare Models")
st.caption(
    "Runs exactly Claude (Anthropic) and OpenAI on the same question, once each, "
    "and reports a transparent, metric-by-metric comparison — never a single "
    "\"winner\" or accuracy score."
)

active_document = get_active_document()
history = get_document_history()

with st.form("compare_models_form"):
    question = st.text_area(
        "Question", placeholder="e.g. What is the overdraft fee?", height=100, key=widget_key("compare_question")
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
        "Document scope", options=list(scope_options.keys()), index=default_index, key=widget_key("compare_scope")
    )

    top_k = st.slider(
        "Evidence chunks to retrieve (top_k)", min_value=1, max_value=50, value=5, key=widget_key("compare_top_k")
    )
    submitted = st.form_submit_button("Compare", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        client = get_api_client()
        with st.spinner("Retrieving evidence and asking both models..."):
            result = client.compare(question=question, document_id=scope_options[scope_label], top_k=top_k)

        if result.ok:
            data = result.data
            render_masked_query_notice(data.question, data.query_was_masked)
            st.caption(f"Evidence chunks supplied: {data.evidence_count}")
            st.divider()

            st.subheader("Answers")
            answer_columns = st.columns(2)
            results_by_provider = {r.provider: r for r in data.model_results}
            for column, provider in zip(answer_columns, ["anthropic", "openai"]):
                with column:
                    if provider in results_by_provider:
                        render_model_result(results_by_provider[provider])

            st.divider()
            st.subheader("Evaluation metrics")
            metrics_columns = st.columns(2)
            metrics_by_provider = {m.provider: m for m in data.provider_metrics}
            for column, provider in zip(metrics_columns, ["anthropic", "openai"]):
                with column:
                    if provider in metrics_by_provider:
                        render_provider_metrics(metrics_by_provider[provider])

            anthropic_metrics = metrics_by_provider.get("anthropic")
            openai_metrics = metrics_by_provider.get("openai")
            if anthropic_metrics is not None and openai_metrics is not None:
                # The exact figures are stated in text right next to the
                # chart (not only visible in the bars themselves), so the
                # comparison doesn't depend on reading bar heights or color.
                st.caption(
                    "Latency comparison (lower is faster) — the only chart here, since it's a direct "
                    f"measurement, not a heuristic. Claude: {format_latency(anthropic_metrics.latency_ms)}; "
                    f"OpenAI: {format_latency(openai_metrics.latency_ms)}."
                )
                st.bar_chart(
                    {"Claude": anthropic_metrics.latency_ms, "OpenAI": openai_metrics.latency_ms},
                    x_label="Provider",
                    y_label="Latency (ms)",
                )

            st.divider()
            render_comparison_summary(data.comparison)
        else:
            render_api_error(result.error)
