"""Streamlit rendering helpers built on top of `formatting.py`.

Rule followed throughout this file: `unsafe_allow_html` is never used
here. Model answers, citation excerpts, filenames, and questions/queries
are all rendered with `st.text` (plain, preformatted -- zero HTML or
Markdown interpretation) or interpolated into `st.markdown` only when
the value is already restricted to a safe character set by the backend
(e.g. a sanitized filename can't contain Markdown-special characters).
This keeps document/model-derived content from ever being rendered as
trusted HTML, and immune to Markdown-syntax surprises either way.
"""

from typing import List, Optional

import streamlit as st

from streamlit_app.api_client import (
    APIError,
    Citation,
    Comparison,
    DocumentUploadResult,
    ModelResult,
    ProviderMetrics,
    SearchResultItem,
)
from streamlit_app.components.formatting import (
    comparison_status_label,
    format_agreement_score,
    format_cost,
    format_latency,
    format_ratio_metric,
    format_relevance,
    format_signed_cost_difference,
    format_signed_latency_difference,
    format_tokens,
    format_word_count,
    pii_summary_text,
    status_icon,
    status_label,
)


def render_api_error(error: APIError) -> None:
    """A single, consistent error box for any failed API call.

    Only ever shows `error.message`, which `api_client.py` guarantees is
    already a short, safe, user-facing string -- never a raw exception,
    stack trace, or backend internal detail.
    """

    if error.kind == "connection":
        st.error(f"🔌 {error.message}")
    elif error.kind == "timeout":
        st.warning(f"⏱️ {error.message}")
    else:
        st.error(f"⚠️ {error.message}")


def render_backend_status(is_reachable: bool, detail: str) -> None:
    if is_reachable:
        st.success(f"✅ Backend connected — {detail}")
    else:
        st.warning(f"🔌 Backend unreachable — {detail}")


def render_pii_summary(document: DocumentUploadResult) -> None:
    st.caption(pii_summary_text(document.pii_detected, document.pii_entity_count, document.pii_categories))


def render_document_summary(document: DocumentUploadResult) -> None:
    st.markdown(f"**File:** {document.filename}")
    columns = st.columns(4)
    columns[0].metric("Pages", document.page_count)
    columns[1].metric("Characters", f"{document.character_count:,}")
    columns[2].metric("Chunks", document.chunk_count)
    columns[3].metric("Indexed chunks", document.indexed_chunk_count)
    render_pii_summary(document)

    with st.expander("Masked preview (first ~200 characters)"):
        st.text(document.preview or "(No extractable text on the first page.)")


def render_masked_query_notice(original_display_value: str, query_was_masked: bool) -> None:
    if query_was_masked:
        st.info("🔒 Personally identifiable information in your query was masked before it was used.")
    st.markdown("**Query used:**")
    st.text(original_display_value)


def render_search_result_card(item: SearchResultItem, index: int) -> None:
    with st.container(border=True):
        header_columns = st.columns([3, 1])
        header_columns[0].markdown(f"**{index}. {item.source_filename}** — page {item.page_number}")
        header_columns[1].metric("Relevance", format_relevance(item.relevance_score), label_visibility="collapsed")
        st.text(item.excerpt)


def render_citation(citation: Citation) -> None:
    with st.container(border=True):
        st.markdown(f"**[{citation.source_label}] {citation.source_filename}** — page {citation.page_number}")
        st.caption(f"Relevance: {format_relevance(citation.relevance_score)}")
        st.text(citation.excerpt)


def render_model_result(result: ModelResult, *, evidence_count: Optional[int] = None) -> None:
    with st.container(border=True):
        title_columns = st.columns([3, 2])
        title_columns[0].markdown(f"### {status_icon(result.status)} {result.provider.capitalize()} ({result.model})")
        title_columns[1].markdown(f"**Status:** {status_label(result.status)}")

        if result.status == "success":
            st.text(result.answer)
        elif result.status == "insufficient_evidence":
            st.warning(result.answer or "The available evidence does not contain enough information to answer this question.")
        else:
            st.error(result.error or "This model could not produce an answer.")

        metadata_columns = st.columns(3)
        metadata_columns[0].caption(f"Latency: {format_latency(result.latency_ms)}")
        metadata_columns[1].caption(f"Tokens: {format_tokens(result.input_tokens, result.output_tokens)}")
        metadata_columns[2].caption(f"Citations: {len(result.citations)}")

        if result.citations:
            with st.expander(f"Sources ({len(result.citations)})"):
                for citation in result.citations:
                    render_citation(citation)


def render_partial_failure_notice(model_results: List[ModelResult]) -> None:
    """If some providers succeeded and others didn't, say so plainly --
    the per-model cards already show each status individually, but a
    one-line summary above them makes a partial failure unmistakable at
    a glance."""

    statuses = {result.status for result in model_results}
    if len(statuses) <= 1:
        return
    succeeded = [r.provider.capitalize() for r in model_results if r.status == "success"]
    other = [r.provider.capitalize() for r in model_results if r.status != "success"]
    if succeeded and other:
        st.info(f"ℹ️ {', '.join(succeeded)} answered successfully; {', '.join(other)} did not. See each card below.")


def render_provider_metrics(metrics: ProviderMetrics) -> None:
    with st.container(border=True):
        st.markdown(f"**{metrics.provider.capitalize()}** ({metrics.model})")
        columns = st.columns(3)
        columns[0].metric("Latency", format_latency(metrics.latency_ms))
        columns[1].metric("Estimated cost", format_cost(metrics.estimated_cost_usd))
        columns[2].metric("Citations used", metrics.valid_citation_count)

        columns2 = st.columns(3)
        columns2[0].metric("Citation coverage", format_ratio_metric(metrics.citation_coverage))
        columns2[1].metric("Mean citation relevance", format_ratio_metric(metrics.mean_citation_relevance))
        columns2[2].metric("Grounded-term ratio", format_ratio_metric(metrics.grounded_term_ratio))
        st.caption(
            "⚠️ Coverage/relevance are retrieval-quality heuristics; grounded-term ratio is a "
            "lexical overlap heuristic. None of these measure factual accuracy."
        )

        st.caption(f"Answer length: {format_word_count(metrics.answer_length)} (context only, not a quality signal)")

        if metrics.evaluation_notes:
            with st.expander("Evaluation notes"):
                for note in metrics.evaluation_notes:
                    st.caption(note)


def render_comparison_summary(comparison: Comparison) -> None:
    st.markdown("### Comparison")
    st.caption(
        "This is a transparent, metric-by-metric comparison — there is no overall "
        "\"winner\" and no accuracy score. Heuristic metrics are labeled as such below."
    )
    st.markdown(f"**{comparison_status_label(comparison.comparison_status)}**")

    columns = st.columns(3)
    columns[0].metric("Answer agreement", format_agreement_score(comparison.answer_agreement_score))
    columns[1].metric("Latency difference", format_signed_latency_difference(comparison.latency_difference_ms))
    columns[2].metric("Cost difference", format_signed_cost_difference(comparison.estimated_cost_difference_usd))

    if comparison.answer_agreement_score is not None:
        st.caption(
            "Answer agreement is embedding cosine similarity (semantic closeness), "
            "NOT proof that either answer is factually correct."
        )

    if comparison.comparison_notes:
        with st.expander("Comparison notes", expanded=True):
            for note in comparison.comparison_notes:
                st.markdown(f"- {note}")
