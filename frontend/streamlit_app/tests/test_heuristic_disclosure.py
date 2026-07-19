"""Tests that heuristic-metric limitations are visible immediately next
to the metrics themselves, not only in a separate, possibly-unread
section of the page (see the Compare Models page's "Evaluation metrics"
column, `components/render.render_provider_metrics`).
"""

import textwrap
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND_ROOT = Path(__file__).resolve().parents[2]


def _run_snippet(tmp_path, snippet: str) -> AppTest:
    script_path = tmp_path / "snippet.py"
    script_path.write_text(textwrap.dedent(snippet), encoding="utf-8")
    at = AppTest.from_file(str(script_path))
    at.run(timeout=15)
    return at


def test_provider_metrics_card_shows_a_heuristic_disclaimer_inline(tmp_path):
    at = _run_snippet(
        tmp_path,
        f"""
        import sys
        sys.path.insert(0, {str(FRONTEND_ROOT)!r})
        from streamlit_app.api_client import ProviderMetrics
        from streamlit_app.components.render import render_provider_metrics

        metrics = ProviderMetrics(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            status="success",
            latency_ms=842.3,
            input_tokens=512,
            output_tokens=41,
            estimated_cost_usd=0.0022,
            valid_citation_count=1,
            citation_coverage=0.5,
            mean_citation_relevance=0.87,
            grounded_term_ratio=0.8,
            answer_length=8,
            evaluation_notes=[],
        )
        render_provider_metrics(metrics)
        """,
    )

    assert at.exception == []
    captions = [c.value.lower() for c in at.caption]
    # The disclaimer must appear in a caption that also mentions the
    # specific heuristic metric names -- not buried only inside a
    # collapsed "Evaluation notes" expander a user might never open.
    assert any("heuristic" in c and ("coverage" in c or "grounded" in c) for c in captions)
    assert any("factual accuracy" in c or "not measure" in c for c in captions)


def test_heuristic_disclaimer_appears_without_needing_to_expand_notes(tmp_path):
    # Same as above, but with evaluation_notes present too -- the inline
    # caption must not depend on (or be replaced by) the notes expander.
    at = _run_snippet(
        tmp_path,
        f"""
        import sys
        sys.path.insert(0, {str(FRONTEND_ROOT)!r})
        from streamlit_app.api_client import ProviderMetrics
        from streamlit_app.components.render import render_provider_metrics

        metrics = ProviderMetrics(
            provider="openai",
            model="gpt-4o-mini",
            status="success",
            latency_ms=962.8,
            input_tokens=480,
            output_tokens=38,
            estimated_cost_usd=None,
            valid_citation_count=1,
            citation_coverage=0.5,
            mean_citation_relevance=0.87,
            grounded_term_ratio=None,
            answer_length=7,
            evaluation_notes=["Per-token pricing is not configured for this provider."],
        )
        render_provider_metrics(metrics)
        """,
    )

    assert at.exception == []
    # The expander ("Evaluation notes") is a separate, collapsed element
    # -- the inline caption disclaimer must exist independently of it.
    captions = [c.value.lower() for c in at.caption]
    assert any("heuristic" in c for c in captions)
    expanders = [e for e in at.expander]
    assert any("evaluation notes" in (e.label or "").lower() for e in expanders)
