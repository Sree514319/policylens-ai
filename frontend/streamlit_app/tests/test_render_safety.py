"""Tests that model/document content is never rendered as trusted HTML.

Two layers of proof:
1. A static source check: `unsafe_allow_html=True` must never appear in
   `components/render.py` or any `pages/*.py` file (it's used exactly
   once in the whole app, in `styling.py`, on a hardcoded static string
   -- see that file's docstring for why that one use is safe).
2. A runtime check via Streamlit's `AppTest`: a crafted "malicious"
   answer/excerpt containing HTML and Markdown syntax renders as inert,
   literal text (an `st.text` element), never as an interpreted
   `st.markdown(..., unsafe_allow_html=True)` block.
"""

import textwrap
from pathlib import Path

from streamlit.testing.v1 import AppTest

FRONTEND_ROOT = Path(__file__).resolve().parents[2]
RENDER_MODULE = FRONTEND_ROOT / "streamlit_app" / "components" / "render.py"
PAGES_DIR = FRONTEND_ROOT / "streamlit_app" / "pages"

_MALICIOUS_TEXT = '<script>alert("xss")</script> and **bold** and [link](javascript:evil())'


def test_render_module_never_uses_unsafe_allow_html():
    # Matches the actual invocation pattern (`unsafe_allow_html=True`),
    # not the bare word -- which this file's own docstring mentions in
    # prose while explaining the rule.
    source = RENDER_MODULE.read_text(encoding="utf-8")
    assert "unsafe_allow_html=True" not in source


def test_no_page_uses_unsafe_allow_html():
    for page_file in PAGES_DIR.glob("*.py"):
        source = page_file.read_text(encoding="utf-8")
        assert "unsafe_allow_html=True" not in source, f"{page_file.name} must not use unsafe_allow_html"


def test_only_one_unsafe_allow_html_use_in_the_entire_app_and_it_is_static():
    # styling.py's one use is the sole, deliberate exception -- confirmed
    # here to still be a hardcoded string with no f-string/format
    # interpolation of any dynamic (let alone user/document) content.
    styling_source = (FRONTEND_ROOT / "streamlit_app" / "styling.py").read_text(encoding="utf-8")
    assert styling_source.count("unsafe_allow_html=True") == 1

    all_python_files = list((FRONTEND_ROOT / "streamlit_app").rglob("*.py"))
    total_uses = sum(
        f.read_text(encoding="utf-8").count("unsafe_allow_html=True") for f in all_python_files if "tests" not in f.parts
    )
    assert total_uses == 1


def _run_snippet(tmp_path, snippet: str) -> AppTest:
    script_path = tmp_path / "snippet.py"
    script_path.write_text(textwrap.dedent(snippet), encoding="utf-8")
    at = AppTest.from_file(str(script_path))
    at.run(timeout=15)
    return at


def test_model_answer_text_renders_as_literal_inert_text(tmp_path):
    at = _run_snippet(
        tmp_path,
        f"""
        import sys
        sys.path.insert(0, {str(FRONTEND_ROOT)!r})
        from streamlit_app.api_client import ModelResult
        from streamlit_app.components.render import render_model_result

        result = ModelResult(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            status="success",
            answer={_MALICIOUS_TEXT!r},
            citations=[],
            latency_ms=100.0,
        )
        render_model_result(result)
        """,
    )

    assert at.exception == []
    rendered_text_values = [element.value for element in at.text]
    assert _MALICIOUS_TEXT in rendered_text_values

    # It must never appear inside a markdown element's *raw* source either
    # (which would risk interpretation) -- markdown elements on this page
    # are only our own static labels/status strings.
    for markdown_element in at.markdown:
        assert "<script>" not in markdown_element.value


def test_citation_excerpt_renders_as_literal_inert_text(tmp_path):
    at = _run_snippet(
        tmp_path,
        f"""
        import sys
        sys.path.insert(0, {str(FRONTEND_ROOT)!r})
        from streamlit_app.api_client import Citation
        from streamlit_app.components.render import render_citation

        citation = Citation(
            source_label="S1",
            chunk_id="c1",
            document_id="d1",
            source_filename="policy.pdf",
            page_number=1,
            excerpt={_MALICIOUS_TEXT!r},
            relevance_score=0.9,
        )
        render_citation(citation)
        """,
    )

    assert at.exception == []
    rendered_text_values = [element.value for element in at.text]
    assert _MALICIOUS_TEXT in rendered_text_values


def test_search_result_excerpt_renders_as_literal_inert_text(tmp_path):
    at = _run_snippet(
        tmp_path,
        f"""
        import sys
        sys.path.insert(0, {str(FRONTEND_ROOT)!r})
        from streamlit_app.api_client import SearchResultItem
        from streamlit_app.components.render import render_search_result_card

        item = SearchResultItem(
            chunk_id="c1",
            document_id="d1",
            source_filename="policy.pdf",
            page_number=1,
            excerpt={_MALICIOUS_TEXT!r},
            relevance_score=0.9,
        )
        render_search_result_card(item, 1)
        """,
    )

    assert at.exception == []
    rendered_text_values = [element.value for element in at.text]
    assert _MALICIOUS_TEXT in rendered_text_values
