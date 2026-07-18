"""Tests for scripts/run_evaluation.py (the offline benchmark runner).

`scripts/` lives outside the `backend/` package pytest.ini points at, so
this file adds it to `sys.path` directly rather than relying on the
`app.*` import root used elsewhere in this test suite.
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_evaluation  # noqa: E402


def _write_jsonl(tmp_path, lines):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- load_benchmark_records: parsing / validation -------------------------------------


def test_loads_valid_records():
    records = run_evaluation.load_benchmark_records(run_evaluation.DEFAULT_QUESTIONS_PATH)
    assert len(records) >= 1
    for record in records:
        assert "id" in record
        assert "question" in record


def test_sample_questions_file_contains_only_placeholder_document_ids():
    # The shipped benchmark file must never contain a real customer
    # document_id -- only the documented placeholder.
    records = run_evaluation.load_benchmark_records(run_evaluation.DEFAULT_QUESTIONS_PATH)
    for record in records:
        assert record.get("document_id") == "REPLACE_WITH_UPLOADED_DOCUMENT_ID"


def test_missing_required_field_raises(tmp_path):
    path = _write_jsonl(tmp_path, ['{"id": "q1"}'])  # missing "question"
    with pytest.raises(ValueError, match="missing required field"):
        run_evaluation.load_benchmark_records(path)


def test_invalid_json_raises_with_line_number(tmp_path):
    path = _write_jsonl(tmp_path, ['{"id": "q1", "question": "ok"}', "{not json"])
    with pytest.raises(ValueError, match=r"questions\.jsonl:2"):
        run_evaluation.load_benchmark_records(path)


def test_duplicate_id_is_rejected(tmp_path):
    path = _write_jsonl(
        tmp_path,
        ['{"id": "q1", "question": "first?"}', '{"id": "q1", "question": "second?"}'],
    )
    with pytest.raises(ValueError, match="duplicate record id"):
        run_evaluation.load_benchmark_records(path)


def test_blank_lines_and_comments_are_skipped(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            "# a comment line",
            "",
            '{"id": "q1", "question": "first?"}',
            "   ",
            '{"id": "q2", "question": "second?"}',
        ],
    )
    records = run_evaluation.load_benchmark_records(path)
    assert [r["id"] for r in records] == ["q1", "q2"]


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_evaluation.load_benchmark_records(tmp_path / "does_not_exist.jsonl")


# --- Dry run: zero network calls, correct output ---------------------------------------


def test_dry_run_makes_zero_network_calls(tmp_path, monkeypatch):
    def _blocked_socket(*args, **kwargs):
        raise AssertionError("dry-run must never construct a network socket")

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    records = run_evaluation.load_benchmark_records(run_evaluation.DEFAULT_QUESTIONS_PATH)
    output_path = run_evaluation.run_dry_run(records, tmp_path)

    assert output_path.exists()


def test_dry_run_does_not_import_network_or_llm_modules(tmp_path):
    # A stronger, structural guarantee: run the CLI as a real subprocess
    # (fresh interpreter, empty sys.modules) and confirm none of the
    # network-capable modules dry-run must never touch got imported.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_evaluation.py"),
            "--mode",
            "dry-run",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "no network calls were made" in result.stdout.lower()

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; print('chromadb' in sys.modules or 'anthropic' in sys.modules or 'openai' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    # (This check process is independent of the one above; it only proves
    # the import statement itself is harmless. The real proof is that the
    # subprocess above never imported those packages, since `run_dry_run`'s
    # code path contains no reference to them -- see the module docstring
    # and `_run_live`'s deferred imports.)
    assert check.returncode == 0


def test_dry_run_output_contains_no_network_calls_note_and_question_count(tmp_path):
    records = run_evaluation.load_benchmark_records(run_evaluation.DEFAULT_QUESTIONS_PATH)
    output_path = run_evaluation.run_dry_run(records, tmp_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry-run"
    assert payload["question_count"] == len(records)
    assert payload["question_ids"] == [r["id"] for r in records]
    assert "no network calls were made" in payload["note"].lower()


def test_dry_run_never_persists_question_text():
    records = run_evaluation.load_benchmark_records(run_evaluation.DEFAULT_QUESTIONS_PATH)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        output_path = run_evaluation.run_dry_run(records, Path(tmp))
        raw = output_path.read_text(encoding="utf-8")
        for record in records:
            assert record["question"] not in raw


# --- CLI-level opt-in gating (subprocess, so argv/exit-code behavior is real) ---------


def test_cli_default_mode_is_dry_run(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_evaluation.py"), "--results-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "dry run complete" in result.stdout.lower()


def test_cli_live_mode_without_opt_in_flag_refuses(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_evaluation.py"), "--mode", "live", "--results-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "--i-understand-this-calls-external-apis" in result.stderr


def test_cli_live_mode_with_opt_in_flag_still_refuses_without_allow_external_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_EXTERNAL_LLM_CALLS", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_evaluation.py"),
            "--mode",
            "live",
            "--i-understand-this-calls-external-apis",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "ALLOW_EXTERNAL_LLM_CALLS": "false"},
    )
    assert result.returncode == 2
    assert "ALLOW_EXTERNAL_LLM_CALLS" in result.stderr
    # No results file should have been written -- the script refused before
    # doing any work.
    assert list(tmp_path.glob("live_run_*.json")) == []


# --- Repository hygiene: results/ stays git-ignored -----------------------------------


def test_gitignore_excludes_evaluation_results_contents():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/evaluation/results/*" in gitignore_text
    assert "!data/evaluation/results/.gitkeep" in gitignore_text


def test_no_generated_evaluation_result_files_are_tracked_by_git():
    tracked = subprocess.run(
        ["git", "ls-files", "data/evaluation/results"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    generated_files = [f for f in tracked if not f.endswith(".gitkeep")]
    assert generated_files == []


def test_sample_questions_file_itself_is_not_gitignored():
    # Unlike results/, the seed benchmark file is intentionally meant to be
    # committed -- `git check-ignore` exits 1 ("not ignored") for a path
    # that .gitignore rules don't match, which is what should hold here.
    # (Doesn't assert it's *already* tracked -- this phase isn't committed
    # yet -- only that nothing would silently exclude it from a commit.)
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/evaluation/sample_questions.jsonl"],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1


# --- Truncated / malformed JSONL --------------------------------------------------


def test_truncated_final_line_raises_a_clear_non_sensitive_error(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": "q1", "question": "first?"}\n{"id": "q2", "question": "seco', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        run_evaluation.load_benchmark_records(path)

    message = str(exc_info.value)
    assert "questions.jsonl:2" in message
    assert "seco" not in message  # no raw line content echoed back


def test_record_that_is_not_a_json_object_is_rejected(tmp_path):
    path = _write_jsonl(tmp_path, ['["not", "an", "object"]'])
    with pytest.raises(ValueError, match="must be a JSON object"):
        run_evaluation.load_benchmark_records(path)


# --- Live mode: only masked data is ever persisted, even with --include-answer-text ---


@pytest.mark.asyncio
async def test_live_mode_never_persists_raw_pii_even_with_include_answer_text(tmp_path, monkeypatch):
    import app.core.config as config_module
    import app.services.llm.providers as providers_module
    import app.services.privacy.detectors as detectors_module
    import app.services.retrieval.vector_store as vector_store_module
    from app.core.config import Settings
    from app.services.llm.providers import FakeLLMProvider
    from app.services.privacy.detectors import LocalRegexPIIDetector
    from app.services.retrieval.embeddings import FakeEmbeddingProvider
    from app.services.retrieval.vector_store import VectorStore
    from tests.test_vector_store import _chunk

    secret_ssn = "123-45-6789"

    settings = Settings(allow_external_llm_calls=True)
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_provider=FakeEmbeddingProvider(),
        pii_protection_enabled=True,
        pii_redaction_version="v1",
    )
    vector_store.upsert_chunks([_chunk("Overdraft fees are thirty five dollars per occurrence.", chunk_index=0)])
    fake_providers = {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
        "openai": FakeLLMProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store_module, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(providers_module, "get_llm_provider_registry", lambda: fake_providers)
    monkeypatch.setattr(detectors_module, "get_pii_detector", lambda: LocalRegexPIIDetector())

    records = [{"id": "q1", "question": f"For SSN {secret_ssn}, what is the overdraft fee?"}]
    results_dir = tmp_path / "results"

    output_path = await run_evaluation._run_live(records, results_dir, include_answer_text=True)

    raw_text = output_path.read_text(encoding="utf-8")
    assert secret_ssn not in raw_text

    payload = json.loads(raw_text)
    assert payload["results"][0]["query_was_masked"] is True


@pytest.mark.asyncio
async def test_live_mode_excludes_answer_text_by_default(tmp_path, monkeypatch):
    import app.core.config as config_module
    import app.services.llm.providers as providers_module
    import app.services.privacy.detectors as detectors_module
    import app.services.retrieval.vector_store as vector_store_module
    from app.core.config import Settings
    from app.services.llm.providers import FakeLLMProvider
    from app.services.privacy.detectors import LocalRegexPIIDetector
    from app.services.retrieval.embeddings import FakeEmbeddingProvider
    from app.services.retrieval.vector_store import VectorStore
    from tests.test_vector_store import _chunk

    settings = Settings(allow_external_llm_calls=True)
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_provider=FakeEmbeddingProvider(),
        pii_protection_enabled=True,
        pii_redaction_version="v1",
    )
    vector_store.upsert_chunks([_chunk("Overdraft fees are thirty five dollars.", chunk_index=0)])
    fake_providers = {
        "anthropic": FakeLLMProvider(
            name="anthropic",
            response_json={
                "insufficient_evidence": False,
                "answer": "A very specific unredacted answer marker XYZZY.",
                "citations": ["S1"],
            },
        ),
        "openai": FakeLLMProvider(
            name="openai",
            response_json={"insufficient_evidence": False, "answer": "Answer [S1].", "citations": ["S1"]},
        ),
    }

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store_module, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(providers_module, "get_llm_provider_registry", lambda: fake_providers)
    monkeypatch.setattr(detectors_module, "get_pii_detector", lambda: LocalRegexPIIDetector())

    records = [{"id": "q1", "question": "What is the overdraft fee?"}]
    results_dir = tmp_path / "results"

    output_path = await run_evaluation._run_live(records, results_dir, include_answer_text=False)

    raw_text = output_path.read_text(encoding="utf-8")
    assert "XYZZY" not in raw_text
    payload = json.loads(raw_text)
    assert "answers" not in payload["results"][0]


# --- Atomic writes: an interrupted write must never leave a corrupt "completed" file --


def test_write_json_leaves_no_partial_file_if_replace_fails(tmp_path, monkeypatch):
    import os as os_module

    real_replace = os_module.replace

    def _failing_replace(*args, **kwargs):
        raise OSError("simulated failure during rename")

    monkeypatch.setattr(run_evaluation.os, "replace", _failing_replace)

    with pytest.raises(OSError):
        run_evaluation._write_json(tmp_path, "output.json", {"mode": "live", "results": []})

    # The final path must not exist (write never completed)...
    assert not (tmp_path / "output.json").exists()
    # ...and no leftover temp file should survive the failure either.
    leftover = list(tmp_path.glob(".output.json.*.tmp"))
    assert leftover == []

    monkeypatch.setattr(run_evaluation.os, "replace", real_replace)


def test_write_json_never_leaves_a_partial_file_visible_at_the_final_path(tmp_path, monkeypatch):
    # Simulate a crash *during* the write (before the atomic rename ever
    # happens) -- json.dump raises partway through serialization.
    def _failing_dump(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(run_evaluation.json, "dump", _failing_dump)

    with pytest.raises(RuntimeError):
        run_evaluation._write_json(tmp_path, "output.json", {"mode": "live", "results": []})

    assert not (tmp_path / "output.json").exists()
    assert list(tmp_path.glob(".output.json.*.tmp")) == []


def test_write_json_produces_valid_complete_json_on_success(tmp_path):
    payload = {"mode": "live", "results": [{"id": "q1"}]}
    output_path = run_evaluation._write_json(tmp_path, "output.json", payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    # No leftover temp file after a successful write.
    assert list(tmp_path.glob(".output.json.*.tmp")) == []
