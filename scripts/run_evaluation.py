#!/usr/bin/env python
"""Offline benchmark runner for PolicyLens AI's grounded RAG answers.

Reads a JSONL file of benchmark question records (see
data/evaluation/sample_questions.jsonl) and, in "live" mode, runs each
question through the exact same RAG orchestration + evaluation/comparison
pipeline as POST /api/v1/compare -- writing results to the git-ignored
data/evaluation/results/ directory.

SAFETY
------
- The default mode is "dry-run": it loads and validates the benchmark
  file and reports what WOULD run. It makes zero network calls and does
  not construct a vector store, an embedding provider, or an LLM
  provider -- those modules are not even imported in dry-run mode.
- "live" mode requires BOTH an explicit --i-understand-this-calls-external-apis
  flag on this script AND ALLOW_EXTERNAL_LLM_CALLS=true in the server's
  own configuration (the same safety switch /api/v1/answers and
  /api/v1/compare already respect). Missing either one refuses to run.
- Live mode still applies this project's local PII masking to every
  question before retrieval/embedding/any provider call, exactly like the
  API does (see app.services.privacy.masking).
- By default, live-mode results persist only metrics (status, latency,
  token counts, estimated cost, citation coverage/relevance, grounded-
  term ratio, comparison notes) -- never the generated answer text or
  citation excerpts. Pass --include-answer-text to also persist each
  provider's (already-masked) answer text for a given run.

NOT AN ACCURACY BENCHMARK
--------------------------
`expected_source_keywords`/`expected_page` in each benchmark record are
optional, informational fields for a human reviewer to eyeball against
the actual citations returned. This script does not compute, store, or
claim any accuracy/pass-fail score from them -- there is no labeled,
verified-correct-answer dataset here, only synthetic/public example
questions with a placeholder document_id you fill in after uploading
your own synthetic sample document.

Usage
-----
    python scripts/run_evaluation.py                        # dry run (default)
    python scripts/run_evaluation.py --mode dry-run
    python scripts/run_evaluation.py --mode live \\
        --i-understand-this-calls-external-apis

Exit codes
----------
    0  Success (dry run completed, or live run completed -- individual
       per-question retrieval failures inside a live run do not change
       this; they're recorded as an "error" entry for that question and
       the run continues).
    1  The benchmark file is missing or fails structural validation
       (malformed JSON, a missing required field, a duplicate id).
    2  Live mode was requested but refused by a safety gate (missing
       --i-understand-this-calls-external-apis, or
       ALLOW_EXTERNAL_LLM_CALLS is not true in the server configuration).
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_QUESTIONS_PATH = _REPO_ROOT / "data" / "evaluation" / "sample_questions.jsonl"
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "evaluation" / "results"

REQUIRED_RECORD_FIELDS = ("id", "question")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_benchmark_records(path: Path) -> List[Dict[str, Any]]:
    """Parse and structurally validate a benchmark JSONL file.

    Each non-blank, non-comment line must be a JSON object with at least
    "id" and "question". "document_id" (a placeholder until you fill in a
    real uploaded document_id), "expected_source_keywords", and
    "expected_page" are optional and purely informational.
    """

    if not path.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    records: List[Dict[str, Any]] = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc}).") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object.")

            missing = [field for field in REQUIRED_RECORD_FIELDS if not record.get(field)]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing required field(s): {missing}.")

            if record["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate record id '{record['id']}'.")
            seen_ids.add(record["id"])

            records.append(record)

    return records


def _write_json(results_dir: Path, filename: str, payload: Dict[str, Any]) -> Path:
    """Write `payload` to `results_dir/filename` atomically.

    Writes to a temp file in the same directory first, then `os.replace`s
    it into place -- `os.replace` is atomic on both POSIX and Windows for
    a same-volume rename. If the process is interrupted (killed, crashes,
    power loss) partway through, the temp file is left behind (or cleaned
    up, best-effort) but the final `filename` either doesn't exist yet or
    still holds its previous complete contents -- never a truncated/
    half-written JSON file that could be mistaken for a completed report.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / filename

    fd, tmp_name = tempfile.mkstemp(dir=str(results_dir), prefix=f".{filename}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return output_path


def run_dry_run(records: List[Dict[str, Any]], results_dir: Path) -> Path:
    """Report what a live run WOULD do. Makes no network calls whatsoever."""

    plan = {
        "mode": "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(records),
        "question_ids": [record["id"] for record in records],
        "providers_that_would_be_queried": ["anthropic", "openai"],
        "note": (
            "Dry run only: no network calls were made, and no vector store, "
            "embedding provider, or LLM provider was constructed. Re-run with "
            "--mode live --i-understand-this-calls-external-apis (and "
            "ALLOW_EXTERNAL_LLM_CALLS=true configured) to actually query "
            "Anthropic/OpenAI against your indexed documents."
        ),
    }
    return _write_json(results_dir, f"dry_run_{_timestamp()}.json", plan)


async def _run_live(
    records: List[Dict[str, Any]], results_dir: Path, include_answer_text: bool
) -> Path:
    # Deferred so dry-run never imports chromadb/provider SDKs at all.
    from dataclasses import asdict

    from app.core.config import get_settings
    from app.services.evaluation.metrics import compare_providers, evaluate_providers
    from app.services.llm.providers import get_llm_provider_registry
    from app.services.llm.rag import answer_question
    from app.services.privacy.detectors import get_pii_detector
    from app.services.privacy.masking import mask_query
    from app.services.retrieval.vector_store import get_vector_store

    settings = get_settings()
    if not settings.allow_external_llm_calls:
        raise RuntimeError(
            "ALLOW_EXTERNAL_LLM_CALLS is not enabled in the server configuration. "
            "Set ALLOW_EXTERNAL_LLM_CALLS=true in .env to run this script in live mode."
        )

    vector_store = get_vector_store()
    llm_providers = get_llm_provider_registry()
    pii_detector = get_pii_detector()

    results: List[Dict[str, Any]] = []

    for record in records:
        entry: Dict[str, Any] = {
            "id": record["id"],
            # Informational only -- never scored into an accuracy metric.
            "expected_source_keywords": record.get("expected_source_keywords"),
            "expected_page": record.get("expected_page"),
        }

        if settings.pii_protection_enabled:
            question, query_was_masked = mask_query(record["question"], pii_detector)
        else:
            question, query_was_masked = record["question"], False
        entry["query_was_masked"] = query_was_masked

        document_id: Optional[str] = record.get("document_id") or None

        try:
            evidence_count, model_answers = await answer_question(
                question=question,
                document_id=document_id,
                top_k=settings.retrieval_top_k,
                vector_store=vector_store,
                providers=llm_providers,
                provider_names=["anthropic", "openai"],
                min_relevance_score=settings.min_relevance_score,
                max_context_characters=settings.max_rag_context_characters,
                allow_external_calls=settings.allow_external_llm_calls,
            )
        except Exception as exc:
            # Never include the raw exception message -- it could echo
            # back document_id/config details. A generic, typed summary is
            # enough for a benchmark run to keep going with the next question.
            entry["error"] = f"{type(exc).__name__}: retrieval failed for this question."
            results.append(entry)
            continue

        metrics_by_provider = evaluate_providers(model_answers, evidence_count, settings)
        answers_by_provider = {answer.provider: answer for answer in model_answers}

        comparison = compare_providers(
            anthropic_answer=answers_by_provider["anthropic"],
            openai_answer=answers_by_provider["openai"],
            anthropic_metrics=metrics_by_provider["anthropic"],
            openai_metrics=metrics_by_provider["openai"],
            embedding_provider=vector_store.embedding_provider,
            tie_threshold=settings.model_comparison_tie_threshold,
        )

        entry["evidence_count"] = evidence_count
        entry["provider_metrics"] = {
            provider: asdict(metrics) for provider, metrics in metrics_by_provider.items()
        }
        entry["comparison"] = asdict(comparison)

        if include_answer_text:
            # Still only ever the already-masked text this run produced --
            # masking happens upstream of every provider call, so no raw
            # PII can appear here even when this opt-in is used.
            entry["answers"] = {
                provider: answer.answer for provider, answer in answers_by_provider.items()
            }

        results.append(entry)

    payload = {
        "mode": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_answer_text": include_answer_text,
        "results": results,
    }
    return _write_json(results_dir, f"live_run_{_timestamp()}.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Path to a benchmark JSONL file."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory to write results to (git-ignored)."
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "live"],
        default="dry-run",
        help="'dry-run' (default): validate the benchmark file, no network calls. "
        "'live': actually query Anthropic/OpenAI (requires the flag below).",
    )
    parser.add_argument(
        "--i-understand-this-calls-external-apis",
        action="store_true",
        dest="live_opt_in",
        help="Required, in addition to --mode live and ALLOW_EXTERNAL_LLM_CALLS=true, to run live.",
    )
    parser.add_argument(
        "--include-answer-text",
        action="store_true",
        help="Live mode only: also persist each (already-masked) generated answer text. "
        "Off by default -- live answers are not persisted unless explicitly requested.",
    )
    args = parser.parse_args()

    try:
        records = load_benchmark_records(args.questions)
    except FileNotFoundError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        # `load_benchmark_records`'s ValueError messages are already
        # generic/structural (line number, field names, a record id the
        # operator wrote themselves) -- never raw file content -- so it's
        # safe to print directly rather than swallowing it into something
        # vaguer.
        print(f"Refusing to run: invalid benchmark file. {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Loaded {len(records)} benchmark question(s) from {args.questions}")

    if args.mode == "dry-run":
        output_path = run_dry_run(records, args.results_dir)
        print(f"Dry run complete -- no network calls were made. Plan written to {output_path}")
        return

    if not args.live_opt_in:
        print(
            "Refusing to run in live mode without --i-understand-this-calls-external-apis.\n"
            "This would call the real Anthropic/OpenAI APIs and may incur cost.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    import asyncio

    try:
        output_path = asyncio.run(_run_live(records, args.results_dir, args.include_answer_text))
    except RuntimeError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"Live run complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
