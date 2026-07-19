"""Canonical sample JSON payloads, shaped exactly like the backend's real
response schemas (see backend/app/schemas/*.py). Reused across test
files so `api_client.py`'s parsing and `components/render.py`'s
rendering are always exercised against the same realistic data.
"""

SAMPLE_HEALTH = {"status": "ok"}

SAMPLE_UPLOAD_RESPONSE = {
    "document_id": "3f1a9c" + "e2" * 29,
    "filename": "bank_policy.pdf",
    "page_count": 3,
    "character_count": 4200,
    "status": "processed",
    "preview": "This document describes account terms and fee schedules...",
    "chunk_count": 6,
    "pages_with_text": 3,
    "indexed_chunk_count": 6,
    "pii_detected": True,
    "pii_entity_count": 2,
    "pii_categories": ["EMAIL", "SSN"],
}

SAMPLE_SEARCH_RESPONSE = {
    "query": "overdraft fee",
    "query_was_masked": False,
    "result_count": 2,
    "results": [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source_filename": "bank_policy.pdf",
            "page_number": 4,
            "excerpt": "Overdraft fees are $35 per occurrence, capped at 3 per day.",
            "relevance_score": 0.87,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "source_filename": "bank_policy.pdf",
            "page_number": 5,
            "excerpt": "Overdraft protection may be enrolled at any branch.",
            "relevance_score": 0.61,
        },
    ],
}

SAMPLE_ANSWER_RESPONSE = {
    "question": "What is the overdraft fee?",
    "query_was_masked": False,
    "evidence_count": 2,
    "model_results": [
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "status": "success",
            "answer": "The overdraft fee is $35 per occurrence, capped at 3 per day [S1].",
            "citations": [
                {
                    "source_label": "S1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "source_filename": "bank_policy.pdf",
                    "page_number": 4,
                    "excerpt": "Overdraft fees are $35 per occurrence, capped at 3 per day.",
                    "relevance_score": 0.87,
                }
            ],
            "latency_ms": 842.3,
            "input_tokens": 512,
            "output_tokens": 41,
            "error": None,
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "status": "error",
            "answer": "",
            "citations": [],
            "latency_ms": 30000.0,
            "input_tokens": None,
            "output_tokens": None,
            "error": "The OpenAI API request timed out.",
        },
    ],
}

SAMPLE_COMPARE_RESPONSE = {
    "question": "What is the overdraft fee?",
    "query_was_masked": False,
    "evidence_count": 2,
    "model_results": [
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "status": "success",
            "answer": "The overdraft fee is $35 per occurrence [S1].",
            "citations": [
                {
                    "source_label": "S1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "source_filename": "bank_policy.pdf",
                    "page_number": 4,
                    "excerpt": "Overdraft fees are $35 per occurrence.",
                    "relevance_score": 0.87,
                }
            ],
            "latency_ms": 842.3,
            "input_tokens": 512,
            "output_tokens": 41,
            "error": None,
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "status": "success",
            "answer": "It costs $35 per occurrence [S1].",
            "citations": [
                {
                    "source_label": "S1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "source_filename": "bank_policy.pdf",
                    "page_number": 4,
                    "excerpt": "Overdraft fees are $35 per occurrence.",
                    "relevance_score": 0.87,
                }
            ],
            "latency_ms": 962.8,
            "input_tokens": 480,
            "output_tokens": 38,
            "error": None,
        },
    ],
    "provider_metrics": [
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "status": "success",
            "latency_ms": 842.3,
            "input_tokens": 512,
            "output_tokens": 41,
            "estimated_cost_usd": 0.002181,
            "valid_citation_count": 1,
            "citation_coverage": 0.5,
            "mean_citation_relevance": 0.87,
            "grounded_term_ratio": 0.8,
            "answer_length": 8,
            "evaluation_notes": [
                "grounded_term_ratio is a lexical overlap heuristic (normalized answer "
                "terms found in cited excerpts) and does NOT verify factual correctness."
            ],
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "status": "success",
            "latency_ms": 962.8,
            "input_tokens": 480,
            "output_tokens": 38,
            "estimated_cost_usd": None,
            "valid_citation_count": 1,
            "citation_coverage": 0.5,
            "mean_citation_relevance": 0.87,
            "grounded_term_ratio": None,
            "answer_length": 7,
            "evaluation_notes": [
                "Per-token pricing is not configured for this provider; estimated_cost_usd not computed.",
                "Answer had no meaningful terms after normalization; grounded_term_ratio not computed.",
            ],
        },
    ],
    "comparison": {
        "answer_agreement_score": 0.94,
        "latency_difference_ms": -120.5,
        "estimated_cost_difference_usd": None,
        "comparison_status": "both_successful",
        "comparison_notes": [
            "latency: anthropic had the lower value (anthropic=842.30 ms, openai=962.80 ms).",
            "estimated cost: could not be compared (unavailable for one or both providers).",
        ],
    },
}

SAMPLE_ERROR_RESPONSE = {"detail": "document_id was given but has no indexed chunks."}
