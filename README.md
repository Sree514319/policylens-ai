# PolicyLens AI

A multi-model financial document intelligence assistant. Upload banking
policy PDFs, ask questions in natural language, and receive cited,
fact-checked answers — generated and cross-validated by both **Anthropic
Claude** and **OpenAI GPT** models — alongside transparent accuracy,
latency, and cost metrics.

> **Status:** 🚧 Active development. Project foundation, secure PDF
> ingestion, and deterministic text chunking are implemented; vector
> retrieval, LLM integration, and the frontend are not yet built.

---

## 1. Overview

Banking and financial institutions publish dense, jargon-heavy policy
documents (terms & conditions, compliance manuals, disclosure statements)
that are difficult for both customers and internal staff to query
accurately. PolicyLens AI lets a user upload one or more PDF policy
documents and ask natural-language questions against them. It then:

- Retrieves the most relevant passages from the document(s) using a
  vector store (retrieval-augmented generation).
- Sends the question and retrieved context to **both** Claude and GPT
  models in parallel.
- Returns each model's answer with inline **citations** back to the
  source passage/page.
- **Compares** the two models' answers side by side.
- Flags any **unsupported claims** — statements in an answer that are
  not backed by the retrieved source text.
- **Masks sensitive information** (e.g., account numbers, SSNs, emails)
  before it is displayed, logged, or sent to a third-party LLM.
- Displays **accuracy, latency, and cost** metrics per model, per query.

The goal is a portfolio-quality demonstration of responsible, production-
style LLM application engineering: multi-model orchestration, grounded
generation, evaluation, and privacy-aware design — not just a chatbot
wrapper.

## 2. Planned Features

- 📄 **PDF ingestion** — parse and chunk banking policy documents (PyMuPDF).
- 🔍 **Semantic retrieval** — embed and search document chunks via ChromaDB.
- 🤖 **Dual-model Q&A** — query Claude and OpenAI models concurrently.
- 📌 **Citations** — every answer links back to the source page/passage.
- ⚖️ **Model comparison view** — side-by-side answers, agreement/disagreement.
- 🚩 **Unsupported-claim detection** — flag statements not grounded in
  retrieved context.
- 🛡️ **PII/sensitive-data masking** — redact account numbers, SSNs, emails,
  phone numbers, etc. before display or before leaving the system.
- 📊 **Metrics dashboard** — accuracy proxy, latency, and per-query token
  cost for each model.
- 🐳 **Containerized deployment** — Docker-based local and cloud deployment.
- ✅ **Automated tests** — unit and integration coverage via pytest.

## 3. Architecture

```
                        ┌─────────────────────────┐
                        │   Streamlit Frontend     │
                        │  (upload, chat, compare,  │
                        │   metrics dashboard)      │
                        └────────────┬─────────────┘
                                     │ HTTP (REST)
                                     ▼
                        ┌─────────────────────────┐
                        │     FastAPI Backend      │
                        │  ┌───────────────────┐   │
                        │  │  API layer (v1)   │   │
                        │  └─────────┬─────────┘   │
                        │            ▼             │
                        │  ┌───────────────────┐   │
                        │  │  Service layer    │   │
                        │  │  - ingestion      │───┼──▶ PyMuPDF (PDF parsing)
                        │  │  - retrieval      │───┼──▶ ChromaDB (vector store)
                        │  │  - llm            │───┼──▶ Claude API / OpenAI API
                        │  │  - privacy        │   │   (PII masking)
                        │  │  - evaluation     │   │   (citations, unsupported
                        │  └───────────────────┘   │    claims, cost/latency)
                        │            │             │
                        │            ▼             │
                        │  ┌───────────────────┐   │
                        │  │ Pydantic schemas  │   │
                        │  │ Core config       │   │
                        │  └───────────────────┘   │
                        └─────────────────────────┘
```

**Design principles**

- **Separation of concerns** — API routing, business logic (services),
  data contracts (schemas), and configuration (core) are isolated so each
  can be tested and evolved independently.
- **Provider-agnostic LLM layer** — Claude and OpenAI are accessed through
  a common internal interface so additional providers can be added without
  touching calling code.
- **Privacy by design** — sensitive-data masking is a distinct service
  invoked before any content is persisted, logged, or forwarded to a
  third-party API.
- **Observability built in** — latency and token/cost accounting are
  first-class parts of the LLM service layer, not an afterthought.
- **Independently deployable tiers** — the FastAPI backend and Streamlit
  frontend are separate processes/containers, communicating over HTTP,
  so either can be scaled or replaced independently.

## 4. Technology Stack

| Layer                  | Technology                          |
|-------------------------|--------------------------------------|
| Backend API             | Python, FastAPI                     |
| Frontend UI             | Streamlit                           |
| LLM Providers           | Anthropic Claude API, OpenAI API    |
| Vector Store            | ChromaDB                            |
| PDF Parsing             | PyMuPDF                             |
| Data Validation         | Pydantic                            |
| Testing                 | pytest                              |
| Containerization        | Docker                              |

## 5. Project Structure

```
policylens-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entry point (GET /health, routers)
│   │   ├── api/v1/
│   │   │   └── documents.py   # POST /api/v1/documents/upload
│   │   ├── core/
│   │   │   ├── config.py      # pydantic-settings configuration
│   │   │   └── exceptions.py  # Typed application exceptions
│   │   ├── models/            # Internal data models
│   │   ├── schemas/
│   │   │   └── document.py    # Upload request/response schemas
│   │   ├── services/
│   │   │   ├── llm/            # Claude / OpenAI client wrappers, comparison logic
│   │   │   ├── ingestion/
│   │   │   │   ├── pdf_processor.py  # PDF validation & text extraction (PyMuPDF)
│   │   │   │   └── text_chunker.py   # Deterministic, page-aware chunking & citation IDs
│   │   │   ├── retrieval/      # ChromaDB indexing & semantic search, citations
│   │   │   ├── privacy/        # PII/sensitive-data detection & masking
│   │   │   └── evaluation/     # Unsupported-claim detection, accuracy, latency, cost
│   │   └── utils/
│   │       └── uploads.py     # Size-bounded UploadFile streaming helper
│   └── tests/                 # Backend unit/integration tests (pytest)
├── frontend/
│   └── streamlit_app/
│       ├── pages/            # Multi-page Streamlit views
│       └── components/       # Reusable UI components
├── data/
│   ├── sample_docs/          # Example policy PDFs for local dev/testing
│   └── vector_store/         # Local ChromaDB persistence (gitignored)
├── docker/                   # Dockerfiles / container configs
├── docs/                     # Design docs, diagrams, ADRs
├── scripts/                  # Dev/ops utility scripts
├── .env.example              # Environment variable template (no real secrets)
├── pytest.ini                 # Pytest configuration (adds backend/ to path)
├── requirements.txt           # Python dependencies
└── README.md
```

## 6. Setup Roadmap

This repository is being built incrementally. Planned phases:

- [x] **Phase 0 — Project foundation**: repository scaffolding, folder
      structure, README, dependency baseline.
- [x] **Phase 1 — Core configuration**: environment/config management via
      `pydantic-settings`, typed exceptions, logging setup.
- [x] **Phase 2 — Secure PDF ingestion**: `POST /api/v1/documents/upload`,
      PyMuPDF-based validation and text extraction, in-memory processing
      only (see [API Reference](#7-api-reference) below).
- [x] **Phase 3 — Deterministic text chunking**: page-aware, word-boundary
      chunking with configurable size/overlap, deterministic chunk IDs,
      and citation metadata (see [Text Chunking](#8-text-chunking) below).
- [ ] **Phase 4 — Vector store & retrieval**: ChromaDB integration,
      embedding pipeline, semantic search over the chunks produced in
      Phase 3.
- [ ] **Phase 5 — LLM integration**: Claude and OpenAI client wrappers,
      prompt templates, concurrent dual-model querying.
- [ ] **Phase 6 — Privacy layer**: sensitive-information detection and
      masking applied before display/storage/outbound calls.
- [ ] **Phase 7 — Evaluation layer**: unsupported-claim detection,
      accuracy scoring, latency and cost tracking.
- [ ] **Phase 8 — Frontend**: Streamlit upload flow, chat interface,
      model comparison view, metrics dashboard.
- [ ] **Phase 9 — Testing**: consolidated pytest coverage review across
      services and API routes.
- [ ] **Phase 10 — Containerization & deployment**: Dockerfiles, compose
      setup, deployment documentation.

### Local setup

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd policylens-ai

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# then fill in real API keys and settings in .env (never commit .env)

# 5. Run the backend
cd backend
uvicorn app.main:app --reload
# API docs available at http://127.0.0.1:8000/docs

# 6. Run the frontend (once implemented)
# streamlit run frontend/streamlit_app/app.py
```

## 7. API Reference

### `GET /health`

Liveness check. Returns `200 OK`:

```json
{ "status": "ok" }
```

### `POST /api/v1/documents/upload`

Accepts a single PDF policy document as `multipart/form-data` (field name
`file`), validates it, extracts its text **in memory only**, and returns
metadata — never the full document text.

**Validation performed, in order:**

1. File is not empty.
2. Filename has a `.pdf` extension.
3. Request content type is exactly `application/pdf`.
4. File begins with the `%PDF-` signature.
5. Upload size does not exceed `MAX_UPLOAD_SIZE_MB` (checked while
   streaming, before the whole file is buffered).
6. File parses as a well-formed PDF (PyMuPDF).
7. File is not encrypted or password-protected.
8. File contains at least one page.

**Success response — `201 Created`:**

```json
{
  "document_id": "3f1a9c...e2 (sha256 hex digest, 64 chars)",
  "filename": "bank_policy.pdf",
  "page_count": 12,
  "character_count": 24831,
  "status": "processed",
  "preview": "First ~200 characters of the extracted text...",
  "chunk_count": 34,
  "pages_with_text": 12
}
```

- `document_id` is a SHA-256 hash of the file's raw bytes, so re-uploading
  identical content (even under a different filename) always yields the
  same ID.
- `filename` is sanitized (directory components and unsafe characters
  stripped) before being echoed back.
- `preview` is capped at ~200 characters — the full extracted text is
  never returned in the response and never written to logs.
- `chunk_count` and `pages_with_text` summarize the Phase 3 chunking pass
  (see [Text Chunking](#8-text-chunking) below) — the chunks themselves,
  and their text, are never included in the response.

**Error responses:**

| Status | Meaning                                              |
|--------|-------------------------------------------------------|
| `400`  | Wrong extension, wrong content type, invalid PDF signature, or empty file |
| `413`  | File exceeds the configured maximum upload size        |
| `422`  | File is corrupted, unparsable, or encrypted/password-protected |
| `500`  | Unexpected internal error, including an invalid chunking configuration (generic message only — no stack traces or file paths are ever exposed) |

Uploaded PDFs are processed entirely in memory for this phase; nothing is
written to disk.

## 8. Text Chunking

After a PDF is validated and its text extracted, each page is split
independently into overlapping, citation-ready chunks. This runs
synchronously as part of the upload request; chunks are not yet persisted
or embedded (that begins in Phase 4).

**Why per-page, not per-document:** chunking each page's text in isolation
means a chunk never spans a page boundary, so every chunk's `page_number`
is always an exact, unambiguous citation back to the source PDF page.

**Algorithm** (no LangChain or other chunking library — plain, dependency-free
string slicing so behavior is fully deterministic):

1. **Normalize whitespace.** Runs of spaces, tabs, and newlines collapse to
   a single space; leading/trailing whitespace is trimmed. Only whitespace
   is touched — digits, currency symbols, decimal points, and percentages
   are never altered, so a figure like `$1,000.00` or `12.5%` is preserved
   exactly.
2. **Slide a window** of `CHUNK_SIZE` characters across the normalized page
   text, advancing by `CHUNK_SIZE - CHUNK_OVERLAP` characters each step, so
   consecutive chunks share `CHUNK_OVERLAP` characters of context.
3. **Snap each cut to the nearest preceding space** so chunks avoid
   splitting a word, falling back to a hard cut only when a single token
   (e.g. a long account number) exceeds `CHUNK_SIZE` with no space to
   break on.
4. **Merge a too-small trailing remainder** (shorter than
   `MIN_CHUNK_LENGTH`) into the previous chunk instead of emitting a tiny,
   low-context final chunk.
5. **Skip empty/whitespace-only pages** entirely — they contribute zero
   chunks and never crash the pipeline.

The same page text and settings always produce byte-identical chunks and
chunk IDs — there is no randomness or wall-clock dependency anywhere in
the pipeline.

**Configuration** (`Settings` / `.env`):

| Variable            | Default | Meaning                                              |
|----------------------|---------|-------------------------------------------------------|
| `CHUNK_SIZE`         | `1000`  | Maximum characters per chunk.                         |
| `CHUNK_OVERLAP`      | `150`   | Shared characters between consecutive chunks. Must be smaller than `CHUNK_SIZE`. |
| `MIN_CHUNK_LENGTH`   | `50`    | Trailing remainders shorter than this are merged into the previous chunk. |

An inconsistent configuration (e.g. `CHUNK_OVERLAP >= CHUNK_SIZE`, a
negative value, or `MIN_CHUNK_LENGTH > CHUNK_SIZE`) raises a typed
`InvalidChunkConfigurationError`, surfaced as an HTTP `500` — this reflects
bad server configuration, not a bad upload.

**Each chunk carries:**

| Field             | Description                                                        |
|--------------------|---------------------------------------------------------------------|
| `chunk_id`         | SHA-256 of `document_id:page_number:chunk_index:text` — deterministic and unique per document/position/content. |
| `document_id`      | The parent document's content hash.                                |
| `chunk_index`      | 0-based position across the whole document, in page order.         |
| `page_number`      | The 1-based source PDF page — the citation anchor.                 |
| `source_filename`  | The sanitized filename the chunk came from.                        |
| `text`             | The normalized chunk text.                                         |
| `character_count`  | `len(text)`.                                                        |
| `start_character` / `end_character` | Character offsets into the page's normalized text.  |
| `content_hash`     | SHA-256 of `text` alone — independent of `document_id`, so identical content hashes identically even across different documents. |

Chunk objects exist only inside the service pipeline for this phase — they
are never persisted, embedded, logged, or returned over the API. The
public upload response only ever exposes the summary counts `chunk_count`
and `pages_with_text`.

## 9. Testing

The backend test suite uses `pytest` and FastAPI's `TestClient`. All test
PDFs (including a password-protected one) are generated in memory with
PyMuPDF — no binary fixture files are committed to the repository.

```bash
# From the repository root, with the virtual environment activated:
pip install -r requirements.txt
pytest
```

Coverage includes: the health endpoint, a valid multi-page upload,
rejection of non-PDF files, wrong content type, invalid PDF signature,
empty files, corrupted PDFs, oversized files, encrypted/password-protected
PDFs, filename sanitization, and document-ID/page-metadata stability.

Phase 3 adds: single-chunk short text, multi-chunk overlapping long text,
deterministic chunk IDs, document/page/source metadata correctness, empty
and whitespace-only pages, whitespace normalization, word-boundary
behavior, invalid size/overlap/minimum-length configuration, minimum
chunk length merging, multi-page PDF chunking, and API-level checks that
`chunk_count`/`pages_with_text` are present while no chunk or page text
ever appears in the response.

## 10. Ethical & Privacy Considerations

- **No real API keys or secrets are ever committed.** `.env` is
  git-ignored; only `.env.example` (placeholder variable names) is
  tracked.
- **Sensitive data masking is a first-class feature, not an add-on.**
  Account numbers, SSNs, emails, and similar identifiers are intended to
  be detected and redacted before content is displayed, logged, or sent
  to any third-party LLM provider.
- **Unsupported-claim detection** aims to reduce hallucination risk in a
  financial context, where an incorrect or fabricated answer about a
  policy term can have real consequences for a reader.
- **Citations are mandatory, not optional**, so users can always verify
  an answer against the original source document rather than trusting
  the model output blindly.
- **Sample documents only.** This project is intended to be exercised
  against synthetic or publicly available sample banking policy
  documents — not real customer data.
- **Model outputs are informational, not advice.** This project is a
  technical/portfolio demonstration and is not intended to provide
  financial, legal, or compliance advice.

## 11. License

See [LICENSE](./LICENSE).
