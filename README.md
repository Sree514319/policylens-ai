# PolicyLens AI

A multi-model financial document intelligence assistant. Upload banking
policy PDFs, ask questions in natural language, and receive cited,
fact-checked answers — generated and cross-validated by both **Anthropic
Claude** and **OpenAI GPT** models — alongside transparent accuracy,
latency, and cost metrics.

> **Status:** 🚧 Active development. Project foundation, secure PDF
> ingestion, deterministic text chunking, and local-embedding vector
> search are implemented; LLM integration and the frontend are not yet
> built.

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
│   │   │   ├── documents.py   # POST /api/v1/documents/upload
│   │   │   └── search.py      # POST /api/v1/search
│   │   ├── core/
│   │   │   ├── config.py      # pydantic-settings configuration
│   │   │   └── exceptions.py  # Typed application exceptions
│   │   ├── models/            # Internal data models
│   │   ├── schemas/
│   │   │   ├── document.py    # Upload request/response schemas
│   │   │   └── search.py      # Search request/response schemas
│   │   ├── services/
│   │   │   ├── llm/            # Claude / OpenAI client wrappers, comparison logic
│   │   │   ├── ingestion/
│   │   │   │   ├── pdf_processor.py  # PDF validation & text extraction (PyMuPDF)
│   │   │   │   └── text_chunker.py   # Deterministic, page-aware chunking & citation IDs
│   │   │   ├── retrieval/
│   │   │   │   ├── embeddings.py     # EmbeddingProvider abstraction (local + fake)
│   │   │   │   └── vector_store.py   # ChromaDB indexing & semantic search
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
- [x] **Phase 4 — Vector store & retrieval**: local (no-API-key) embedding
      provider, ChromaDB persistence, and `POST /api/v1/search` semantic
      search over the chunks produced in Phase 3 (see
      [Vector Search & Retrieval](#9-vector-search--retrieval) below).
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
  "pages_with_text": 12,
  "indexed_chunk_count": 34
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
- `indexed_chunk_count` is the number of those chunks embedded and
  upserted into the vector store (see
  [Vector Search & Retrieval](#9-vector-search--retrieval) below). Because
  upserting is keyed by each chunk's deterministic `chunk_id`, re-uploading
  the same PDF re-indexes the same chunks in place instead of duplicating
  them.

**Error responses:**

| Status | Meaning                                              |
|--------|-------------------------------------------------------|
| `400`  | Wrong extension, wrong content type, invalid PDF signature, or empty file |
| `413`  | File exceeds the configured maximum upload size        |
| `422`  | File is corrupted, unparsable, or encrypted/password-protected |
| `500`  | Unexpected internal error, including an invalid chunking configuration or an embedding/collection configuration mismatch (generic message only — no stack traces or file paths are ever exposed) |
| `503`  | The vector store is unavailable (see below)             |

Uploaded PDFs are processed entirely in memory for this phase; the
original PDF bytes are never written to disk. Extracted chunk *text* and
its embedding are persisted to the vector store — see
[Vector Search & Retrieval](#9-vector-search--retrieval) for that privacy
decision.

### `POST /api/v1/search`

Runs a semantic search over previously indexed chunks and returns
ranked, citation-ready results.

**Request body:**

```json
{
  "query": "What is the overdraft fee?",
  "document_id": "3f1a9c...e2",
  "top_k": 5
}
```

- `query` — required, non-empty after trimming (max 2000 characters).
- `document_id` — optional; restricts the search to one previously
  uploaded document. If given and no chunks are indexed under it, the
  request returns `404`.
- `top_k` — optional, `1`-`50`; defaults to the server's configured
  `RETRIEVAL_TOP_K`.

**Success response — `200 OK`:**

```json
{
  "query": "What is the overdraft fee?",
  "result_count": 2,
  "results": [
    {
      "chunk_id": "b7e2...",
      "document_id": "3f1a9c...e2",
      "source_filename": "bank_policy.pdf",
      "page_number": 4,
      "excerpt": "Overdraft fees are $35 per occurrence, capped at...",
      "relevance_score": 0.87
    }
  ]
}
```

- `results` is ordered best-match first.
- `excerpt` is capped (see `EXCERPT_CHAR_LIMIT` in
  [Vector Search & Retrieval](#9-vector-search--retrieval)) — never the
  complete stored chunk text.
- Results scoring below the server's configured `MIN_RELEVANCE_SCORE` are
  filtered out before this response is built.

**Error responses:**

| Status | Meaning                                              |
|--------|-------------------------------------------------------|
| `404`  | `document_id` was given but has no indexed chunks       |
| `422`  | `query` is empty/whitespace-only, or `top_k` is out of the `1`-`50` range |
| `500`  | An embedding/collection configuration mismatch was detected (see below) |
| `503`  | The vector store is unavailable or returned a malformed result |

## 8. Text Chunking

After a PDF is validated and its text extracted, each page is split
independently into overlapping, citation-ready chunks. This runs
synchronously as part of the upload request; the resulting chunks are
then embedded and persisted to the vector store (see
[Vector Search & Retrieval](#9-vector-search--retrieval) below).

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

## 9. Vector Search & Retrieval

After chunking, each chunk is embedded and upserted into a persistent
ChromaDB collection as part of the same upload request, making it
immediately searchable.

**Embedding-provider abstraction.** `VectorStore` depends only on an
`EmbeddingProvider` interface (`embed_documents` / `embed_query`), never on
a concrete model, so the embedding backend can change without touching
storage or search code. Two providers exist:

- `LocalEmbeddingProvider` (production) — ChromaDB's bundled local ONNX
  model (`all-MiniLM-L6-v2`, 384 dimensions). Runs entirely on-device; the
  small model file is downloaded once on first use and cached under
  `~/.cache/chroma`. **No OpenAI or Anthropic API calls, and no API key,
  are involved anywhere in this phase.**
- `FakeEmbeddingProvider` (tests only) — a deterministic, dependency-free,
  hash-based bag-of-words embedding. No model download, no network
  access. Same text always produces the same vector, and texts sharing
  more vocabulary end up more similar, which is enough to exercise
  indexing, idempotency, ranking, and filtering in tests.

**Storage.** Each chunk is stored in Chroma with its embedding, its text
(so it can be returned as a search excerpt), and metadata: `document_id`,
`chunk_index`, `page_number`, `source_filename`, `start_character`,
`end_character`, and `content_hash`. Indexing uses `upsert`, keyed by each
chunk's deterministic `chunk_id` (see [Text Chunking](#8-text-chunking)),
so re-uploading identical content re-indexes the same entries in place
instead of duplicating them.

**Distance-to-relevance-score formula.** The collection uses cosine space,
where `distance = 1 - cosine_similarity`. Since cosine similarity ranges
over `[-1, 1]`, distance ranges over `[0, 2]`. That is linearly remapped
onto a `[0.0, 1.0]` relevance score, where `1.0` is the best possible
match and `0.0` is the worst:

```
relevance_score = 1 - (distance / 2)
```

(clamped to `[0.0, 1.0]` defensively against floating-point drift at the
extremes).

**Search.** `VectorStore.search()` supports both document-scoped search
(pass `document_id`) and cross-document search (omit it), a configurable
`top_k`, and post-filters results below `MIN_RELEVANCE_SCORE`. It never
returns raw ChromaDB objects — only the plain `SearchResult` dataclass
(and, at the API layer, capped excerpts) cross that boundary.

**Configuration** (`Settings` / `.env`):

| Variable                    | Default                  | Meaning                                              |
|-------------------------------|---------------------------|---------------------------------------------------------|
| `CHROMA_PERSIST_DIRECTORY`   | `./data/vector_store`     | Where ChromaDB persists its SQLite/HNSW files.          |
| `CHROMA_COLLECTION_NAME`     | `policylens_documents`    | The Chroma collection name.                             |
| `EMBEDDING_MODEL_NAME`       | `all-MiniLM-L6-v2`        | Documents which local model `LocalEmbeddingProvider` wraps. |
| `RETRIEVAL_TOP_K`            | `5`                       | Default result count when a search request omits `top_k`. |
| `MIN_RELEVANCE_SCORE`        | `0.0`                     | Results scoring below this are filtered out.            |

**Safety & consistency guarantees.**

- **Embedding/collection drift is rejected, not silently mixed.** Every
  collection records the embedding provider's `name` and `dimension`
  (plus the HNSW space) as metadata at first creation. Because ChromaDB
  ignores the `metadata` argument on an *already-existing* collection,
  that recorded value reflects whichever provider first created it. On
  every `VectorStore` startup, the active provider is compared against
  it; if `EMBEDDING_MODEL_NAME` changed since the collection was built,
  or the collection isn't in cosine space, startup fails fast with a
  typed `EmbeddingConfigurationMismatchError` (`500`) instead of quietly
  storing incompatible vectors side by side. (A collection with no
  tracking metadata at all — i.e. pre-existing data this code never
  wrote — can't be verified on provider/dimension and is let through.)
- **Malformed embeddings are rejected before reaching Chroma.** Every
  batch returned by the embedding provider is checked for the right
  shape (a list of same-length numeric vectors, no `NaN`s) before being
  sent to `upsert`/`query`. A provider bug here raises `VectorStoreError`
  (`503`) rather than corrupting the index or crashing with an opaque
  ChromaDB error.
- **Large uploads are indexed in safe batches.** ChromaDB rejects a
  single `upsert` call beyond its backend's max batch size (queried at
  startup via `client.get_max_batch_size()`, ~5,461 items for the local
  SQLite backend at time of writing). `upsert_chunks` transparently
  splits into batches at that limit, so a very large PDF can't fail on
  this alone.
- **`top_k` cannot exceed its bounds from either direction.** A client-
  supplied `top_k` is schema-bounded to `[1, 50]`. The server-side
  default (`RETRIEVAL_TOP_K`, used when a request omits `top_k`) is
  additionally clamped into that same range at request time, so a
  misconfigured `.env` value can't push an unbounded result count into
  Chroma.
- **A malformed Chroma response is rejected, not silently mis-paired.**
  If `ids`/`documents`/`metadatas`/`distances` in a query result ever
  came back with mismatched lengths, naively zipping them could drop or
  misalign results. `VectorStore.search()` checks the lengths match
  before pairing them up and raises `VectorStoreError` otherwise.

**Privacy decision: what gets persisted, and why.** Unlike the original
PDF bytes (never written to disk — see [Phase 2](#7-api-reference)),
chunk *text* and its embedding **are** persisted, in
`CHROMA_PERSIST_DIRECTORY` (`data/vector_store/` by default), so that
search can return an excerpt and so the index survives a server restart.
This is a deliberate trade-off: semantic search is not possible without
storing something to search against. Mitigations:

- That directory is git-ignored (`data/vector_store/*` in `.gitignore`,
  keeping only `.gitkeep`) — indexed content is never committed to the
  repository.
- Only a capped excerpt (`EXCERPT_CHAR_LIMIT` = 300 characters) is ever
  returned by the search API — not the full stored chunk text.
- No chunk text, query text, or excerpt is ever logged — only IDs,
  counts, and durations (see [Ethical & Privacy Considerations](#11-ethical--privacy-considerations)).
- Nothing here is sent to a third-party API: embedding happens locally,
  and storage is local disk.

## 10. Testing

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

Phase 4 adds: chunk indexing, idempotent re-indexing (no duplication on
re-upload), metadata round-trip, semantic result ordering, the
distance-to-relevance-score formula, document-scoped vs. cross-document
search, `top_k` limiting, minimum-relevance-score filtering, excerpt
capping, persistence across `VectorStore` re-instantiation (proving data
survives on disk, not just in memory), unavailable/corrupted vector-store
error handling, unknown-`document_id` rejection, empty/whitespace-query
and `top_k`-bounds validation, the search response's exact key set, and a
repository-hygiene check that no generated Chroma files are tracked by
Git.

The Phase 4 pre-commit review adds targeted regression coverage for:
malformed/wrong-dimension/non-numeric/`NaN` embedding vectors from a
provider; an embedding-provider or HNSW-space mismatch against an
already-existing collection (including a legacy collection created
without any tracking metadata); large-batch indexing split across
multiple `upsert` calls at the backend's max batch size; a malformed or
length-inconsistent Chroma query result; a failing local-embedding-
provider construction; and a misconfigured `RETRIEVAL_TOP_K` being
clamped rather than passed straight to Chroma.

All Phase 4 tests use an isolated temporary directory
(`tmp_chroma_dir`/`vector_store` fixtures) and the deterministic
`FakeEmbeddingProvider` — no model download or network access ever
happens during the test suite.

## 11. Ethical & Privacy Considerations

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
- **Vector store persistence is a documented trade-off, not an oversight.**
  Chunk text and embeddings are persisted locally (see
  [Vector Search & Retrieval](#9-vector-search--retrieval)) so search can
  work at all; that directory stays git-ignored, only capped excerpts are
  ever returned by the API, and no query, chunk, or excerpt text is ever
  logged.

## 12. License

See [LICENSE](./LICENSE).
