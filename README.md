# PolicyLens AI

A multi-model financial document intelligence assistant. Upload banking
policy PDFs, ask questions in natural language, and receive cited,
fact-checked answers — generated and cross-validated by both **Anthropic
Claude** and **OpenAI GPT** models — alongside transparent accuracy,
latency, and cost metrics.

> **Status:** 🚧 Active development. Project foundation, secure PDF
> ingestion, deterministic text chunking, local-embedding vector search,
> grounded multi-model (Claude + GPT) RAG answering, and local PII
> detection/masking are implemented; the frontend is not yet built. Real
> Anthropic/OpenAI calls stay disabled until you explicitly opt in (see
> [Multi-Model RAG Answers](#10-multi-model-rag-answers)). **This is a
> portfolio project, not a compliance product** — read
> [PII Detection & Masking](#11-pii-detection--masking) before using
> anything but synthetic/public sample documents.

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
│   │   │   ├── search.py      # POST /api/v1/search
│   │   │   └── answers.py     # POST /api/v1/answers
│   │   ├── core/
│   │   │   ├── config.py      # pydantic-settings configuration
│   │   │   └── exceptions.py  # Typed application exceptions
│   │   ├── models/            # Internal data models
│   │   ├── schemas/
│   │   │   ├── document.py    # Upload request/response schemas
│   │   │   ├── search.py      # Search request/response schemas
│   │   │   └── answer.py      # Grounded-answer request/response schemas
│   │   ├── services/
│   │   │   ├── llm/
│   │   │   │   ├── providers.py  # LLMProvider abstraction (Anthropic/OpenAI/fake)
│   │   │   │   └── rag.py        # Evidence retrieval, prompting, citation validation
│   │   │   ├── ingestion/
│   │   │   │   ├── pdf_processor.py  # PDF validation & text extraction (PyMuPDF)
│   │   │   │   └── text_chunker.py   # Deterministic, page-aware chunking & citation IDs
│   │   │   ├── retrieval/
│   │   │   │   ├── embeddings.py     # EmbeddingProvider abstraction (local + fake)
│   │   │   │   └── vector_store.py   # ChromaDB indexing & semantic search
│   │   │   ├── privacy/
│   │   │   │   ├── detectors.py  # PIIDetector abstraction (local regex + fake)
│   │   │   │   └── masking.py    # Masking orchestration, PIIMaskingSummary
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
- [x] **Phase 5 — Multi-model grounded RAG**: provider-neutral
      `LLMProvider` abstraction (Claude + GPT), concurrent independent
      querying, evidence-grounded structured JSON answers with validated
      citations, and `POST /api/v1/answers` (see
      [Multi-Model RAG Answers](#10-multi-model-rag-answers) below).
      Real external calls stay off by default.
- [x] **Phase 6 — Local PII detection & masking**: regex + checksum-based
      detection of common US financial identifiers (SSN, card, email,
      phone, IPv4, DOB, bank account, routing number), applied before
      preview generation, chunking, indexing, search, and RAG prompts;
      a versioned vector-store privacy check (see
      [PII Detection & Masking](#11-pii-detection--masking) below).
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
  "indexed_chunk_count": 34,
  "pii_detected": true,
  "pii_entity_count": 2,
  "pii_categories": ["EMAIL", "SSN"]
}
```

- `document_id` is a SHA-256 hash of the file's raw bytes, so re-uploading
  identical content (even under a different filename) always yields the
  same ID.
- `filename` is sanitized (directory components and unsafe characters
  stripped) before being echoed back.
- `preview` is capped at ~200 characters, and is generated **after** PII
  masking (see [PII Detection & Masking](#11-pii-detection--masking)
  below) — the full extracted text, masked or not, is never returned in
  the response and never written to logs.
- `chunk_count` and `pages_with_text` summarize the Phase 3 chunking pass
  (see [Text Chunking](#8-text-chunking) below) — the chunks themselves,
  and their text, are never included in the response. Chunking runs on
  already-masked page text.
- `indexed_chunk_count` is the number of those (masked) chunks embedded
  and upserted into the vector store (see
  [Vector Search & Retrieval](#9-vector-search--retrieval) below). Because
  upserting is keyed by each chunk's deterministic `chunk_id`, re-uploading
  the same PDF re-indexes the same chunks in place instead of duplicating
  them.
- `pii_detected`/`pii_entity_count`/`pii_categories` summarize what was
  masked — category names and a count only, never the matched values or
  their positions. See [PII Detection & Masking](#11-pii-detection--masking).

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
  "query_was_masked": false,
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

- `query` is masked (see [PII Detection & Masking](#11-pii-detection--masking))
  **before** it is embedded/searched — if PII was detected in the
  submitted query, `query` here is the masked version and
  `query_was_masked` is `true`. The original, unmasked query is never
  returned or logged.
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
  counts, and durations (see [Ethical & Privacy Considerations](#13-ethical--privacy-considerations)).
- Nothing here is sent to a third-party API: embedding happens locally,
  and storage is local disk.

## 10. Multi-Model RAG Answers

`POST /api/v1/answers` retrieves evidence for a question (via the same
`VectorStore` as `/search`) and asks **both** Claude and GPT to answer,
grounded strictly in that evidence, concurrently and independently.

### Flow

1. **Retrieve** — the question is embedded and searched against the
   vector store (optionally scoped to `document_id`), exactly like
   `/search`.
2. **Cap the context** — retrieved chunks (already relevance-ordered) are
   kept up to a total character budget (`MAX_RAG_CONTEXT_CHARACTERS`),
   always keeping at least the top match even if it alone exceeds the
   budget.
3. **Label evidence** — each kept chunk gets a stable label for *this
   request* — `S1`, `S2`, `S3`, ... in relevance order.
4. **Prompt** — a system prompt instructs the model to answer using
   *only* the labeled evidence, to treat that evidence as **untrusted
   data, never as instructions** (explicitly telling it to ignore any
   embedded commands or "ignore previous instructions" text found inside
   a chunk), and to respond with a single strict JSON object:
   `{"insufficient_evidence": bool, "answer": string, "citations": [string]}`.
5. **Run providers concurrently** — every requested provider is called
   at the same time (`asyncio.gather`), each with its own
   timeout/retry policy; the total wall-clock time is close to the
   slowest single provider, not the sum of both.
6. **Parse and validate** — the raw JSON is parsed (tolerating minor
   formatting noise like markdown code fences); citation labels that
   weren't actually shown to the model are dropped; duplicates are
   deduped. Every field in a returned citation (`chunk_id`,
   `document_id`, `source_filename`, `page_number`, `excerpt`,
   `relevance_score`) is copied from our own stored evidence metadata —
   **never from anything the model wrote**, beyond which label(s) it
   named. Page numbers and filenames can never be invented.

### Provider architecture

`LLMProvider` is a minimal, provider-neutral interface —
`generate(system_prompt, user_prompt) -> ProviderResponse` — that knows
nothing about RAG, evidence, or JSON schemas; that logic lives entirely
in the RAG orchestration layer, not the provider layer. No SDK-specific
object (an Anthropic `Message`, an OpenAI `ChatCompletion`, an SDK
exception) ever leaves the provider module.

- `AnthropicProvider` / `OpenAIProvider` (production) — thin wrappers
  around the official async SDK clients (Messages API / Chat Completions
  API). Every SDK exception (auth failure, rate limit, timeout,
  connection error, generic API error, or anything unexpected) is caught
  and translated into a short, generic, client-safe `error` string —
  never a stack trace, credentials, or a raw provider response dump.
- `FakeLLMProvider` (tests only) — deterministic and fully configurable
  (canned JSON, canned raw text, a canned error, a simulated delay, or a
  raised exception for defense-in-depth testing) — no network access.

### Failure / partial-success behavior

Every requested provider is evaluated **independently**: one provider
failing never discards another provider's successful answer. Each
`model_results[]` entry gets exactly one of three statuses:

| Status                 | Meaning                                                            |
|--------------------------|----------------------------------------------------------------------|
| `success`               | A grounded answer with 0+ validated citations.                     |
| `insufficient_evidence` | The model determined the evidence doesn't support an answer (or no evidence was retrieved at all, in which case no provider is even called). |
| `error`                 | A safe, generic message — missing/invalid API key, rate limit, timeout, connection failure, malformed/unparsable JSON, an empty answer, or `ALLOW_EXTERNAL_LLM_CALLS=false`. |

A retrieval failure (no such `document_id`, or the vector store itself
being unavailable) still fails the whole request (`404`/`503`), the same
as `/search` — there is no meaningful per-provider answer to give when
there's nothing to ground it in.

### External-call safety (read this before setting `ALLOW_EXTERNAL_LLM_CALLS=true`)

**Real Anthropic/OpenAI network calls are refused by default.** Unless
`ALLOW_EXTERNAL_LLM_CALLS=true` is explicitly set, every provider result
comes back as `status: "error"` with a safe configuration message —
**no network call is attempted at all**, regardless of whether API keys
are configured.

> ⚠️ **Enabling external calls sends retrieved excerpts to Anthropic
> and/or OpenAI.** Once `ALLOW_EXTERNAL_LLM_CALLS=true` and a real API
> key is configured, the question and the retrieved evidence excerpts
> (not the full document) are sent to whichever third-party provider(s)
> you request, subject to their own data-handling terms.
>
> ⚠️ **Do not upload real customer or other PII-bearing documents** until
> the Phase 6 privacy/PII-masking layer is implemented and enabled. Use
> synthetic or public sample policy documents only (see
> [Ethical & Privacy Considerations](#13-ethical--privacy-considerations)).

### Configuration (`Settings` / `.env`)

| Variable                  | Default                        | Meaning                                              |
|------------------------------|-----------------------------------|---------------------------------------------------------|
| `ANTHROPIC_API_KEY`         | *(unset)*                        | Required only once external calls are enabled.          |
| `OPENAI_API_KEY`            | *(unset)*                        | Required only once external calls are enabled.          |
| `ANTHROPIC_MODEL`           | `claude-3-5-sonnet-20241022`     | Never hardcoded in provider logic — always read from Settings. |
| `OPENAI_MODEL`              | `gpt-4o-mini`                    | Same.                                                    |
| `ALLOW_EXTERNAL_LLM_CALLS`  | `false`                          | Hard safety switch (see above).                          |
| `LLM_TIMEOUT_SECONDS`       | `30.0`                           | Per-provider request timeout.                            |
| `LLM_MAX_OUTPUT_TOKENS`     | `1024`                           | Per-provider max output tokens.                           |
| `LLM_MAX_RETRIES`           | `2`                              | Per-provider SDK-level retry count.                        |
| `MAX_RAG_CONTEXT_CHARACTERS`| `6000`                           | Total evidence character budget per request.               |

### `POST /api/v1/answers`

**Request body:**

```json
{
  "question": "What is the overdraft fee?",
  "document_id": "3f1a9c...e2",
  "providers": ["anthropic", "openai"],
  "top_k": 5
}
```

- `question` — required, non-empty after trimming (max 2000 characters).
- `document_id` — optional; scopes retrieval to one document (`404` if
  it has no indexed chunks).
- `providers` — optional list of `"anthropic"` / `"openai"`; defaults to
  both. An unknown provider name or an empty list is rejected (`422`).
- `top_k` — optional, `1`-`50`; defaults to `RETRIEVAL_TOP_K`.

**Success response — `200 OK`:**

```json
{
  "question": "What is the overdraft fee?",
  "query_was_masked": false,
  "evidence_count": 3,
  "model_results": [
    {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "status": "success",
      "answer": "The overdraft fee is $35 per occurrence, capped at 3 per day [S1].",
      "citations": [
        {
          "source_label": "S1",
          "chunk_id": "b7e2...",
          "document_id": "3f1a9c...e2",
          "source_filename": "bank_policy.pdf",
          "page_number": 4,
          "excerpt": "Overdraft fees are $35 per occurrence, capped at...",
          "relevance_score": 0.87
        }
      ],
      "latency_ms": 842.3,
      "input_tokens": 512,
      "output_tokens": 41,
      "error": null
    },
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "status": "error",
      "answer": "",
      "citations": [],
      "latency_ms": 30000.0,
      "input_tokens": null,
      "output_tokens": null,
      "error": "The OpenAI API request timed out."
    }
  ]
}
```

- `question` is masked **before** retrieval and before it's placed in any
  provider prompt (see [PII Detection & Masking](#11-pii-detection--masking)).
  If PII was detected, `question` here is the masked version and
  `query_was_masked` is `true` — the original, unmasked question is never
  returned, logged, embedded, or sent to Anthropic/OpenAI.

**Error responses:**

| Status | Meaning                                              |
|--------|-------------------------------------------------------|
| `404`  | `document_id` was given but has no indexed chunks       |
| `422`  | `question` is empty/whitespace-only, `top_k` is out of range, or `providers` contains an unknown name / is an empty list |
| `500`  | An embedding/collection configuration mismatch, PII configuration error, or vector-store privacy-version mismatch was detected |
| `503`  | The vector store is unavailable or returned a malformed result |

Individual provider failures (auth, rate limit, timeout, malformed
output, etc.) are **not** HTTP errors — they surface as
`status: "error"` within that provider's own `model_results[]` entry, so
a client always gets `200 OK` with whatever succeeded.

## 11. PII Detection & Masking

Before any extracted text is previewed, chunked, embedded, indexed,
searched, or sent to an external LLM provider, it passes through a
**local, regex-based** PII detection and masking layer. Nothing here
calls a cloud PII/DLP service or any external API.

> ⚠️ **This is a portfolio-quality, best-effort layer — not a compliance
> product.** It does not detect personal names, free-form postal
> addresses, or most non-US identifiers, and it makes no HIPAA, PCI-DSS,
> GDPR, GLBA, or other regulatory compliance claim. **Only upload
> synthetic or public sample documents to this project** — never real
> customer or other sensitive data. See "Limitations" below.

### Masking order (where raw text can exist, and for how long)

1. **Extraction** — `pdf_processor.process_pdf` extracts raw text from
   the PDF, in memory only. The original PDF bytes are never written to
   disk (unchanged since Phase 2).
2. **Masking** — immediately after extraction, `mask_extracted_document`
   detects and masks PII **page by page**, replacing `ExtractedDocument`'s
   pages, `character_count`, and `preview` in place. This happens
   *before* anything else touches the extracted text.
3. **Chunking, embedding, indexing** — `text_chunker` and `VectorStore`
   only ever see the already-masked pages; chunk text, embeddings, and
   what's persisted to Chroma are all derived from masked text.
4. **Search / RAG queries** — a submitted `query`/`question` is masked
   *before* it is embedded or used for retrieval, and before it is placed
   in any RAG prompt — so external providers receive only the masked
   question and already-masked retrieved evidence.
5. **Responses** — upload previews, search results, and RAG answers only
   ever contain masked text. Nothing downstream of step 2 (for documents)
   or the query-masking step (for search/RAG) ever has access to the
   original unmasked value again.

### Supported categories

| Category | Placeholder | Detection rule |
|----------|-------------|-----------------|
| US SSN | `[SSN_REDACTED]` | Formatted `XXX-XX-XXXX` always; unformatted 9 digits only near "SSN"/"social security" context. Rejects area `000`/`666`/`900-999`, group `00`, and serial `0000` as structurally invalid. |
| Credit/debit card | `[CARD_REDACTED]` | 13-19 digit candidate (formatted in 4-4-4-(1-7) groups, or unformatted) **and** Luhn-valid. |
| Email address | `[EMAIL_REDACTED]` | Standard email pattern. Togglable via `PII_MASK_EMAILS`. |
| US phone number | `[PHONE_REDACTED]` | Common formatted patterns, or an unformatted 10-digit NANP-shaped number. Togglable via `PII_MASK_PHONES`. |
| IPv4 address | `[IP_REDACTED]` | Octet-bounded (0-255) dotted-quad pattern. |
| Date of birth | `[DOB_REDACTED]` | A date pattern found near "Date of Birth"/"DOB"/"Born" context **only** — an unlabeled date is never treated as a DOB — **and** the date must be calendar-valid (e.g. "02/30/2020" is rejected). |
| Bank account number | `[ACCOUNT_REDACTED]` | 6-17 digits near "Account Number"/"Acct No"/"A/C" context only. Part of `PII_MASK_FINANCIAL_IDENTIFIERS`. |
| US routing number | `[ROUTING_REDACTED]` | 9 digits near "Routing Number"/"ABA"/"RTN" context **and** ABA checksum-valid. Part of `PII_MASK_FINANCIAL_IDENTIFIERS`. |

SSN and DOB mask whenever `PII_PROTECTION_ENABLED=true`, regardless of
the three category toggles above (which only gate email/phone/financial
identifiers).

**Filenames are masked too.** The sanitized filename is run through the
same detector and masking as page text (e.g. `Customer 555-123-4567
statement.pdf` becomes `Customer [PHONE_REDACTED] statement.pdf`) before
it is used anywhere — the `filename`/`source_filename` fields returned by
every endpoint, stored in Chroma metadata, and included in RAG citation
headers sent to external providers all carry the masked filename, never
the original. Note that email syntax (`@`) cannot itself survive
filename sanitization (see the allow-list in `sanitize_filename`), so
filename-based PII in practice is limited to digit/dash-shaped
identifiers (SSN, phone, card, account, routing numbers).

### Deliberately *not* masked (false-positive protection)

Dollar amounts (`$1,000.00`), interest rates/percentages (`12.5%`),
policy numbers without sensitive context, page numbers, ZIP/ZIP+4 codes,
and general financial statistics never match any detector pattern —
either because they don't fit a category's structural shape at all, or
because a required context keyword (DOB, account, routing) is absent.
Detection is deterministic and, when two candidate entities overlap
(e.g. a phone-shaped run inside a longer digit sequence), resolved by a
fixed category-priority order — the same input always masks the same way.

### Configuration (`Settings` / `.env`)

| Variable                          | Default | Meaning                                              |
|--------------------------------------|---------|---------------------------------------------------------|
| `PII_PROTECTION_ENABLED`            | `true`  | Master switch. When `false`, no detection/masking runs anywhere (text flows through as-is). |
| `PII_MODE`                          | `mask`  | Only `mask` is implemented; any other value is a config error (`500`) at startup. |
| `PII_REDACTION_VERSION`             | `v1`    | Bumped whenever detection/masking rules change materially — see the vector-store check below. |
| `PII_MASK_EMAILS`                   | `true`  | Toggles the EMAIL category. |
| `PII_MASK_PHONES`                   | `true`  | Toggles the PHONE category. |
| `PII_MASK_FINANCIAL_IDENTIFIERS`    | `true`  | Toggles CARD/BANK_ACCOUNT/ROUTING_NUMBER. |

Setting `PII_PROTECTION_ENABLED=false` does **not**, by itself, allow raw
questions or evidence to reach Anthropic/OpenAI: external calls are
independently gated by `ALLOW_EXTERNAL_LLM_CALLS` (default `false`, see
[Multi-Model RAG Answers](#10-multi-model-rag-answers)). Both flags
default to the safe state, so an operator must explicitly opt into
*both* disabling PII masking *and* enabling external calls before any
unmasked content can be sent to a third-party provider — this is a
deliberate, documented combination, never a silent side effect of either
flag alone.

### Response privacy metadata

Only category names and counts are ever returned — never matched values
or their positions in the text:

- Upload responses: `pii_detected` (bool), `pii_entity_count` (int),
  `pii_categories` (sorted, deduped list of category names).
- Search/answer responses: `query_was_masked` (bool) — the `query`/
  `question` field itself is the masked text when this is `true`.

### Vector-store privacy-version enforcement

Every Chroma collection records the active `PII_REDACTION_VERSION` as
metadata at first creation (alongside the existing embedding-provider
tracking from [Vector Search & Retrieval](#9-vector-search--retrieval)).
On every `VectorStore` startup, **whenever `PII_PROTECTION_ENABLED=true`**,
the stored version is compared against the current one:

- **Match** → proceeds normally.
- **Mismatch, or no version recorded at all** → refused with a typed
  `PrivacyVersionMismatchError` (`500`). A *missing* version is refused
  here — unlike the more lenient embedding-provider check — because it
  could mean the collection holds entirely unmasked chunks; silently
  accepting it would risk mixing raw and masked chunks with no way to
  tell them apart later.

This check is skipped entirely when `PII_PROTECTION_ENABLED=false` (the
operator has explicitly accepted the risk).

**Migration:** if you change `PII_REDACTION_VERSION`, change the masking
rules, or need to move from an unprotected to a protected collection,
delete the local vector-store directory (`CHROMA_PERSIST_DIRECTORY`,
`data/vector_store/` by default — it is git-ignored, so this only
affects your local index) and re-upload your documents so they're
indexed under the current version. There is no in-place migration; that
is the deliberately safe, documented path.

### Limitations

- **Regex-only.** This does not use a machine-learning NER model. It
  will not reliably catch personal names, free-form postal addresses, or
  most identifiers outside the specific US-centric patterns listed above.
- **Not exhaustive for international formats.** Non-US SSN-equivalents,
  IBANs, non-US phone numbers, and non-US card/account conventions are
  out of scope for this phase.
- **Detection does not cross page boundaries.** Each page's text (and the
  filename) is masked independently. A sensitive value split across two
  pages — e.g. an SSN with its first digits at the very end of one page
  and the rest at the start of the next — will not be detected as a
  single entity.
- **No compliance claim.** This layer does not make this project HIPAA,
  PCI-DSS, GDPR, GLBA, or otherwise regulatorily compliant, and must not
  be relied upon as a sole safeguard for real sensitive data.
- **Synthetic/public data only.** Use sample or synthetic banking policy
  documents with this project — never real customer data — until (if
  ever) a more rigorous PII/DLP solution is integrated.

## 12. Testing

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

Phase 5 adds: `AnthropicProvider`/`OpenAIProvider` error mapping (missing
API key, authentication failure, rate limit, timeout, connection error,
generic API error, unexpected exception, empty response) with the real
SDK client's network-making method mocked via `AsyncMock` — never a live
call; both-providers-succeed, one-succeeds-one-fails, both-fail, and a
provider-that-raises-doesn't-take-down-the-other; concurrent (not
sequential) execution timing; malformed/unparsable JSON; empty answer;
unknown and duplicate citation-label filtering; a proof that uncited
evidence metadata never leaks into the citations list; model-reported
insufficient evidence; zero-evidence short-circuiting without calling any
provider; a structural prompt-injection check (evidence text is delimited
and verbatim, and the system prompt contains the untrusted-data/ignore-
instructions guidance); evidence context-size capping (including the
always-keep-at-least-one-result edge case); document-scoped retrieval;
`ALLOW_EXTERNAL_LLM_CALLS=false` blocking every provider by default with
no network attempt; the answer response's exact key set; and a check that
no question/answer/evidence text ever appears in an error message. All
Phase 5 tests use `FakeLLMProvider` or a mocked SDK client — no live API
calls, no real API keys, no model downloads.

Phase 6 adds: every supported PII category (formatted and unformatted
SSNs/cards/phones, valid and invalid Luhn cards, valid and invalid ABA
routing checksums, context-dependent DOB/account/routing detection,
rejection of structurally-invalid SSNs and calendar-impossible DOB
dates); false-positive avoidance for dollar amounts, interest rates,
policy numbers, page numbers, ZIP codes, and general financial
statistics; deterministic overlap resolution across every category pair
that can collide on the same span, duplicate values, adjacent
no-separator values, and unicode/multiline surrounding text; that an
already-masked placeholder is never re-detected and re-masking is a
no-op; masking of PII in the uploaded filename (propagated to every
chunk's `source_filename`, Chroma metadata, search results, and RAG
citations); masking before preview generation, chunking, indexing, and
persistence; masked search queries and RAG questions, including a
`_CapturingProvider` proof that external providers receive only the
masked question/evidence/filename; that no original PII value ever
appears in an API response, a Chroma-stored document, a provider prompt,
a log line, or an exception message; `PII_PROTECTION_ENABLED=false`
behavior (and the explicit two-flag interaction with
`ALLOW_EXTERNAL_LLM_CALLS`); `PII_REDACTION_VERSION` mismatch/missing-
version rejection in the vector store (including a check that a failed
reopen attempt never reads or overwrites existing data) with its
documented migration path; and the upload/search/answer responses' exact
key sets including the new `pii_detected`/`pii_entity_count`/
`pii_categories`/`query_was_masked` fields. All Phase 6 tests use the
real `LocalRegexPIIDetector` (pure regex, no network/model dependency)
or `FakePIIDetector` — no cloud PII service is ever called.

## 13. Ethical & Privacy Considerations

- **No real API keys or secrets are ever committed.** `.env` is
  git-ignored; only `.env.example` (placeholder variable names) is
  tracked.
- **Sensitive data masking is a first-class feature, not an add-on.**
  Account numbers, SSNs, emails, and similar identifiers are detected
  and redacted locally — see
  [PII Detection & Masking](#11-pii-detection--masking) — before content
  is chunked, indexed, displayed, logged, or sent to any third-party LLM
  provider. This is a best-effort regex layer, not a compliance
  guarantee (no HIPAA/PCI-DSS/GDPR/GLBA claim is made).
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
- **Real external LLM calls are off by default and must be opted into
  explicitly.** `ALLOW_EXTERNAL_LLM_CALLS=false` refuses every
  Anthropic/OpenAI request with a safe configuration error and no network
  attempt (see [Multi-Model RAG Answers](#10-multi-model-rag-answers)).
  Enabling it sends the masked question and masked retrieved evidence
  excerpts (not the full document) to whichever third-party provider(s)
  are requested.
  **Do not upload real customer or other sensitive documents.** The
  local PII masking layer (see
  [PII Detection & Masking](#11-pii-detection--masking)) is a best-effort
  regex-based safeguard, not a regulatory-compliance guarantee — use
  synthetic or public sample policy documents only.

## 14. License

See [LICENSE](./LICENSE).
