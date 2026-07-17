# PolicyLens AI

A multi-model financial document intelligence assistant. Upload banking
policy PDFs, ask questions in natural language, and receive cited,
fact-checked answers — generated and cross-validated by both **Anthropic
Claude** and **OpenAI GPT** models — alongside transparent accuracy,
latency, and cost metrics.

> **Status:** 🚧 Project foundation stage. Architecture and folder
> structure are in place; core functionality is not yet implemented.

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
│   │   ├── api/v1/         # FastAPI routers (versioned)
│   │   ├── core/           # Settings, config, shared infrastructure
│   │   ├── models/         # Internal data models
│   │   ├── schemas/        # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── llm/          # Claude / OpenAI client wrappers, comparison logic
│   │   │   ├── ingestion/    # PDF parsing & chunking
│   │   │   ├── retrieval/    # ChromaDB indexing & semantic search, citations
│   │   │   ├── privacy/      # PII/sensitive-data detection & masking
│   │   │   └── evaluation/   # Unsupported-claim detection, accuracy, latency, cost
│   │   └── utils/           # Shared helpers
│   └── tests/               # Backend unit/integration tests
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
├── requirements.txt           # Python dependencies
└── README.md
```

## 6. Setup Roadmap

This repository is being built incrementally. Planned phases:

- [x] **Phase 0 — Project foundation**: repository scaffolding, folder
      structure, README, dependency baseline (this step).
- [ ] **Phase 1 — Core configuration**: environment/config management,
      settings validation, logging setup.
- [ ] **Phase 2 — Document ingestion**: PDF upload endpoint, PyMuPDF
      parsing, chunking strategy.
- [ ] **Phase 3 — Vector store & retrieval**: ChromaDB integration,
      embedding pipeline, semantic search with citation metadata.
- [ ] **Phase 4 — LLM integration**: Claude and OpenAI client wrappers,
      prompt templates, concurrent dual-model querying.
- [ ] **Phase 5 — Privacy layer**: sensitive-information detection and
      masking applied before display/storage/outbound calls.
- [ ] **Phase 6 — Evaluation layer**: unsupported-claim detection,
      accuracy scoring, latency and cost tracking.
- [ ] **Phase 7 — Frontend**: Streamlit upload flow, chat interface,
      model comparison view, metrics dashboard.
- [ ] **Phase 8 — Testing**: pytest unit/integration coverage across
      services and API routes.
- [ ] **Phase 9 — Containerization & deployment**: Dockerfiles, compose
      setup, deployment documentation.

### Local setup (once implementation begins)

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd policylens-ai

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# then fill in real API keys and settings in .env

# 5. Run the backend and frontend (once implemented)
# uvicorn backend.app.main:app --reload
# streamlit run frontend/streamlit_app/app.py
```

## 7. Ethical & Privacy Considerations

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

## 8. License

See [LICENSE](./LICENSE).
