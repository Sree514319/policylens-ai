"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider credentials -- never given real defaults here so a
    # missing .env never accidentally enables live calls. The real safety
    # gate is allow_external_llm_calls below, but a missing key is also
    # independently checked before any request is attempted.
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openai_model: str = "gpt-4o-mini"

    # Hard safety switch: real Anthropic/OpenAI network calls are refused
    # (with a safe configuration error, no network attempt) unless this is
    # explicitly set to true. Off by default so cloning this repo and
    # running it never silently sends data to a third party.
    allow_external_llm_calls: bool = False
    # Bounded so a misconfigured .env can't produce a degenerate value
    # (a zero/negative timeout, an unbounded retry count, a context budget
    # so small no evidence fits, or so large it defeats the point of a cap).
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    max_rag_context_characters: int = Field(default=6000, ge=100, le=100_000)

    app_env: str = "development"
    log_level: str = "INFO"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    streamlit_server_port: int = 8501

    chroma_persist_directory: str = "./data/vector_store"
    chroma_collection_name: str = "policylens_documents"

    max_upload_size_mb: int = 10
    allowed_file_types: str = ".pdf"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_length: int = 50

    embedding_model_name: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    min_relevance_score: float = 0.0

    # Local, regex-based PII detection/masking (see app/services/privacy).
    # Runs entirely on-device -- no cloud PII service, no external API call.
    # This is a best-effort layer for common US financial identifiers, not
    # a substitute for a professional PII/DLP tool or a compliance control;
    # see the README's "Limitations" section before relying on it.
    pii_protection_enabled: bool = True
    # "mask" is the only mode implemented; a Literal makes an invalid value
    # a hard config error (caught by Settings validation) rather than a
    # silently-ignored typo.
    pii_mode: Literal["mask"] = "mask"
    # Bumped whenever detection/masking rules change materially. Chunks
    # already indexed under a different (or missing) version are refused
    # rather than silently mixed with chunks masked under the current
    # rules -- see VectorStore's privacy-version check.
    pii_redaction_version: str = "v1"
    pii_mask_emails: bool = True
    pii_mask_phones: bool = True
    pii_mask_financial_identifiers: bool = True

    enable_cost_tracking: bool = True
    enable_latency_tracking: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
