"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider credentials (not used until later phases; never given real
    # defaults here so a missing .env never accidentally enables live calls).
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    openai_model: Optional[str] = None

    app_env: str = "development"
    log_level: str = "INFO"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    streamlit_server_port: int = 8501

    chroma_db_path: str = "./data/vector_store"
    chroma_collection_name: str = "policylens_documents"

    max_upload_size_mb: int = 10
    allowed_file_types: str = ".pdf"

    enable_pii_masking: bool = True
    enable_cost_tracking: bool = True
    enable_latency_tracking: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
