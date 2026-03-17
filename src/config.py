"""Application configuration using pydantic-settings."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM settings (Mistral API)
    MISTRAL_API_KEY: str
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    LLM_MODEL: str = "mistral-small-latest"
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 4096

    # SGR RAG Settings
    SGR_MAX_ITERATIONS: int = 3
    SGR_MAX_ADDITIONAL_QUESTIONS: int = 5

    # Embeddings Settings (local model)
    EMBEDDING_MODEL: str = "ai-sage/Giga-Embeddings-instruct"
    EMBEDDING_DIMENSION: int = 2048
    HF_TOKEN: Optional[str] = None

    # RAG Settings
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 64
    TOP_K_RESULTS: int = 3
    TOP_K_ADDITIONAL: int = 2
    DOC_MIN_SCORE: float = 0.5

    # Vector DB Settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Data Settings
    DATA_DIR: str = "./data"

    # Server Settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
