"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Settings (Mistral API)
    MISTRAL_API_KEY: str
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    LLM_MODEL: str = "mistral-small-latest"

    # Embeddings Settings (local model)
    EMBEDDING_MODEL: str = "deepvk/USER-bge-m3"
    EMBEDDING_DIMENSION: int = 1024

    # RAG Settings
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 64
    TOP_K_RESULTS: int = 3

    # Vector DB Settings
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # Data Settings
    DATA_DIR: str = "./data"

    # Server Settings
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000


settings = Settings()
