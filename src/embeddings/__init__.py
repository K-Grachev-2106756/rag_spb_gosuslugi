"""Embeddings module for vector representations."""

from src.embeddings.embedding import (
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    create_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "create_embedding_provider",
]
