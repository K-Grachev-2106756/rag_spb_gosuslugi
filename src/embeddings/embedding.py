"""Embeddings module using sentence-transformers."""

from abc import ABC, abstractmethod
import logging

import torch
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# Prompt for encoding user queries with Giga-Embeddings-instruct model
QUERY_PROMPT = "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: "

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, data: str) -> list[float]:
        """Generate embeddings for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, data: list[str]) -> list[list[float]]:
        """Generate embeddings for all provided texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers with local models.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        logger.info(f"Loading embedding model: {model_name}")

        self._model = SentenceTransformer(
            model_name_or_path=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
            },
            config_kwargs={
                "trust_remote_code": True,
            },
        ).eval()
        self._model.max_seq_length = 4096
        self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded. Dimension: {self._dimension}")


    def embed(self, data: str, is_query: bool = False) -> list[float]:
        """
        Generate embeddings for single text.

        Args:
            data: Text to embed.
            is_query: If True, apply query prompt for user queries.

        Returns:
            Embedding or list of embeddings.
        """
        with torch.no_grad():
            embeddings = self._model.encode(
                data,
                convert_to_numpy=True,
                **({"prompt": QUERY_PROMPT} if is_query else {}),
            )

        return embeddings.tolist()

    def embed_batch(
        self,
        data: list[str],
        batch_size: int = 16,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for a large list of texts in batches.

        Args:
            data: List of texts to embed.
            batch_size: Number of texts to process in each batch.
            show_progress: Whether to show progress bar.
            is_query: If True, apply query prompt for user queries.

        Returns:
            List of embeddings.
        """
        with torch.no_grad():
            embeddings = self._model.encode(
                data,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                **({"prompt": QUERY_PROMPT} if is_query else {}),
            )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


def create_embedding_provider() -> EmbeddingProvider:
    """Factory function to create an embedding provider."""
    return SentenceTransformerEmbeddings(settings.EMBEDDING_MODEL)
