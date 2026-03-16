"""Retrieval module for finding relevant document chunks."""

import logging
from dataclasses import dataclass

from src.config import settings
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""

    content: str
    document_title: str
    metadata: str
    parent_section: str
    relevance_score: float

    def format_for_context(self) -> str:
        """Format the result for inclusion in LLM context."""
        return (
            f"[Источник: {self.document_title}]\n" +
            self.metadata
        )


class Retriever:
    """
    Retriever module for finding relevant document chunks.
    
    Implements semantic search using vector embeddings.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = settings.TOP_K_RESULTS,
        min_score: int = settings.DOC_MIN_SCORE,
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance for searching.
            top_k: Number of results to retrieve.
        """
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_score = min_score
        logger.info(f"Retriever initialized with {top_k=} and {min_score=}")

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text.
            
        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        logger.debug(f"Retrieving for query: {query}")

        raw_results = self._vector_store.search(query, top_k=self._top_k)

        results = []
        for raw in raw_results:
            metadata, distance = raw.get("metadata", {}), raw.get("distance", 1.)
            
            relevance_score = 1. - min(distance, 1.)
            if (self._min_score is None) or (relevance_score >= self._min_score):
                results.append(
                    RetrievalResult(
                        content=raw["content"],
                        document_title=metadata.get("document_title", "Unknown"),
                        metadata=metadata.get("metadata", ""),
                        parent_section=metadata.get("parent_section", ""),
                        relevance_score=relevance_score,
                    )
                )

        logger.info(f"Retrieved {len(results)} results")
        return results

    def retrieve_formatted(self, query: str) -> list[str]:
        """
        Retrieve and format results for LLM context.
        
        Args:
            query: User query text.
            
        Returns:
            Formatted list of strings with all relevant context.
        """
        results = self.retrieve(query)

        if not results:
            return []

        formatted_parts, used_documents = list(), set()
        for result in results:
            if result.metadata in used_documents:
                continue
            
            formatted_parts.append(result.format_for_context())
            used_documents.add(result.metadata)

        return formatted_parts
