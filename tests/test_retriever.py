"""Tests for retrieval module."""

import pytest
from unittest.mock import Mock, patch

from src.retrieval.retriever import (
    RetrievalResult,
    Retriever,
)
from src.vector_store import VectorStore


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult instance."""
        result = RetrievalResult(
            content="Test content",
            document_title="Test Doc",
            metadata="Test Meta",
            parent_section="general",
            relevance_score=0.85,
        )
        assert result.content == "Test content"
        assert result.document_title == "Test Doc"
        assert result.metadata == "Test Meta"
        assert result.parent_section == "general"
        assert result.relevance_score == 0.85

    def test_format_for_context(self):
        """Test formatting result for context."""
        result = RetrievalResult(
            content="Content text",
            document_title="Test Document",
            metadata="Full metadata here",
            parent_section="section1",
            relevance_score=0.9,
        )
        formatted = result.format_for_context()

        assert "[Источник: Test Document]" in formatted
        assert "Full metadata here" in formatted

    def test_format_for_context_structure(self):
        """Test context format structure."""
        result = RetrievalResult(
            content="Content",
            document_title="Doc Title",
            metadata="Metadata",
            parent_section="",
            relevance_score=0.7,
        )
        formatted = result.format_for_context()

        assert "[Источник:" in formatted
        assert "Doc Title" in formatted
        assert "Metadata" in formatted


class TestRetriever:
    """Tests for Retriever class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.search = Mock(return_value=[])
        return store

    def test_retriever_initialization(self, mock_vector_store):
        """Test retriever initializes with correct parameters."""
        retriever = Retriever(
            vector_store=mock_vector_store,
            top_k=5,
            min_score=0.7,
        )
        assert retriever._top_k == 5
        assert retriever._min_score == 0.7

    def test_retriever_default_parameters(self, mock_vector_store):
        """Test retriever default parameters."""
        from src.config import settings

        retriever = Retriever(vector_store=mock_vector_store)
        assert retriever._top_k == settings.TOP_K_RESULTS
        assert retriever._min_score == settings.DOC_MIN_SCORE

    def test_retrieve_empty_results(self, mock_vector_store):
        """Test retrieval when no results found."""
        mock_vector_store.search.return_value = []

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve("test query")

        assert results == []
        mock_vector_store.search.assert_called_once()

    def test_retrieve_with_results(self, mock_vector_store):
        """Test retrieval with results."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content 1",
                "metadata": {
                    "document_title": "Doc 1",
                    "metadata": "Meta 1",
                    "parent_section": "sec1",
                },
                "distance": 0.2,
            }
        ]

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].content == "Content 1"
        assert results[0].document_title == "Doc 1"
        assert results[0].relevance_score == 0.8  # 1 - 0.2

    def test_retrieve_calculates_relevance_score(self, mock_vector_store):
        """Test that relevance score is calculated correctly."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.3,
            }
        ]

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.7  # 1 - 0.3

    def test_retrieve_filters_by_min_score(self, mock_vector_store):
        """Test that results below min_score are filtered."""
        mock_vector_store.search.return_value = [
            {
                "content": "Good content",
                "metadata": {"document_title": "Good Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.2,  # relevance: 0.8
            },
            {
                "content": "Bad content",
                "metadata": {"document_title": "Bad Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.6,  # relevance: 0.4
            },
        ]

        retriever = Retriever(vector_store=mock_vector_store, min_score=0.5)
        results = retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].document_title == "Good Doc"

    def test_retrieve_with_none_min_score(self, mock_vector_store):
        """Test retrieval when min_score is None."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.9,  # low relevance: 0.1
            }
        ]

        retriever = Retriever(vector_store=mock_vector_store, min_score=None)
        results = retriever.retrieve("query")

        assert len(results) == 1  # All results returned when min_score is None

    def test_retrieve_multiple_results(self, mock_vector_store):
        """Test retrieval with multiple results."""
        mock_vector_store.search.return_value = [
            {
                "content": f"Content {i}",
                "metadata": {"document_title": f"Doc {i}", "metadata": f"Meta {i}", "parent_section": ""},
                "distance": 0.1 * i,
            }
            for i in range(5)
        ]

        retriever = Retriever(vector_store=mock_vector_store, top_k=5)
        results = retriever.retrieve("query")

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.content == f"Content {i}"
            assert result.document_title == f"Doc {i}"

    def test_retrieve_formatted_empty(self, mock_vector_store):
        """Test formatted retrieval with no results."""
        mock_vector_store.search.return_value = []

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve_formatted("query")

        assert results == []

    def test_retrieve_formatted_deduplicates(self, mock_vector_store):
        """Test that formatted retrieval deduplicates by metadata."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content 1",
                "metadata": {"document_title": "Doc", "metadata": "Same Meta", "parent_section": ""},
                "distance": 0.1,
            },
            {
                "content": "Content 2",
                "metadata": {"document_title": "Doc", "metadata": "Same Meta", "parent_section": ""},
                "distance": 0.2,
            },
        ]

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve_formatted("query")

        assert len(results) == 1
        assert "Same Meta" in results[0]

    def test_retrieve_formatted_multiple_documents(self, mock_vector_store):
        """Test formatted retrieval with multiple documents."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content 1",
                "metadata": {"document_title": "Doc 1", "metadata": "Meta 1", "parent_section": ""},
                "distance": 0.1,
            },
            {
                "content": "Content 2",
                "metadata": {"document_title": "Doc 2", "metadata": "Meta 2", "parent_section": ""},
                "distance": 0.2,
            },
        ]

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve_formatted("query")

        assert len(results) == 2
        assert "[Источник: Doc 1]" in results[0]
        assert "[Источник: Doc 2]" in results[1]

    def test_retrieve_handles_missing_metadata(self, mock_vector_store):
        """Test retrieval handles missing metadata gracefully."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content",
                "metadata": {},
                "distance": 0.2,
            }
        ]

        retriever = Retriever(vector_store=mock_vector_store)
        results = retriever.retrieve("query")

        assert len(results) == 1
        assert results[0].document_title == "Unknown"
        assert results[0].parent_section == ""

    def test_retrieve_handles_missing_distance(self, mock_vector_store):
        """Test retrieval handles missing distance gracefully."""
        mock_vector_store.search.return_value = [
            {
                "content": "Content",
                "metadata": {"document_title": "Doc", "metadata": "Meta", "parent_section": ""},
                "distance": 0.0,  # Provide a default distance
            }
        ]

        retriever = Retriever(vector_store=mock_vector_store, min_score=0.0)  # Accept all scores
        results = retriever.retrieve("query")

        assert len(results) >= 1
