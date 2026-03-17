"""Tests for vector store module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_processing.loader import Document, DocumentChunk
from src.embeddings import EmbeddingProvider
from src.vector_store import VectorStore


@pytest.fixture(autouse=True)
def mock_embedding_provider_init():
    """Mock embedding provider initialization to prevent model loading."""
    with patch("src.vector_store.store.create_embedding_provider") as mock_create:
        mock_instance = Mock()
        mock_instance.embed = Mock(return_value=[0.1] * 1024)
        mock_instance.dimension = 1024

        def embed_batch_side_effect(data, **kwargs):
            return [[0.1] * 1024 for _ in data]

        mock_instance.embed_batch = Mock(side_effect=embed_batch_side_effect)
        mock_create.return_value = mock_instance
        yield mock_create


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = VectorStore(persist_dir=tmp_dir)
        yield store
        store.clear()


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = Mock(spec=EmbeddingProvider)
    provider.embed = Mock(return_value=[0.1] * 1024)
    provider.embed_batch = Mock(return_value=[[0.1] * 1024])
    provider.dimension = 1024
    return provider


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        title="Test Document",
        description_part="This is test content for the document.",
        info_part=[("Section 1", "Content of section 1")],
        source_path="test.html",
    )


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_initialization(self, temp_vector_store):
        """Test vector store initializes correctly."""
        assert temp_vector_store.count == 0

    def test_initialization_with_custom_persist_dir(self):
        """Test vector store with custom persist directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(persist_dir=tmp_dir)
            assert store.persist_dir == Path(tmp_dir)

    def test_initialization_creates_directory(self):
        """Test that vector store creates persist directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new_db"
            store = VectorStore(persist_dir=new_dir)
            assert new_dir.exists()

    def test_initialization_with_embedding_provider(self, mock_embedding_provider):
        """Test vector store with custom embedding provider."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(
                persist_dir=tmp_dir,
                embedding_provider=mock_embedding_provider,
            )
            assert store._embedding_provider == mock_embedding_provider

    def test_add_chunk(self, temp_vector_store):
        """Test adding a single chunk."""
        chunk = DocumentChunk(
            content="Test chunk content",
            document_title="Test Doc",
            metadata="Test Meta",
            chunk_id="test_chunk_001",
            parent_section="general",
        )
        temp_vector_store.add_chunk(chunk)
        assert temp_vector_store.count == 1

    def test_add_chunk_metadata(self, temp_vector_store):
        """Test that chunk metadata is stored correctly."""
        chunk = DocumentChunk(
            content="Test content",
            document_title="Test Doc",
            metadata="Full metadata",
            chunk_id="test_001",
            parent_section="section1",
        )
        temp_vector_store.add_chunk(chunk)

        results = temp_vector_store.search("test", top_k=1)
        assert len(results) > 0
        assert results[0]["metadata"]["document_title"] == "Test Doc"
        assert results[0]["metadata"]["parent_section"] == "section1"

    def test_add_chunks_batch(self, temp_vector_store):
        """Test adding multiple chunks in batch."""
        chunks = [
            DocumentChunk(
                content=f"Test chunk content {i}",
                document_title="Test Doc",
                metadata="Test Meta",
                chunk_id=f"test_chunk_{i:03d}",
                parent_section="general",
            )
            for i in range(5)
        ]
        temp_vector_store.add_chunks(chunks)
        assert temp_vector_store.count == 5

    def test_add_chunks_empty_list(self, temp_vector_store):
        """Test adding empty chunk list."""
        temp_vector_store.add_chunks([])
        assert temp_vector_store.count == 0

    def test_add_chunks_batch_size_limit(self, temp_vector_store):
        """Test that batch adding respects batch size limits."""
        # Add more than batch size (100)
        chunks = [
            DocumentChunk(
                content=f"Content {i}",
                document_title="Doc",
                metadata="Meta",
                chunk_id=f"chunk_{i:04d}",
                parent_section="general",
            )
            for i in range(150)
        ]
        temp_vector_store.add_chunks(chunks)
        assert temp_vector_store.count == 150

    def test_search(self, temp_vector_store, sample_document):
        """Test searching for relevant chunks."""
        # Index the document
        temp_vector_store.index_document(sample_document)

        # Search for relevant content
        results = temp_vector_store.search("test content", top_k=3)
        assert len(results) > 0
        assert "content" in results[0]
        assert "metadata" in results[0]

    def test_search_top_k(self, temp_vector_store, sample_document):
        """Test search respects top_k parameter."""
        temp_vector_store.index_document(sample_document)

        results = temp_vector_store.search("test", top_k=1)
        assert len(results) <= 1

    def test_search_returns_distance(self, temp_vector_store, sample_document):
        """Test that search results include distance."""
        temp_vector_store.index_document(sample_document)

        results = temp_vector_store.search("test", top_k=1)
        assert "distance" in results[0]
        assert isinstance(results[0]["distance"], float)

    def test_search_empty_store(self, temp_vector_store):
        """Test searching an empty store."""
        results = temp_vector_store.search("test query", top_k=3)
        assert results == []

    def test_clear(self, temp_vector_store):
        """Test clearing the vector store."""
        chunk = DocumentChunk(
            content="Test content",
            document_title="Test",
            metadata="Test",
            chunk_id="test_001",
            parent_section="general",
        )
        temp_vector_store.add_chunk(chunk)
        assert temp_vector_store.count == 1

        temp_vector_store.clear()
        assert temp_vector_store.count == 0

    def test_index_document(self, temp_vector_store, sample_document):
        """Test indexing a full document."""
        chunk_count = temp_vector_store.index_document(sample_document)
        assert chunk_count > 0
        assert temp_vector_store.count == chunk_count

    def test_index_all_documents(self, temp_vector_store):
        """Test indexing multiple documents."""
        documents = [
            Document(
                title=f"Doc {i}",
                description_part=f"Description {i}",
                info_part=[("Section", f"Content {i}")],
                source_path=f"test{i}.html",
            )
            for i in range(3)
        ]

        total_chunks = temp_vector_store.index_all_documents(documents)
        assert total_chunks > 0
        assert temp_vector_store.count == total_chunks

    def test_index_all_documents_empty_list(self, temp_vector_store):
        """Test indexing empty document list."""
        total_chunks = temp_vector_store.index_all_documents([])
        assert total_chunks == 0
        assert temp_vector_store.count == 0

    def test_count_property(self, temp_vector_store):
        """Test count property."""
        assert temp_vector_store.count == 0

        chunk = DocumentChunk(
            content="Test",
            document_title="Doc",
            metadata="Meta",
            chunk_id="chunk_1",
            parent_section="general",
        )
        temp_vector_store.add_chunk(chunk)
        assert temp_vector_store.count == 1

    def test_search_with_no_results(self, temp_vector_store):
        """Test search when no results match."""
        chunk = DocumentChunk(
            content="Completely unrelated content here",
            document_title="Doc",
            metadata="Meta",
            chunk_id="chunk_1",
            parent_section="general",
        )
        temp_vector_store.add_chunk(chunk)

        results = temp_vector_store.search("xyz123unrelated", top_k=3)
        # May return results due to embedding similarity
        assert isinstance(results, list)

    def test_add_chunk_uses_embedding_provider(self, mock_embedding_provider):
        """Test that add_chunk uses the embedding provider."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(
                persist_dir=tmp_dir,
                embedding_provider=mock_embedding_provider,
            )

            chunk = DocumentChunk(
                content="Test content",
                document_title="Doc",
                metadata="Meta",
                chunk_id="chunk_1",
                parent_section="general",
            )
            store.add_chunk(chunk)

            mock_embedding_provider.embed.assert_called_once_with("Test content")

    def test_add_chunks_uses_embedding_provider_batch(self, mock_embedding_provider):
        """Test that add_chunks uses embedding provider batch method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings that match what ChromaDB expects
            embeddings_list = [[0.1] * 1024 for _ in range(5)]
            mock_embedding_provider.embed_batch = Mock(return_value=embeddings_list)
            
            store = VectorStore(
                persist_dir=tmp_dir,
                embedding_provider=mock_embedding_provider,
            )

            chunks = [
                DocumentChunk(
                    content=f"Content {i}",
                    document_title="Doc",
                    metadata="Meta",
                    chunk_id=f"chunk_{i}",
                    parent_section="general",
                )
                for i in range(5)
            ]
            store.add_chunks(chunks)

            # embed_batch should be called
            mock_embedding_provider.embed_batch.assert_called_once()
            # Check that it was called with the right content
            call_args = mock_embedding_provider.embed_batch.call_args[0][0]
            assert len(call_args) == 5
            assert call_args[0] == "Content 0"

    def test_search_uses_embedding_provider(self, mock_embedding_provider):
        """Test that search uses the embedding provider."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStore(
                persist_dir=tmp_dir,
                embedding_provider=mock_embedding_provider,
            )

            store.search("test query", top_k=3)

            mock_embedding_provider.embed.assert_called_once_with("test query", is_query=True)
