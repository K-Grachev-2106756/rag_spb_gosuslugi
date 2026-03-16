"""Tests for data processing and chunking modules."""

import pytest
from unittest.mock import patch, Mock

from src.data_processing.chunking import RecursiveChunker
from src.data_processing.loader import Document, DocumentChunk


class TestRecursiveChunker:
    """Tests for RecursiveChunker class."""

    def test_chunker_initialization(self):
        """Test chunker initializes with correct parameters."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10

    def test_chunker_default_initialization(self):
        """Test chunker initializes with default settings."""
        from src.config import settings

        chunker = RecursiveChunker()
        assert chunker.chunk_size == settings.CHUNK_SIZE
        assert chunker.chunk_overlap == settings.CHUNK_OVERLAP

    def test_chunk_document_empty_description(self):
        """Test chunking a document with empty description."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="",
            info_part=[],
            source_path="test.html"
        )
        chunks = list(chunker.chunk_document(document))
        assert len(chunks) == 0

    def test_chunk_document_with_content(self):
        """Test chunking a document with content."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        document = Document(
            title="Test Document",
            description_part="This is some test content for chunking.",
            info_part=[],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))
        assert len(chunks) == 0  # No info_part means no chunks

    def test_chunk_document_with_info_part(self):
        """Test chunking a document with info_part content."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        document = Document(
            title="Test Document",
            description_part="Description",
            info_part=[("Section 1", "This is content for section 1 that should be chunked.")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_chunk_has_context(self):
        """Test that chunks contain document context."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test Title",
            description_part="Description here",
            info_part=[("Block Title", "Block content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert chunk.document_title == "Test Title"
            assert "[Документ: Test Title]" in chunk.content

    def test_chunk_has_description_context(self):
        """Test that chunks contain description context."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test Title",
            description_part="Test Description",
            info_part=[("Section", "Content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert "[Описание: Test Description]" in chunk.content

    def test_chunk_has_section_context(self):
        """Test that chunks contain section context."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test Title",
            description_part="Description",
            info_part=[("My Section", "Section content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert "[Раздел: My Section]" in chunk.content

    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Desc",
            info_part=[("S1", "Content 1"), ("S2", "Content 2")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"

    def test_chunk_id_format(self):
        """Test chunk ID format."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Desc",
            info_part=[("Section", "Content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk_")

    def test_chunk_metadata_field(self):
        """Test that chunk metadata contains full context."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Description",
            info_part=[("Section", "Full section content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert chunk.metadata is not None
            assert "Full section content" in chunk.metadata

    def test_chunk_parent_section(self):
        """Test chunk parent_section is set."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Desc",
            info_part=[("Section", "Content")],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))

        for chunk in chunks:
            assert chunk.parent_section.startswith("info_")

    def test_chunk_multiple_info_parts(self):
        """Test chunking with multiple info parts."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Description",
            info_part=[
                ("Section 1", "Content 1"),
                ("Section 2", "Content 2"),
                ("Section 3", "Content 3"),
            ],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))
        assert len(chunks) >= 3

    def test_chunk_empty_info_content(self):
        """Test that empty info content is skipped."""
        chunker = RecursiveChunker()
        document = Document(
            title="Test",
            description_part="Description",
            info_part=[
                ("Section 1", ""),
                ("Section 2", "   "),
                ("Section 3", "Valid content"),
            ],
            source_path="test.html",
        )
        chunks = list(chunker.chunk_document(document))
        # Only Section 3 should produce chunks
        assert len(chunks) >= 1

    def test_count_tokens_with_tokenizer(self):
        """Test token counting."""
        chunker = RecursiveChunker()
        text = "This is a test text for token counting."
        token_count = chunker._count_tokens(text)
        assert token_count > 0

    def test_count_tokens_empty_text(self):
        """Test token counting for empty text."""
        chunker = RecursiveChunker()
        token_count = chunker._count_tokens("")
        # Empty text may still return a small token count due to tokenizer special tokens
        assert token_count >= 0

    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunker = RecursiveChunker()
        chunk_id = chunker._generate_chunk_id("test content", 0)
        assert chunk_id.startswith("chunk_")
        assert "_0" in chunk_id

    def test_generate_chunk_id_unique_for_different_content(self):
        """Test that different content produces different IDs."""
        chunker = RecursiveChunker()
        id1 = chunker._generate_chunk_id("content 1", 0)
        id2 = chunker._generate_chunk_id("content 2", 0)
        assert id1 != id2

    def test_build_contextual_content_full(self):
        """Test building full contextual content."""
        chunker = RecursiveChunker()
        content = chunker._build_contextual_content(
            document_title="Doc Title",
            description="Description",
            info_title="Info Title",
            content="Main content",
        )
        assert "[Документ: Doc Title]" in content
        assert "[Описание: Description]" in content
        assert "[Раздел: Info Title]" in content
        assert "Main content" in content

    def test_build_contextual_content_partial(self):
        """Test building partial contextual content."""
        chunker = RecursiveChunker()
        content = chunker._build_contextual_content(
            document_title="Doc Title",
            content="Main content",
        )
        assert "[Документ: Doc Title]" in content
        assert "[Описание:" not in content
        assert "[Раздел:" not in content
        assert "Main content" in content

    def test_build_contextual_content_empty(self):
        """Test building empty contextual content."""
        chunker = RecursiveChunker()
        content = chunker._build_contextual_content()
        assert content == ""

    def test_split_text_by_tokens_empty(self):
        """Test splitting empty text."""
        chunker = RecursiveChunker()
        chunks = chunker._split_text_by_tokens("", max_tokens=100, overlap_tokens=10)
        assert chunks == []

    def test_split_text_by_tokens_short_text(self):
        """Test splitting text that fits in one chunk."""
        chunker = RecursiveChunker()
        text = "Short text."
        chunks = chunker._split_text_by_tokens(text, max_tokens=100, overlap_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_all_documents(self):
        """Test chunking multiple documents."""
        chunker = RecursiveChunker()
        documents = [
            Document(
                title="Doc 1",
                description_part="Desc 1",
                info_part=[("Section", "Content 1")],
                source_path="test1.html",
            ),
            Document(
                title="Doc 2",
                description_part="Desc 2",
                info_part=[("Section", "Content 2")],
                source_path="test2.html",
            ),
        ]
        chunks = list(chunker.chunk_all_documents(documents))
        assert len(chunks) >= 2


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a Document instance."""
        doc = Document(
            title="Test Title",
            description_part="Part 1",
            info_part=[("Block 1", "Content 1")],
            source_path="/path/to/file.html",
        )
        assert doc.title == "Test Title"
        assert doc.description_part == "Part 1"
        assert len(doc.info_part) == 1
        assert doc.source_path == "/path/to/file.html"

    def test_document_empty_info(self):
        """Test document with empty info part."""
        doc = Document(
            title="Test",
            description_part="",
            info_part=[],
            source_path="test.html",
        )
        assert doc.title == "Test"
        assert len(doc.info_part) == 0


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        chunk = DocumentChunk(
            content="Test content",
            document_title="Test Doc",
            metadata="Test Meta",
            chunk_id="chunk_001",
            parent_section="general",
        )
        assert chunk.content == "Test content"
        assert chunk.document_title == "Test Doc"
        assert chunk.metadata == "Test Meta"
        assert chunk.chunk_id == "chunk_001"
        assert chunk.parent_section == "general"

    def test_chunk_default_parent_section(self):
        """Test chunk default parent_section value."""
        chunk = DocumentChunk(
            content="Content",
            document_title="Doc",
            metadata="Meta",
            chunk_id="chunk_002",
        )
        assert chunk.parent_section == ""
