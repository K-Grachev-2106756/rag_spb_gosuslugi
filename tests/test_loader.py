"""Tests for document loader module."""

import pytest
import tempfile
from pathlib import Path

from src.data_processing.loader import DocumentLoader, Document, DocumentChunk


class TestDocumentDataclass:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a Document instance."""
        doc = Document(
            title="Test Title",
            description_part="Test description",
            info_part=[("Section 1", "Content 1"), ("Section 2", "Content 2")],
            source_path="/path/to/file.html",
        )
        assert doc.title == "Test Title"
        assert doc.description_part == "Test description"
        assert len(doc.info_part) == 2
        assert doc.source_path == "/path/to/file.html"

    def test_document_empty_info(self):
        """Test document with empty info part."""
        doc = Document(
            title="Empty Doc",
            description_part="",
            info_part=[],
            source_path="test.html",
        )
        assert doc.title == "Empty Doc"
        assert doc.description_part == ""
        assert len(doc.info_part) == 0


class TestDocumentChunkDataclass:
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


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_loader_initialization(self):
        """Test loader initializes with correct path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = DocumentLoader(tmp_dir)
            assert loader.data_dir == Path(tmp_dir)

    def test_loader_with_string_path(self):
        """Test loader accepts string paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = DocumentLoader(str(tmp_dir))
            assert loader.data_dir == Path(tmp_dir)

    def test_load_document(self, tmp_path):
        """Test loading a single document."""
        # Create a test HTML file
        html_content = """
        <html>
            <head><title>Test Document</title></head>
            <body>
                <div class="text-container"><p>Test content</p></div>
                <!-- Начало разворачивающегося блока -->
                <div>
                    <button><span class="title-base">Section 1</span></button>
                    <div class="text-container"><p>Section content</p></div>
                </div>
                <!-- Конец разворачивающегося блока -->
            </body>
        </html>
        """
        test_file = tmp_path / "test.html"
        test_file.write_text(html_content)

        loader = DocumentLoader(tmp_path)
        doc = loader.load_document(test_file)

        assert doc.title == "Test Document"
        assert "Test content" in doc.description_part
        assert len(doc.info_part) == 1
        assert doc.info_part[0][0] == "Section 1"
        assert doc.source_path == str(test_file)

    def test_load_all_documents_empty_dir(self, tmp_path):
        """Test loading from empty directory."""
        loader = DocumentLoader(tmp_path)
        docs = list(loader.load_all_documents())
        assert len(docs) == 0

    def test_load_all_documents_multiple(self, tmp_path):
        """Test loading multiple documents."""
        # Create multiple test HTML files
        for i in range(3):
            html_content = f"""
            <html>
                <head><title>Document {i}</title></head>
                <body>
                    <div class="text-container"><p>Content {i}</p></div>
                </body>
            </html>
            """
            test_file = tmp_path / f"test_{i}.html"
            test_file.write_text(html_content)

        loader = DocumentLoader(tmp_path)
        docs = list(loader.load_all_documents())

        assert len(docs) == 3
        titles = [doc.title for doc in docs]
        assert "Document 0" in titles
        assert "Document 1" in titles
        assert "Document 2" in titles

    def test_load_all_documents_skips_non_html(self, tmp_path):
        """Test that non-HTML files are skipped."""
        # Create HTML and non-HTML files
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><head><title>HTML</title></head></html>")

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not HTML")

        loader = DocumentLoader(tmp_path)
        docs = list(loader.load_all_documents())

        assert len(docs) == 1
        assert docs[0].title == "HTML"

    def test_load_document_error_handling(self, tmp_path, caplog):
        """Test error handling when loading invalid file."""
        # Create an invalid HTML file
        test_file = tmp_path / "invalid.html"
        test_file.write_text("")

        loader = DocumentLoader(tmp_path)
        # Should not raise, but return a document with empty content
        doc = loader.load_document(test_file)
        assert doc.title == ""
