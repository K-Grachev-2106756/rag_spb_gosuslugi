"""Tests for HTML parsing module."""

import pytest
import tempfile
from pathlib import Path

from src.data_processing.parse import (
    extract_blocks,
    parse_title,
    parse_text_containers,
    parse_expandable_block,
    postprocess_text,
)


class TestPostprocessText:
    """Tests for postprocess_text function."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert postprocess_text("  hello   world  ") == "hello world"

    def test_remove_newlines(self):
        """Test newline removal."""
        assert postprocess_text("hello\nworld") == "hello world"

    def test_remove_multiple_spaces(self):
        """Test multiple spaces removal."""
        assert postprocess_text("a    b    c") == "a b c"

    def test_empty_string(self):
        """Test empty string handling."""
        assert postprocess_text("") == ""

    def test_only_whitespace(self):
        """Test whitespace-only string."""
        assert postprocess_text("   ") == ""


class TestParseTitle:
    """Tests for parse_title function."""

    def test_parse_title_from_html(self):
        """Test parsing title from HTML."""
        html = "<html><head><title>Test Title</title></head><body></body></html>"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            try:
                title = parse_title(html)
                assert title == "Test Title"
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_parse_title_no_title_tag(self):
        """Test parsing when no title tag exists."""
        html = "<html><body></body></html>"
        title = parse_title(html)
        assert title == ""

    def test_parse_title_with_whitespace(self):
        """Test parsing title with surrounding whitespace."""
        html = "<html><head><title>  Test Title  </title></head></html>"
        title = parse_title(html)
        assert title == "Test Title"


class TestParseTextContainers:
    """Tests for parse_text_containers function."""

    def test_parse_text_containers_basic(self):
        """Test basic text container parsing."""
        html = """
        <div class="text-container">
            <p>First paragraph</p>
            <p>Second paragraph</p>
        </div>
        """
        result = parse_text_containers(html)
        assert len(result) == 1
        assert "First paragraph" in result[0]
        assert "Second paragraph" in result[0]

    def test_parse_text_containers_with_title(self):
        """Test text container parsing with title."""
        html = """
        <div class="text-container">
            <div class="title-base">Container Title</div>
            <p>Content paragraph</p>
        </div>
        """
        result = parse_text_containers(html, with_title=True)
        assert len(result) == 1
        assert "Container Title" in result[0]

    def test_parse_text_containers_empty(self):
        """Test parsing empty containers."""
        html = '<div class="text-container"></div>'
        result = parse_text_containers(html)
        assert result == []

    def test_parse_text_containers_multiple(self):
        """Test parsing multiple containers."""
        html = """
        <div class="text-container"><p>Container 1</p></div>
        <div class="text-container"><p>Container 2</p></div>
        """
        result = parse_text_containers(html)
        assert len(result) == 2


class TestParseExpandableBlock:
    """Tests for parse_expandable_block function."""

    def test_parse_expandable_block_basic(self):
        """Test basic expandable block parsing."""
        html = """
        <div>
            <button><span class="title-base">Block Title</span></button>
            <div class="text-container"><p>Block content</p></div>
        </div>
        """
        title, content = parse_expandable_block(html)
        assert title == "Block Title"
        assert "Block content" in content

    def test_parse_expandable_block_with_list(self):
        """Test parsing block with lists."""
        html = """
        <div>
            <button><span class="title-base">List Title</span></button>
            <div class="text-container">
                <ul><li>Item 1</li><li>Item 2</li></ul>
            </div>
        </div>
        """
        title, content = parse_expandable_block(html)
        assert title == "List Title"
        assert "- Item 1" in content
        assert "- Item 2" in content

    def test_parse_expandable_block_with_ordered_list(self):
        """Test parsing block with ordered lists."""
        html = """
        <div>
            <button><span class="title-base">Steps</span></button>
            <div class="text-container">
                <ol><li>Step 1</li><li>Step 2</li></ol>
            </div>
        </div>
        """
        title, content = parse_expandable_block(html)
        assert title == "Steps"
        assert "1. Step 1" in content
        assert "2. Step 2" in content

    def test_parse_expandable_block_empty(self):
        """Test parsing empty block."""
        html = "<div></div>"
        title, content = parse_expandable_block(html)
        assert title == ""
        assert content == ""


class TestExtractBlocks:
    """Tests for extract_blocks function."""

    def test_extract_blocks_basic(self):
        """Test basic block extraction."""
        html = """
        <html>
            <head><title>Document Title</title></head>
            <body>
                <div class="text-container"><p>Before content</p></div>
                <!-- Начало разворачивающегося блока -->
                <div>
                    <button><span class="title-base">Block 1</span></button>
                    <div class="text-container"><p>Block 1 content</p></div>
                </div>
                <!-- Конец разворачивающегося блока -->
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            try:
                title, description, info = extract_blocks(f.name)
                assert title == "Document Title"
                assert "Before content" in description
                assert len(info) == 1
                assert info[0][0] == "Block 1"
                assert "Block 1 content" in info[0][1]
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_extract_blocks_multiple_blocks(self):
        """Test extraction with multiple blocks."""
        html = """
        <html>
            <head><title>Multi Block</title></head>
            <body>
                <!-- Начало разворачивающегося блока -->
                <div>
                    <button><span class="title-base">Block A</span></button>
                    <div class="text-container"><p>Content A</p></div>
                </div>
                <!-- Конец разворачивающегося блока -->
                <!-- Начало разворачивающегося блока -->
                <div>
                    <button><span class="title-base">Block B</span></button>
                    <div class="text-container"><p>Content B</p></div>
                </div>
                <!-- Конец разворачивающегося блока -->
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            try:
                title, description, info = extract_blocks(f.name)
                assert len(info) == 2
                assert info[0][0] == "Block A"
                assert info[1][0] == "Block B"
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_extract_blocks_no_expandable(self):
        """Test extraction when no expandable blocks exist."""
        html = """
        <html>
            <head><title>No Blocks</title></head>
            <body>
                <div class="text-container"><p>Only before content</p></div>
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            try:
                title, description, info = extract_blocks(f.name)
                assert title == "No Blocks"
                assert "Only before content" in description
                assert info == []
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_extract_blocks_with_sorry_text(self):
        """Test that sorry text is filtered from description."""
        html = """
        <html>
            <head><title>Sorry Test</title></head>
            <body>
                <div class="text-container">
                    <p>Good content</p>
                    <p>Приносим извинения за доставленные неудобства.</p>
                </div>
            </body>
        </html>
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html)
            f.flush()
            try:
                title, description, info = extract_blocks(f.name)
                assert title == "Sorry Test"
                # The description may be empty if there's no content before expandable blocks
                # The sorry text should be filtered from description
                assert "Приносим извинения за доставленные неудобства" not in description
            finally:
                Path(f.name).unlink(missing_ok=True)
