"""Document loader for HTML files."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.data_processing.parse import extract_blocks

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document with metadata."""

    content: str
    document_title: str
    metadata: str
    chunk_id: str
    parent_section: str = ""


@dataclass
class Document:
    """Represents a loaded document with its components."""

    title: str
    description_part: str
    info_part: list[tuple[str, str]]
    source_path: str


class DocumentLoader:
    """
    Loads and parses HTML documents from the data directory.
    
    Uses the existing parse.py module for HTML extraction.
    """

    def __init__(self, data_dir: str | Path):
        """
        Initialize the document loader.
        
        Args:
            data_dir: Path to the directory containing HTML files.
        """
        self.data_dir = Path(data_dir)
        logger.info(f"DocumentLoader initialized with data_dir: {self.data_dir}")

    def load_document(self, file_path: str | Path) -> Document:
        """
        Load a single HTML document.
        
        Args:
            file_path: Path to the HTML file.
            
        Returns:
            Document object with parsed content.
        """
        file_path = Path(file_path)
        logger.debug(f"Loading document: {file_path}")

        title, description_part, info_part = extract_blocks(str(file_path))

        return Document(
            title=title,
            description_part=description_part,
            info_part=info_part,
            source_path=str(file_path),
        )

    def load_all_documents(self) -> Iterator[Document]:
        """
        Load all HTML documents from the data directory.
        
        Yields:
            Document objects one by one.
        """
        html_files = list(self.data_dir.glob("*.html"))
        logger.info(f"Found {len(html_files)} HTML files in {self.data_dir}")

        for html_file in html_files:
            try:
                yield self.load_document(html_file)
            except Exception as e:
                logger.error(f"Failed to load {html_file}: {e}")
                continue
