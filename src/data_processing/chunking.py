"""Recursive chunking strategy for document processing."""

import hashlib
import logging
from typing import Iterator, Optional

from transformers import AutoTokenizer

from src.config import settings
from src.data_processing.loader import Document, DocumentChunk

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Implements recursive chunking strategy for documents.

    Each chunk contains:
    - Document title
    - Description (static part)
    - Info title + info content fragment
    - Metadata information

    This ensures context preservation for RAG retrieval.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in tokens.
            chunk_overlap: Overlap between consecutive chunks in tokens.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer from the embedding model
        self._tokenizer = self._load_tokenizer()
        
        logger.info(
            f"RecursiveChunker initialized: chunk_size={chunk_size} tokens, "
            f"overlap={chunk_overlap} tokens"
        )

    def _load_tokenizer(self) -> callable:
        """Load tokenizer from the embedding model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
            if tokenizer is None:
                logger.warning(
                    f"No tokenizer found for {settings.EMBEDDING_MODEL}, "
                    "using fallback character-based chunking"
                )
                return None
            logger.info(f"Tokenizer loaded from {settings.EMBEDDING_MODEL}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using fallback")
            return None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: estimate tokens (1 token ≈ 3.5 characters for Russian/English)
        return len(text) // 3.5

    def _generate_chunk_id(self, content: str, index: int) -> str:
        """Generate a unique chunk ID based on content hash."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"chunk_{content_hash}_{index}"

    def _create_chunk_with_context(
        self,
        document_title: str,
        description: str,
        info_title: str,
        content: str,
        full_content: str,
        index: int,
        parent_section: str = "",
    ) -> DocumentChunk:
        """
        Create a chunk with document context (title + description + info_title).

        Args:
            content: The main content of the chunk.
            document_title: Title of the source document.
            description: Static description part of the document.
            info_title: Title of the info section.
            index: Index of the chunk.
            parent_section: Parent section name if applicable.

        Returns:
            DocumentChunk with full context.
        """
        chunk_content = self._build_contextual_content(
            document_title=document_title,
            description=description,
            info_title=info_title,
            content=content,
        )  # Chunk to embed

        return DocumentChunk(
            content=chunk_content,
            document_title=document_title,
            metadata=self._build_contextual_content(
                document_title=document_title, 
                description=description,
                info_title=info_title,
                content=full_content,
            ),  # Chunk for augmented generation
            chunk_id=self._generate_chunk_id(chunk_content, index),
            parent_section=parent_section,
        )

    def _build_contextual_content(
        self,
        document_title: Optional[str] = None,
        description: Optional[str] = None,
        info_title: Optional[str] = None,
        content: Optional[str] = None,
    ) -> str:
        """
        Build content string with document context.

        Format: (
            [Document: title] 
            [Description: description] 
            [Info: info_title] 
            Content
        )
        """
        context_parts = []

        if document_title:
            context_parts.append(f"[Документ: {document_title}]")

        if description:
            context_parts.append(f"[Описание: {description}]")

        if info_title:
            context_parts.append(f"[Раздел: {info_title}]")

        if content:
            context_parts.append(content)

        return "\n".join(context_parts)

    def _split_text_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """
        Recursively split text into token-limited chunks with overlap.

        Strategy:
            1. If text fits - return as is.
            2. Try splitting by paragraph.
            3. Then by sentence.
            4. Then by word.
            5. Fallback to hard token split.
        """

        if not text.strip():
            return []

        if not self._tokenizer:
            # Fallback to char-based logic
            return self._split_by_char_size(
                text,
                max_tokens * 4,
                overlap_tokens * 4,
            )

        def token_len(t: str) -> int:
            return len(self._tokenizer.encode(t))

        def hard_split(t: str) -> list[str]:
            tokens = self._tokenizer.encode(t)
            chunks = []
            start = 0

            while start < len(tokens):
                end = start + max_tokens
                chunk_tokens = tokens[start:end]
                chunk_text = self._tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True,
                )
                chunks.append(chunk_text)

                if end >= len(tokens):
                    break

                start = end - overlap_tokens

            return chunks

        def recursive_split(t: str) -> list[str]:
            if token_len(t) <= max_tokens:
                return [t]

            separators = ["\n\n", "\n", "\t", ". ", "? ", "! ", ".", "?", "!", " "]

            for sep in separators:
                if sep not in t:
                    continue

                parts = t.split(sep)
                chunks = []
                current = ""

                for i, part in enumerate(parts):
                    piece = part if i == len(parts) - 1 else part + sep

                    if token_len(current + piece) <= max_tokens:
                        current += piece
                    else:
                        if current:
                            chunks.extend(recursive_split(current))
                        current = piece

                if current:
                    chunks.extend(recursive_split(current))

                if len(chunks) > 1:  # separator was applied
                    return chunks

            return hard_split(t)

        base_chunks = recursive_split(text)

        # overlap
        if overlap_tokens <= 0 or len(base_chunks) <= 1:
            return base_chunks

        final_chunks = []
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                final_chunks.append(chunk)
                continue

            prev_tokens = self._tokenizer.encode(final_chunks[-1])
            curr_tokens = self._tokenizer.encode(chunk)

            overlap = prev_tokens[-overlap_tokens:]
            merged_tokens = overlap + curr_tokens

            merged_text = self._tokenizer.decode(
                merged_tokens,
                skip_special_tokens=True,
            )

            final_chunks.append(merged_text)

        return final_chunks

    def _split_by_char_size(self, text: str, max_chars: int, overlap_chars: int) -> list[str]:
        """Fallback: split text by character size."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(" ")
                if last_space > 0:
                    chunk = chunk[:last_space]

            chunks.append(chunk)
            start = end - overlap_chars
            if start >= len(text):
                break

        return chunks

    def chunk_document(self, document: Document) -> Iterator[DocumentChunk]:
        """
        Chunk a document using recursive strategy.

        Args:
            document: Document to chunk.

        Yields:
            DocumentChunk objects with contextual information.
        """
        logger.debug(f"Chunking document: {document.title}")
        chunk_index = 0

        # Static context: title + description (same for all chunks)
        static_context = self._build_contextual_content(
            document_title=document.title, 
            description=document.description_part,
        )
        static_tokens = self._count_tokens(static_context)

        # Available tokens for info content
        content_max_tokens = 64 if (free_space := self.chunk_size - static_tokens) < 64 else free_space
        content_overlap_tokens = min(self.chunk_overlap, content_max_tokens // 2)
        
        # Process info_part: each (info_title, info) tuple is processed separately
        for info_index, (info_title, info_content) in enumerate(document.info_part):
            if not info_content.strip():
                continue

            # Chunk the info content
            content_chunks = self._split_text_by_tokens(
                info_content, content_max_tokens, content_overlap_tokens
            )

            for chunk_text in content_chunks:
                if chunk_text.strip():
                    yield self._create_chunk_with_context(
                        document_title=document.title,
                        description=document.description_part,
                        info_title=info_title,
                        content=chunk_text,
                        full_content=info_content,
                        index=chunk_index,
                        parent_section=f"info_{info_index}",
                    )
                    chunk_index += 1

        logger.info(f"Document '{document.title}' chunked into {chunk_index} chunks")

    def chunk_all_documents(
        self, documents: Iterator[Document]
    ) -> Iterator[DocumentChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: Iterator of Document objects.

        Yields:
            DocumentChunk objects from all documents.
        """
        for document in documents:
            yield from self.chunk_document(document)
