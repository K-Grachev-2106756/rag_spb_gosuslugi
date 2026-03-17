"""Vector store module using ChromaDB."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.data_processing.chunking import RecursiveChunker
from src.data_processing.loader import Document, DocumentChunk
from src.embeddings import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document chunks using ChromaDB.
    
    Provides indexing and retrieval capabilities for RAG pipeline.
    """

    def __init__(
        self,
        persist_dir: str | Path = settings.CHROMA_PERSIST_DIR,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistent storage.
            embedding_provider: Embedding provider instance.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_provider = embedding_provider or create_embedding_provider()

        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"VectorStore initialized at {self.persist_dir}")

    def add_chunk(self, chunk: DocumentChunk) -> None:
        """
        Add a single chunk to the vector store.
        
        Args:
            chunk: DocumentChunk to add.
        """
        embedding = self._embedding_provider.embed(chunk.content)

        self._collection.add(
            ids=[chunk.chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[
                {
                    "document_title": chunk.document_title,
                    "metadata": chunk.metadata,
                    "parent_section": chunk.parent_section,
                }
            ],
        )
        logger.debug(f"Added chunk: {chunk.chunk_id}")

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Add multiple chunks to the vector store in batch.
        
        Args:
            chunks: List of DocumentChunk objects.
        """
        if not chunks:
            return

        # Prepare batch data
        ids = [chunk.chunk_id for chunk in chunks]
        contents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_title": chunk.document_title,
                "metadata": chunk.metadata,
                "parent_section": chunk.parent_section,
            }
            for chunk in chunks
        ]

        # Generate embeddings in batch
        embeddings = self._embedding_provider.embed_batch(contents)

        # Add to collection in batches (ChromaDB limit: 5461 per batch)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_documents = contents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            self._collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def index_document(self, document: Document) -> int:
        """
        Index a single document.
        
        Args:
            document: Document to index.
            
        Returns:
            Number of chunks created.
        """
        chunker = RecursiveChunker()
        chunks = list(chunker.chunk_document(document))
        self.add_chunks(chunks)
        return len(chunks)

    def index_all_documents(self, documents: list[Document]) -> int:
        """
        Index multiple documents.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            Total number of chunks created.
        """
        chunker = RecursiveChunker()
        all_chunks = []

        for document in documents:
            chunks = list(chunker.chunk_document(document))
            all_chunks.extend(chunks)

        self.add_chunks(all_chunks)
        return len(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Search for relevant chunks.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of search results with content and metadata.
        """
        query_embedding = self._embedding_provider.embed(query, is_query=True)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    }
                )

        logger.debug(f"Search returned {len(formatted_results)} results")
        return formatted_results

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self._client.delete_collection("documents")
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")

    @property
    def count(self) -> int:
        """Return the number of chunks in the store."""
        return self._collection.count()
