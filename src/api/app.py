"""FastAPI application for RAG document generation service."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# Global pipeline instance
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(data_dir=settings.DATA_DIR)
    return _pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Document Generation Service")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    logger.info(f"LLM Model: {settings.LLM_MODEL}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    
    # Initialize pipeline
    get_pipeline()
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Document Generation Service")


app = FastAPI(
    title="RAG Document Generation Service",
    description=(
        "Service for generating official documents (instructions/regulations) "
        "using RAG architecture with knowledge base search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models

class GenerateRequest(BaseModel):
    """Request model for document generation."""

    query: str = Field(
        ...,
        description="User query describing the document to generate",
        examples=["Мне необходимо составить инструкцию по получению соц обслуживания для жителя блокадного Ленинграда"],
    )


class GenerateResponse(BaseModel):
    """Response model for document generation."""

    query: str = Field(..., description="Original user query")
    document: str = Field(..., description="Generated document text")
    sources: list[str] = Field(..., description="List of source documents used")


class IndexRequest(BaseModel):
    """Request model for document indexing."""

    data_dir: str | None = Field(
        None,
        description="Optional path to data directory",
    )


class IndexResponse(BaseModel):
    """Response model for indexing result."""

    documents_processed: int = Field(..., description="Number of documents processed")
    total_chunks: int = Field(..., description="Total number of chunks created")
    status: str = Field(..., description="Indexing status")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    chunk_count: int = Field(..., description="Number of indexed chunks")


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check(pipeline: PipelineDep) -> HealthResponse:
    """Check service health status."""
    return HealthResponse(
        status="healthy",
        chunk_count=pipeline.chunk_count,
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest | None = None, pipeline: PipelineDep = None) -> IndexResponse:
    """
    Index documents from the data directory.

    This endpoint loads all HTML documents from the data directory,
    chunks them, and adds them to the vector store for retrieval.
    """
    try:
        result = pipeline.index_documents(
            data_dir=request.data_dir if request else None
        )

        return IndexResponse(
            documents_processed=result.documents_processed,
            total_chunks=result.total_chunks,
            status=result.status,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate_document(request: GenerateRequest, pipeline: PipelineDep) -> GenerateResponse:
    """
    Generate an official document based on user query.

    Uses RAG pipeline to:
    1. Search for relevant information in the knowledge base
    2. Generate a structured document using LLM

    Returns the generated document along with source references.
    """
    try:
        response = pipeline.query(request.query)

        return GenerateResponse(
            query=response.query,
            document=response.generated_document,
            sources=response.sources,
        )
    except Exception as e:
        logger.exception("Document generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document generation failed: {str(e)}",
        )


async def generate_stream_generator(request: GenerateRequest, pipeline: PipelineDep):
    """Generator for streaming document generation."""
    async for chunk in pipeline.query_stream(request.query):
        yield chunk


@app.post("/generate/stream")
async def generate_document_stream(request: GenerateRequest, pipeline: PipelineDep):
    """
    Generate a document with streaming response.

    Returns a stream of text chunks as the document is generated.
    """
    try:
        return StreamingResponse(
            generate_stream_generator(request, pipeline),
            media_type="text/plain; charset=utf-8",
        )
    except Exception as e:
        logger.exception("Streaming generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming generation failed: {str(e)}",
        )


@app.get("/stats")
async def get_stats(pipeline: PipelineDep) -> dict:
    """Get service statistics."""
    return {
        "chunk_count": pipeline.chunk_count,
        "data_dir": settings.DATA_DIR,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
    }
