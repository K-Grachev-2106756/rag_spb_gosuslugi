"""Tests for API endpoints."""

import pytest
import pytest_asyncio
from unittest.mock import patch, Mock
import httpx

from src.api.app import app, get_pipeline, GenerateRequest, GenerateResponse, IndexRequest, IndexResponse, HealthResponse
from src.pipeline import RAGPipeline, RAGResponse


@pytest.fixture
def mock_pipeline():
    """Create a mock RAG pipeline."""
    pipeline = Mock(spec=RAGPipeline)
    pipeline.chunk_count = 10
    pipeline.query = Mock(return_value=RAGResponse(
        query="Test query",
        generated_document="Generated doc",
        retrieved_results=["Result 1"],
        sources=["Source 1"],
    ))
    pipeline.query_stream = Mock(return_value=iter(["Chunk 1", "Chunk 2"]))
    pipeline.index_documents = Mock(return_value=Mock(
        documents_processed=5,
        total_chunks=20,
        status="success",
    ))
    return pipeline


@pytest_asyncio.fixture
async def client(mock_pipeline):
    """Create an async test client with mocked pipeline."""
    # Override the get_pipeline dependency
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
    
    # Clean up overrides
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client, mock_pipeline):
        """Test health check returns healthy status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["chunk_count"] == 10


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats(self, client, mock_pipeline):
        """Test stats endpoint returns configuration."""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "chunk_count" in data
        assert "embedding_model" in data
        assert "llm_model" in data


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    @pytest.mark.asyncio
    async def test_generate_missing_query(self, client, mock_pipeline):
        """Test generate endpoint validates required query."""
        response = await client.post("/generate", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_generate_with_query(self, client, mock_pipeline):
        """Test generate endpoint accepts query."""
        response = await client.post(
            "/generate",
            json={"query": "Test query for document generation"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Test query"
        assert data["document"] == "Generated doc"
        assert data["sources"] == ["Source 1"]

    @pytest.mark.asyncio
    async def test_generate_handles_exception(self, client, mock_pipeline):
        """Test generate endpoint handles exceptions."""
        mock_pipeline.query = Mock(side_effect=Exception("Test error"))
        
        response = await client.post(
            "/generate",
            json={"query": "Test query"},
        )
        assert response.status_code == 500
        assert "Document generation failed" in response.json()["detail"]


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    @pytest.mark.asyncio
    async def test_index_documents(self, client, mock_pipeline):
        """Test indexing documents."""
        response = await client.post("/index", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 5
        assert data["total_chunks"] == 20
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_index_documents_with_data_dir(self, client, mock_pipeline):
        """Test indexing with custom data directory."""
        response = await client.post("/index", json={"data_dir": "/custom/path"})
        assert response.status_code == 200
        mock_pipeline.index_documents.assert_called_once_with(data_dir="/custom/path")

    @pytest.mark.asyncio
    async def test_index_documents_value_error(self, client, mock_pipeline):
        """Test indexing handles ValueError."""
        mock_pipeline.index_documents = Mock(side_effect=ValueError("No documents found"))
        
        response = await client.post("/index", json={})
        assert response.status_code == 400
        assert "No documents found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_index_documents_general_error(self, client, mock_pipeline):
        """Test indexing handles general exceptions."""
        mock_pipeline.index_documents = Mock(side_effect=Exception("Unexpected error"))
        
        response = await client.post("/index", json={})
        assert response.status_code == 500
        assert "Indexing failed" in response.json()["detail"]


class TestStreamEndpoint:
    """Tests for /generate/stream endpoint."""

    @pytest.mark.asyncio
    async def test_stream_generate(self, client, mock_pipeline):
        """Test streaming generation endpoint."""
        # Set up the mock to return an async generator (as the real pipeline does)
        async def async_gen(query):
            for chunk in ["Chunk 1", "Chunk 2"]:
                yield chunk

        mock_pipeline.query_stream = async_gen

        response = await client.post(
            "/generate/stream",
            json={"query": "Test streaming query"},
        )
        assert response.status_code == 200


class TestGetPipeline:
    """Tests for get_pipeline function."""

    def test_get_pipeline_creates_instance(self):
        """Test that get_pipeline creates a new pipeline instance."""
        # Import module properly - use sys.modules to avoid app shadowing
        import sys
        app_module = sys.modules['src.api.app']
        
        with patch('src.api.app.RAGPipeline') as mock_rag_pipeline_class:
            # Temporarily set _pipeline to None
            original = app_module._pipeline
            app_module._pipeline = None
            
            try:
                mock_pipeline = Mock()
                mock_rag_pipeline_class.return_value = mock_pipeline
                
                result = app_module.get_pipeline()
                
                assert result == mock_pipeline
                mock_rag_pipeline_class.assert_called_once()
            finally:
                app_module._pipeline = original

    def test_get_pipeline_returns_cached(self):
        """Test that get_pipeline returns cached instance."""
        import sys
        app_module = sys.modules['src.api.app']
        
        with patch('src.api.app.RAGPipeline') as mock_rag_pipeline_class:
            original = app_module._pipeline
            app_module._pipeline = None
            
            try:
                mock_pipeline1 = Mock()
                mock_rag_pipeline_class.return_value = mock_pipeline1
                
                # First call creates instance
                result1 = app_module.get_pipeline()
                
                # Second call returns cached instance
                result2 = app_module.get_pipeline()
                
                assert result1 == mock_pipeline1
                assert result2 == mock_pipeline1  # Same instance
                assert mock_rag_pipeline_class.call_count == 1
            finally:
                app_module._pipeline = original


class TestLifespan:
    """Tests for application lifespan."""

    @pytest.mark.asyncio
    async def test_lifespan_startup(self):
        """Test lifespan startup logging."""
        from src.api.app import lifespan
        
        app_mock = Mock()
        
        with patch('src.api.app.logger') as mock_logger:
            async with lifespan(app_mock):
                pass
            
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Starting RAG Document Generation Service" in call for call in calls)

    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self):
        """Test lifespan shutdown logging."""
        from src.api.app import lifespan
        
        app_mock = Mock()
        
        with patch('src.api.app.logger') as mock_logger:
            async with lifespan(app_mock):
                pass
            
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Shutting down RAG Document Generation Service" in call for call in calls)


class TestGenerateRequest:
    """Tests for GenerateRequest model."""

    def test_generate_request_creation(self):
        """Test creating GenerateRequest."""
        request = GenerateRequest(query="Test query")
        assert request.query == "Test query"


class TestGenerateResponse:
    """Tests for GenerateResponse model."""

    def test_generate_response_creation(self):
        """Test creating GenerateResponse."""
        response = GenerateResponse(
            query="Test query",
            document="Generated doc",
            sources=["Source 1", "Source 2"],
        )
        assert response.query == "Test query"
        assert response.document == "Generated doc"
        assert len(response.sources) == 2


class TestIndexRequest:
    """Tests for IndexRequest model."""

    def test_index_request_creation(self):
        """Test creating IndexRequest."""
        request = IndexRequest(data_dir="/test/path")
        assert request.data_dir == "/test/path"

    def test_index_request_optional_data_dir(self):
        """Test IndexRequest with optional data_dir."""
        request = IndexRequest()
        assert request.data_dir is None


class TestIndexResponse:
    """Tests for IndexResponse model."""

    def test_index_response_creation(self):
        """Test creating IndexResponse."""
        response = IndexResponse(
            documents_processed=5,
            total_chunks=20,
            status="success",
        )
        assert response.documents_processed == 5
        assert response.total_chunks == 20
        assert response.status == "success"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response_creation(self):
        """Test creating HealthResponse."""
        response = HealthResponse(
            status="healthy",
            chunk_count=100,
        )
        assert response.status == "healthy"
        assert response.chunk_count == 100
