"""Tests for generation module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json

from src.generation.generator import (
    GenerationProvider,
    MistralGenerator,
    GenerationError,
    create_generator,
)


class TestGenerationProviderAbstract:
    """Tests for GenerationProvider abstract class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that GenerationProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GenerationProvider()

    def test_abstract_methods_defined(self):
        """Test that abstract methods are defined."""
        assert hasattr(GenerationProvider, 'generate')
        assert hasattr(GenerationProvider, 'generate_stream')


class TestMistralGenerator:
    """Tests for MistralGenerator class."""

    @pytest.fixture
    def mock_http_client(self):
        """Create mock async HTTP client context."""
        with patch('src.generation.generator.httpx.AsyncClient') as mock:
            client_instance = Mock()
            mock.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            mock.return_value.__aexit__ = AsyncMock(return_value=False)
            yield client_instance

    @pytest.fixture
    def mock_sync_http_client(self):
        """Create mock sync HTTP client for generate() tests."""
        with patch('src.generation.generator.httpx.Client') as mock:
            client_instance = Mock()
            mock.return_value.__enter__ = Mock(return_value=client_instance)
            mock.return_value.__exit__ = Mock(return_value=False)
            yield client_instance

    def test_initialization(self):
        """Test generator initialization with default values."""
        generator = MistralGenerator(
            api_key="test_key",
            base_url="https://test.api.com/v1",
        )
        assert generator._model == "mistral-small-latest"
        assert generator._headers["Authorization"] == "Bearer test_key"
        assert generator._payload["temperature"] == 0.3
        assert generator._payload["top_p"] == 0.9
        assert generator._payload["max_tokens"] == 4096

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        generator = MistralGenerator(
            api_key="test_key",
            model="custom-model",
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )
        assert generator._model == "custom-model"
        assert generator._payload["temperature"] == 0.7
        assert generator._payload["top_p"] == 0.95
        assert generator._payload["max_tokens"] == 2048

    def test_generate_success(self, mock_sync_http_client):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated text"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_sync_http_client.post.return_value = mock_response

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test message"}]
        result = generator.generate(messages)

        assert result == "Generated text"
        mock_sync_http_client.post.assert_called_once()

    def test_generate_with_correct_payload(self, mock_sync_http_client):
        """Test that generate sends correct payload."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Result"}}]
        }
        mock_response.raise_for_status = Mock()
        mock_sync_http_client.post.return_value = mock_response

        generator = MistralGenerator(
            api_key="test_key",
            model="test-model",
            temperature=0.5,
        )
        messages = [{"role": "user", "content": "Test"}]
        generator.generate(messages)

        call_args = mock_sync_http_client.post.call_args
        assert call_args[1]["json"]["messages"] == messages
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["json"]["temperature"] == 0.5

    def test_generate_http_error(self, mock_sync_http_client):
        """Test handling of HTTP errors."""
        import httpx
        mock_sync_http_client.post.side_effect = httpx.HTTPError("Request failed")

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(GenerationError):
            generator.generate(messages)

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, mock_http_client):
        """Test successful streaming generation."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        # Create async iterator for aiter_lines
        async def async_iter_lines():
            for line in [
                "data: {\"choices\": [{\"delta\": {\"content\": \"Chunk 1\"}}]}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"Chunk 2\"}}]}",
                "data: [DONE]",
            ]:
                yield line
        
        mock_response.aiter_lines = async_iter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.stream.return_value = mock_stream_context

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test message"}]
        chunks = [chunk async for chunk in generator.generate_stream(messages)]

        assert len(chunks) == 2
        assert "Chunk 1" in chunks
        assert "Chunk 2" in chunks

    @pytest.mark.asyncio
    async def test_generate_stream_handles_malformed_json(self, mock_http_client):
        """Test that streaming handles malformed JSON gracefully."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        async def async_iter_lines():
            for line in [
                "data: invalid json",
                "data: {\"choices\": [{\"delta\": {\"content\": \"Valid chunk\"}}]}",
                "data: [DONE]",
            ]:
                yield line
        
        mock_response.aiter_lines = async_iter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.stream.return_value = mock_stream_context

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test"}]
        chunks = [chunk async for chunk in generator.generate_stream(messages)]

        assert len(chunks) == 1
        assert chunks[0] == "Valid chunk"

    @pytest.mark.asyncio
    async def test_generate_stream_http_error(self, mock_http_client):
        """Test handling of HTTP errors in streaming."""
        import httpx
        mock_http_client.stream.side_effect = httpx.HTTPError("Stream failed")

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(GenerationError):
            [chunk async for chunk in generator.generate_stream(messages)]

    @pytest.mark.asyncio
    async def test_generate_stream_stops_at_done(self, mock_http_client):
        """Test that streaming stops at [DONE] marker."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        async def async_iter_lines():
            for line in [
                "data: {\"choices\": [{\"delta\": {\"content\": \"Before done\"}}]}",
                "data: [DONE]",
                "data: {\"choices\": [{\"delta\": {\"content\": \"After done\"}}]}",
            ]:
                yield line
        
        mock_response.aiter_lines = async_iter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.stream.return_value = mock_stream_context

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test"}]
        chunks = [chunk async for chunk in generator.generate_stream(messages)]

        assert len(chunks) == 1
        assert chunks[0] == "Before done"

    @pytest.mark.asyncio
    async def test_generate_stream_skips_empty_content(self, mock_http_client):
        """Test that streaming skips empty content chunks."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        async def async_iter_lines():
            for line in [
                "data: {\"choices\": [{\"delta\": {\"content\": \"\"}}]}",
                "data: {\"choices\": [{\"delta\": {\"content\": \"Non-empty\"}}]}",
                "data: [DONE]",
            ]:
                yield line
        
        mock_response.aiter_lines = async_iter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_context.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.stream.return_value = mock_stream_context

        generator = MistralGenerator(api_key="test_key")
        messages = [{"role": "user", "content": "Test"}]
        chunks = [chunk async for chunk in generator.generate_stream(messages)]

        assert len(chunks) == 1
        assert chunks[0] == "Non-empty"


class TestGenerationError:
    """Tests for GenerationError exception."""

    def test_generation_error_creation(self):
        """Test creating GenerationError with message."""
        error = GenerationError("Test error message")
        assert str(error) == "Test error message"

    def test_generation_error_inheritance(self):
        """Test that GenerationError inherits from Exception."""
        error = GenerationError("Test")
        assert isinstance(error, Exception)


class TestCreateGenerator:
    """Tests for create_generator factory function."""

    def test_create_generator_uses_settings(self):
        """Test that factory uses settings values."""
        from src.config import settings

        with patch('src.generation.generator.MistralGenerator') as mock_mistral:
            mock_instance = Mock()
            mock_mistral.return_value = mock_instance

            create_generator()

            mock_mistral.assert_called_once()
            call_kwargs = mock_mistral.call_args[1]
            assert call_kwargs['api_key'] == settings.MISTRAL_API_KEY
            assert call_kwargs['base_url'] == settings.MISTRAL_BASE_URL
            assert call_kwargs['model'] == settings.LLM_MODEL
            assert call_kwargs['temperature'] == settings.LLM_TEMPERATURE
            assert call_kwargs['top_p'] == settings.LLM_TOP_P
            assert call_kwargs['max_tokens'] == settings.LLM_MAX_TOKENS
