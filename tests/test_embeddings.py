"""Tests for embeddings module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.embeddings.embedding import (
    EmbeddingProvider,
    SentenceTransformerEmbeddings,
    create_embedding_provider,
)


class TestEmbeddingProviderAbstract:
    """Tests for EmbeddingProvider abstract class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_abstract_methods_defined(self):
        """Test that abstract methods are defined."""
        assert hasattr(EmbeddingProvider, 'embed')
        assert hasattr(EmbeddingProvider, 'embed_batch')
        assert hasattr(EmbeddingProvider, 'dimension')


class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformerEmbeddings class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        with patch('src.embeddings.embedding.SentenceTransformer') as mock:
            mock_instance = Mock()
            mock_instance.get_sentence_embedding_dimension.return_value = 1024
            mock_instance.encode.return_value = np.zeros(1024).tolist()
            mock_instance.eval.return_value = mock_instance  # Chain eval()
            mock.return_value = mock_instance
            yield mock_instance

    def test_initialization(self, mock_model):
        """Test embedding provider initialization."""
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings(model_name="test-model")
            assert provider.dimension == 1024

    def test_initialization_with_custom_model(self, mock_model):
        """Test initialization with custom model name."""
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings(model_name="custom-model")
            assert provider.dimension == 1024

    def test_embed_single_text(self, mock_model):
        """Test embedding a single text."""
        mock_model.encode.return_value = np.ones(1024)
        
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            result = provider.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 1024
            assert all(x == 1.0 for x in result)

    def test_embed_batch(self, mock_model):
        """Test embedding a batch of texts."""
        mock_model.encode.return_value = np.ones((3, 1024))
        
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            texts = ["text 1", "text 2", "text 3"]
            results = provider.embed_batch(texts)

            assert isinstance(results, list)
            assert len(results) == 3
            assert all(len(r) == 1024 for r in results)

    def test_embed_batch_with_custom_batch_size(self, mock_model):
        """Test batch embedding with custom batch size."""
        mock_model.encode.return_value = np.ones((2, 1024))
        
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            texts = ["text 1", "text 2"]
            results = provider.embed_batch(texts, batch_size=32)

            assert len(results) == 2
            mock_model.encode.assert_called_once()

    def test_embed_batch_with_progress(self, mock_model):
        """Test batch embedding with progress bar."""
        mock_model.encode.return_value = np.ones((2, 1024))
        
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            texts = ["text 1", "text 2"]
            results = provider.embed_batch(texts, show_progress=True)

            assert len(results) == 2

    def test_dimension_property(self, mock_model):
        """Test dimension property."""
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            assert provider.dimension == 1024

    def test_embed_returns_list_not_numpy(self, mock_model):
        """Test that embed returns Python list, not numpy array."""
        mock_model.encode.return_value = np.array([1.0, 2.0, 3.0])
        
        with patch('src.embeddings.embedding.SentenceTransformer', return_value=mock_model):
            provider = SentenceTransformerEmbeddings()
            result = provider.embed("test")
            assert isinstance(result, list)
            assert not isinstance(result, np.ndarray)


class TestCreateEmbeddingProvider:
    """Tests for create_embedding_provider factory function."""

    @patch('src.embeddings.embedding.SentenceTransformer')
    def test_create_provider_returns_instance(self, mock_st):
        """Test that factory returns SentenceTransformerEmbeddings instance."""
        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_instance

        provider = create_embedding_provider()

        assert isinstance(provider, SentenceTransformerEmbeddings)

    @patch('src.embeddings.embedding.SentenceTransformer')
    def test_create_provider_uses_settings_model(self, mock_st):
        """Test that factory uses settings model name."""
        from src.config import settings

        mock_instance = Mock()
        mock_instance.get_sentence_embedding_dimension.return_value = 1024
        mock_st.return_value = mock_instance

        create_embedding_provider()

        # Check that SentenceTransformer was called with the settings model
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert settings.EMBEDDING_MODEL in call_args[0] or call_args[1].get('model_name_or_path') == settings.EMBEDDING_MODEL
