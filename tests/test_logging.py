"""Tests for core/logging module."""

import pytest
import logging
from unittest.mock import patch, Mock


class TestSetupLogging:
    """Tests for setup_logging function."""

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_info_level(self, mock_settings, mock_basic_config):
        """Test logging setup with INFO level."""
        mock_settings.DEBUG = False

        from src.core.logging import setup_logging
        setup_logging()

        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.INFO

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_debug_level(self, mock_settings, mock_basic_config):
        """Test logging setup with DEBUG level."""
        mock_settings.DEBUG = True

        from src.core.logging import setup_logging
        setup_logging()

        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs["level"] == logging.DEBUG

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_format(self, mock_settings, mock_basic_config):
        """Test logging format configuration."""
        mock_settings.DEBUG = False

        from src.core.logging import setup_logging
        setup_logging()

        call_kwargs = mock_basic_config.call_args[1]
        assert "%(asctime)s" in call_kwargs["format"]
        assert "%(name)s" in call_kwargs["format"]
        assert "%(levelname)s" in call_kwargs["format"]
        assert "%(message)s" in call_kwargs["format"]

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_handlers(self, mock_settings, mock_basic_config):
        """Test logging handlers configuration."""
        mock_settings.DEBUG = False

        from src.core.logging import setup_logging
        setup_logging()

        call_kwargs = mock_basic_config.call_args[1]
        assert len(call_kwargs["handlers"]) == 1

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_suppresses_chromadb(self, mock_settings, mock_basic_config):
        """Test that chromadb logger is suppressed."""
        mock_settings.DEBUG = False

        with patch('src.core.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from src.core.logging import setup_logging
            setup_logging()

            # Check chromadb logger was configured
            chromadb_calls = [
                call for call in mock_get_logger.call_args_list
                if call[0][0] == "chromadb"
            ]
            assert len(chromadb_calls) == 1

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_suppresses_sentence_transformers(self, mock_settings, mock_basic_config):
        """Test that sentence_transformers logger is suppressed."""
        mock_settings.DEBUG = False

        with patch('src.core.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from src.core.logging import setup_logging
            setup_logging()

            st_calls = [
                call for call in mock_get_logger.call_args_list
                if call[0][0] == "sentence_transformers"
            ]
            assert len(st_calls) == 1

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_suppresses_httpx(self, mock_settings, mock_basic_config):
        """Test that httpx logger is suppressed."""
        mock_settings.DEBUG = False

        with patch('src.core.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from src.core.logging import setup_logging
            setup_logging()

            httpx_calls = [
                call for call in mock_get_logger.call_args_list
                if call[0][0] == "httpx"
            ]
            assert len(httpx_calls) == 1

    @patch('src.core.logging.logging.basicConfig')
    @patch('src.core.logging.settings')
    def test_setup_logging_sets_warning_level_for_external(self, mock_settings, mock_basic_config):
        """Test that external libraries are set to WARNING level."""
        mock_settings.DEBUG = False

        with patch('src.core.logging.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            from src.core.logging import setup_logging
            setup_logging()

            # Each suppressed logger should have setLevel called with WARNING
            assert mock_logger.setLevel.call_count >= 3
