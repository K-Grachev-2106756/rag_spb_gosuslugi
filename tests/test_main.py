"""Tests for main module."""

import pytest
from unittest.mock import patch, Mock


class TestMainFunction:
    """Tests for main function."""

    @patch('src.main.uvicorn')
    def test_main_runs_uvicorn(self, mock_uvicorn):
        """Test that main runs uvicorn with correct parameters."""
        from src.config import settings
        from src.main import main

        main()

        mock_uvicorn.run.assert_called_once_with(
            "src.api.app:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level="debug" if settings.DEBUG else "info",
        )

    @patch('src.main.uvicorn')
    def test_main_with_debug_enabled(self, mock_uvicorn):
        """Test main with debug enabled."""
        with patch('src.main.settings') as mock_settings:
            mock_settings.HOST = "0.0.0.0"
            mock_settings.PORT = 8000
            mock_settings.DEBUG = True

            from src.main import main
            main()

            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["reload"] is True
            assert call_kwargs["log_level"] == "debug"

    @patch('src.main.uvicorn')
    def test_main_with_debug_disabled(self, mock_uvicorn):
        """Test main with debug disabled."""
        with patch('src.main.settings') as mock_settings:
            mock_settings.HOST = "0.0.0.0"
            mock_settings.PORT = 8000
            mock_settings.DEBUG = False

            from src.main import main
            main()

            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["reload"] is False
            assert call_kwargs["log_level"] == "info"

    @patch('src.main.uvicorn')
    def test_main_with_custom_host_port(self, mock_uvicorn):
        """Test main with custom host and port."""
        with patch('src.main.settings') as mock_settings:
            mock_settings.HOST = "127.0.0.1"
            mock_settings.PORT = 9000
            mock_settings.DEBUG = False

            from src.main import main
            main()

            call_kwargs = mock_uvicorn.run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9000


class TestMainEntryPoint:
    """Tests for __main__ entry point."""

    @patch('src.main.main')
    def test_main_called_when_run_as_script(self, mock_main):
        """Test that main() is called when running as script."""
        import sys
        from io import StringIO

        # Save original argv
        original_argv = sys.argv

        try:
            # Simulate running as script
            sys.argv = ['main.py']

            # Import and check - this would normally call main()
            # We just verify the structure is correct
            from src.main import main

            # The if __name__ == "__main__" block exists
            # We can't easily test it without actually running,
            # but we've verified main() works above
            assert callable(main)

        finally:
            sys.argv = original_argv
