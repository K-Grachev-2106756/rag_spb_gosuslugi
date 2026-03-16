"""Tests for configuration module."""

import pytest
import os
from pydantic import ValidationError

from src.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_custom_values(self, monkeypatch):
        """Test settings can be overridden."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        
        settings = Settings(
            MISTRAL_API_KEY="test_key",
            MISTRAL_BASE_URL="https://custom.api.com/v1",
            LLM_MODEL="custom-model",
            CHUNK_SIZE=512,
            DEBUG=True,
            PORT=9000,
        )
        assert settings.MISTRAL_BASE_URL == "https://custom.api.com/v1"
        assert settings.LLM_MODEL == "custom-model"
        assert settings.CHUNK_SIZE == 512
        assert settings.DEBUG is True
        assert settings.PORT == 9000

    def test_settings_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_settings_temperature_range(self, monkeypatch):
        """Test temperature accepts valid float values."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        
        settings = Settings(
            MISTRAL_API_KEY="test_key",
            LLM_TEMPERATURE=0.7,
        )
        assert settings.LLM_TEMPERATURE == 0.7

    def test_settings_max_iterations(self, monkeypatch):
        """Test max iterations setting."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        
        settings = Settings(
            MISTRAL_API_KEY="test_key",
            SGR_MAX_ITERATIONS=5,
        )
        assert settings.SGR_MAX_ITERATIONS == 5
