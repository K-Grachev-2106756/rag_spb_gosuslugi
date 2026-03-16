"""Generation module using Mistral API."""

from abc import ABC, abstractmethod
import logging
from typing import AsyncIterator, Iterator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class GenerationProvider(ABC):
    """Abstract base class for text generation providers."""

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """Generate text based on messages."""
        pass

    @abstractmethod
    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Generate text as a stream of chunks."""
        pass


class MistralGenerator(GenerationProvider):
    """
    Text generator using Mistral API.

    The generator creates responses based on provided prompts and context.
    """

    def __init__(
        self,
        api_key: str = settings.MISTRAL_API_KEY,
        base_url: str = settings.MISTRAL_BASE_URL,
        model: str = settings.LLM_MODEL,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ):
        """
        Initialize the Mistral generator.

        Args:
            api_key: Mistral API key.
            base_url: Mistral API base URL.
            model: Model name to use for generation.
            temperature: Sampling temperature for generation.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.
        """
        self._base_url = base_url
        self._model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._payload = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        logger.info(
            f"MistralGenerator initialized with model: {model}, "
            f"temperature: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}"
        )

    def generate(self, messages: list[dict]) -> str:
        """
        Generate a complete response.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Generated text.
        """
        logger.info(f"Generating with model: {self._model}")

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=dict(
                        messages=messages,
                        **self._payload,
                    ),
                )
                response.raise_for_status()
                result = response.json()

                generated_text = result["choices"][0]["message"]["content"]
                logger.info("Generation completed")
                return generated_text

        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            raise GenerationError(f"Failed to generate: {e}")

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """
        Generate response as a stream of chunks.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Yields:
            Text chunks as they are generated.
        """
        logger.info(f"Streaming generation with model: {self._model}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=dict(
                        messages=messages,
                        stream=True,
                        **self._payload,
                    ),
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break

                            try:
                                import json
                                chunk = json.loads(data)
                                content = chunk["choices"][0]["delta"].get(
                                    "content", ""
                                )
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

        except httpx.HTTPError as e:
            logger.error(f"Streaming request failed: {e}")
            raise GenerationError(f"Failed to stream: {e}")


class GenerationError(Exception):
    """Exception raised for generation errors."""
    pass


def create_generator() -> GenerationProvider:
    """Factory function to create a generation provider."""
    return MistralGenerator(
        api_key=settings.MISTRAL_API_KEY,
        base_url=settings.MISTRAL_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        top_p=settings.LLM_TOP_P,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
