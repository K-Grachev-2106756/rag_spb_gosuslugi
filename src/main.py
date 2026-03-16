"""Main entry point for the RAG Document Generation Service."""

import uvicorn

from src.config import settings


def main() -> None:
    """Run the FastAPI application with uvicorn."""
    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )


if __name__ == "__main__":
    main()
