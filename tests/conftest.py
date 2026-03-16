"""Pytest configuration."""

import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> str:
    """Create a temporary directory for test data."""
    return str(tmp_path_factory.mktemp("data"))
