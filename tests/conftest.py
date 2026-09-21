"""Shared pytest fixtures"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from esci.serving.app import app

@pytest.fixture(scope="session")
def client() -> TestClient:
    """An in-process HTTP client. No running server needed."""
    return TestClient(app)