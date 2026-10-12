"""Shared pytest fixtures"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from esci.serving.app import create_app
from esci.serving.engine import Hit, SearchResult


class FakeEngine:
    """Deterministic engine: k hits with strictly falling scores."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def search(self, query: str, k: int) -> SearchResult:
        self.calls.append((query, k))
        hits = [
            Hit(f"P{i:04d}", f"product {i}", "brand", round(1.0 - i * 0.01, 4)) for i in range(k)
        ]
        return SearchResult(hits, cached=False, timings_ms={"total": 1.0})

    def info(self) -> dict[str, Any]:
        return {"model": "fake"}


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """The real HTTP layer over a fake engine. Entering the context runs the
    app's startup, as a server would."""
    with TestClient(create_app(FakeEngine)) as c:
        yield c
