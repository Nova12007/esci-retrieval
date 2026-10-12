"""HTTP surface for the retrieval service.

    ESCI_SERVING_DIR=artifacts/serving/biencoder_v3 \
        uvicorn esci.serving.app:create_app --factory --port 8000

The engine is injected, so tests run the real HTTP layer against a fake
engine without the 2 GB of serving artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from esci.serving.engine import Engine, RetrievalEngine

__version__ = "1.0.0"
STATIC = Path(__file__).parent / "static"

log = logging.getLogger("esci.serving")


def _configure_logging() -> None:
    """One JSON object per line on stdout: what Spaces, Cloud Run and any log
    pipeline ingest without a parser."""
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


class HitOut(BaseModel):
    product_id: str
    title: str
    brand: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[HitOut]
    cached: bool
    timings_ms: dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


def engine_from_env() -> Engine:
    path = os.environ.get("ESCI_SERVING_DIR")
    if not path:
        raise RuntimeError("set ESCI_SERVING_DIR to a folder from build_serving_artifacts.py")
    return RetrievalEngine.from_dir(Path(path))


def create_app(engine_factory: Callable[[], Engine] = engine_from_env) -> FastAPI:
    _configure_logging()
    state: dict[str, Engine] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load before accepting traffic: the 2 GB index takes seconds, and a
        # health check that passes before it is loaded would route requests
        # to a server that cannot answer them.
        t0 = time.perf_counter()
        engine = engine_factory()
        await engine.start()
        state["engine"] = engine
        log.info(json.dumps({"event": "ready", "load_s": round(time.perf_counter() - t0, 1)}))
        yield
        await engine.stop()

    app = FastAPI(
        title="ESCI Retrieval",
        version=__version__,
        description="Dense retrieval over 1.2M Amazon ESCI products.",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/info")
    def info() -> dict[str, Any]:
        """Model, index settings, cache hit rate, batching stats."""
        return {"version": __version__, **state["engine"].info()}

    @app.get("/search", response_model=SearchResponse)
    async def search(
        q: Annotated[str, Query(min_length=1, max_length=256)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> SearchResponse:
        """Top-k products for a query, with per-stage timings."""
        t0 = time.perf_counter()
        result = await state["engine"].search(q, k)
        latency = (time.perf_counter() - t0) * 1000
        log.info(
            json.dumps(
                {
                    "event": "search",
                    # a hash, not the text: enough to spot repeats, not to read queries
                    "query_hash": hashlib.sha256(q.encode()).hexdigest()[:12],
                    "k": k,
                    "cached": result.cached,
                    **{f"{name}_ms": v for name, v in result.timings_ms.items()},
                    "latency_ms": round(latency, 3),
                }
            )
        )
        return SearchResponse(
            query=q,
            results=[HitOut(**h.__dict__) for h in result.hits],
            cached=result.cached,
            timings_ms=result.timings_ms,
            latency_ms=round(latency, 3),
        )

    return app
