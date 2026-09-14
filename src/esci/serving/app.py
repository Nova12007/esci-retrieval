"""HTTP surface for the retrieval service"""

import time
from typing import Annotated

from fastapi import FastAPI, Query
from pydantic import BaseModel

__version__ = "0.1.0"

app = FastAPI(
    title="ESCI Retrieval",
    version=__version__,
    description="Two-stage neural retrieval over Amazon ESCI products.",
)


class Hit(BaseModel):
    """One ranked product"""

    product_id: str
    title: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Hit]
    latency__ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe. Cloud Run and your load test both hit this."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/search", response_model=SearchResponse)
def search(
    q: Annotated[str, Query(min_length=1, max_length=256)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SearchResponse:
    """Return the top-k products for a query."""
    start = time.perf_counter()
    results = [
        Hit(
            product_id=f"STUB{i:04d}",
            title=f"stub result {i} for {q!r}",
            score=round(1.0 - i * 0.01, 4),
        )
        for i in range(k)
    ]
    return SearchResponse(
        query=q, results=results, latency__ms=round((time.perf_counter() - start) * 1000, 3)
    )
