from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("k", [1, 5, 10, 100])
def test_search_returns_exactly_k_results(client: TestClient, k: int) -> None:
    response = client.get("/search", params={"q": "running shoes", "k": k})
    assert response.status_code == 200
    assert len(response.json()["results"]) == k


@pytest.mark.parametrize("k", [0, -1, 101, 10_000])
def test_search_rejects_out_of_range_k(client: TestClient, k: int) -> None:
    response = client.get("/search", params={"q": "shoes", "k": k})
    assert response.status_code == 422


def test_search_requires_a_query(client: TestClient) -> None:
    assert client.get("/search").status_code == 422


def test_search_rejects_empty_query(client: TestClient) -> None:
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_scores_are_monotonically_non_increasing(client: TestClient) -> None:
    """A ranked list must never score a lower position higher.

    Trivially true for the stub. In week 8 this guards a real invariant --
    which is exactly why it is worth writing now.
    """
    results = client.get("/search", params={"q": "laptop", "k": 20}).json()["results"]
    scores = [hit["score"] for hit in results]
    assert scores == sorted(scores, reverse=True)


def test_info_reports_engine_details(client: TestClient) -> None:
    body = client.get("/info").json()
    assert body["model"] == "fake" and "version" in body


def test_demo_page_is_served_at_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200 and "<form" in response.text


def test_response_carries_timings_and_cache_flag(client: TestClient) -> None:
    body = client.get("/search", params={"q": "lamp", "k": 3}).json()
    assert body["cached"] is False and "total" in body["timings_ms"] and body["latency_ms"] >= 0
