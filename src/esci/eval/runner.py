"""Evaluation path for every retrieval system"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import polars as pl
from tqdm.auto import tqdm

from esci.eval.aggregate import Summary, score_rerank, score_retrieval, summarise

PER_QUERY = Path("artifacts/per_query")
RESULTS = Path("benchmarks/results.jsonl")


class Ranker(Protocol):
    name: str

    def rank_candidates(
        self, queries: dict[int, str], candidates: dict[int, list[str]]
    ) -> dict[int, list[str]]:
        """Mode A: reorder each query's judged candidate set."""
        ...

    def retrieve(self, queries: dict[int, str], k: int) -> dict[int, list[str]]:
        """Mode B: retrieve top-k from the full corpus."""
        ...


def _persist(per_query: pl.DataFrame, system: str, mode: str) -> None:
    PER_QUERY.mkdir(parents=True, exist_ok=True)
    per_query.write_parquet(PER_QUERY / f"{system}__{mode}.parquet")


def _append_result(summary: Summary) -> None:
    """Append one line to results.jsonl"""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary.row()) + "\n")


def evaluate_rerank(
    ranker: Ranker,
    judgements: pl.DataFrame,
    k: int = 10,
) -> Summary:
    """Mode A -- leaderboard-comparable reranking of the judged set."""
    queries = dict(zip(judgements["query_id"], judgements["query"], strict=True))
    candidates: dict[int, list[str]] = {}
    for qid, pid in zip(judgements["query_id"], judgements["product_id"], strict=True):
        candidates.setdefault(qid, []).append(pid)

    # ranked = ranker.rank_candidates(queries, candidates)
    ranked: dict[int, list[str]] = {}

    for qid in tqdm(queries, desc="Mode A: Reranking", unit="query"):
        ranked.update(
            ranker.rank_candidates(
                {qid: queries[qid]},
                {qid: candidates[qid]},
            )
        )
    per_query = score_rerank(ranked, judgements, k=k)

    summary = summarise(per_query, ranker.name, "rerank")
    _persist(per_query, ranker.name, "rerank")
    _append_result(summary)
    return summary


def evaluate_retrieval(
    ranker: Ranker,
    judgements: pl.DataFrame,
    k: int = 100,
) -> Summary:
    """Mode B -- full-corpus retrieval. Recall is a lower bound."""
    queries = dict(zip(judgements["query_id"], judgements["query"], strict=True))
    # retrieved = ranker.retrieve(queries, k=k)
    retrieved: dict[int, list[str]] = {}

    for qid in tqdm(queries, desc="Mode B: Retrieval", unit="query"):
        retrieved.update(ranker.retrieve({qid: queries[qid]}, k=k))
    per_query = score_retrieval(retrieved, judgements, k=k)

    summary = summarise(per_query, ranker.name, "retrieval")
    _persist(per_query, ranker.name, "retrieval")
    _append_result(summary)
    return summary


def render_table() -> str:
    """Render results.jsonl as a markdown table -- latest run per system."""
    if not RESULTS.exists():
        return "_no results yet_"

    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        latest[(str(row["system"]), str(row["mode"]))] = row

    ordered = list(latest.values())
    columns = list(ordered[0].keys())

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in ordered]
    return "\n".join([header, divider, *body])
