"""Simulated team-draft interleaving between retrieval systems.

    uv run python scripts/run_interleaving.py

Methodology demonstration: clicks are simulated from ESCI labels with assumed
probabilities (eval/interleave.py), so the output says how interleaving would
behave, not how users do. On a seeded sample of test queries it interleaves
each pair's top 10 and reports the preference next to the offline nDCG@10
difference on the same queries, to check the two agree in direction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.aggregate import score_retrieval
from esci.eval.interleave import interleave
from esci.index.dense import DenseRanker
from esci.index.fusion import FusionRanker
from esci.index.lexical import BM25Ranker
from esci.index.rerank import RerankRanker
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.models.crossencoder import CrossEncoder, CrossEncoderConfig

OUT = Path("benchmarks/interleaving.md")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--sessions", type=int, default=5, help="simulated sessions per query")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    qids = test["query_id"].unique().sort().sample(args.queries, seed=args.seed)
    sample = test.filter(pl.col("query_id").is_in(qids.implode()))
    queries = dict(zip(sample["query_id"], sample["query"], strict=True))
    labels: dict[int, dict[str, str]] = {}
    for q, pid, lab in sample.select("query_id", "product_id", "esci_label").iter_rows():
        labels.setdefault(q, {})[pid] = lab

    corpus = pl.read_parquet("data/processed/corpus_tbb.parquet")
    text = dict(zip(corpus["product_id"], corpus["text"], strict=True))
    bm25 = BM25Ranker(corpus["product_id"].to_list(), corpus["text"].to_list(), "bm25")
    enc = BiEncoder(EncoderConfig.load(Path("artifacts/biencoder_v3/encoder_config.json")), "cuda")
    enc.model.load_state_dict(torch.load("artifacts/biencoder_v3/final.pt", map_location="cuda"))
    stem = "artifacts/embeddings/biencoder_v3_tbb"
    dense = DenseRanker(
        Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n"),
        np.load(f"{stem}.npy"),
        lambda qs: enc.encode(qs, is_query=True),
        "biencoder_v3",
        backend="torch",
        device="cuda",
    )
    rrf = FusionRanker([bm25, dense], "rrf")
    ce = CrossEncoder(CrossEncoderConfig.load(Path("artifacts/crossencoder_v1/ce_config.json")))
    ce.model.load_state_dict(torch.load("artifacts/crossencoder_v1/final.pt", map_location="cuda"))
    rrf_ce = RerankRanker(ce.score, text, "rrf_ce", first_stage=rrf, depth=100)

    runs = {
        "bm25": bm25.retrieve(queries, 100),
        "biencoder_v3": dense.retrieve(queries, 100),
        "rrf": rrf.retrieve(queries, 100),
        "rrf_ce": rrf_ce.retrieve(queries, 100),
    }
    offline = {
        name: dict(score_retrieval(run, sample, k=100).select("query_id", "ndcg@10").iter_rows())
        for name, run in runs.items()
    }

    lines = [
        f"Simulated team-draft interleaving, {args.queries:,} test queries x "
        f"{args.sessions} sessions, top 10. Clicks are simulated from ESCI labels "
        "(eval/interleave.py): a methodology demonstration, not user evidence.\n",
        "| A | B | B wins | A wins | ties | preference for B [95% CI] "
        "| offline nDCG@10 B - A | agree |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a, b in [("bm25", "biencoder_v3"), ("biencoder_v3", "rrf"), ("biencoder_v3", "rrf_ce")]:
        r = interleave(
            {q: v[:10] for q, v in runs[a].items()},
            {q: v[:10] for q, v in runs[b].items()},
            labels,
            sessions_per_query=args.sessions,
            seed=args.seed,
        )
        decided = r.wins_a + r.wins_b
        half = 1.96 * math.sqrt(max(1e-12, 1 - r.preference_b**2) / max(1, decided))
        diffs = [
            offline[b][q] - offline[a][q]
            for q in offline[a]
            if offline[a][q] is not None and offline[b].get(q) is not None
        ]
        delta = float(np.mean(diffs))
        agree = "yes" if np.sign(delta) == np.sign(r.preference_b) else "NO"
        lines.append(
            f"| {a} | {b} | {r.wins_b:,} | {r.wins_a:,} | {r.ties:,} | "
            f"{r.preference_b:+.3f} [{r.preference_b - half:+.3f}, {r.preference_b + half:+.3f}] "
            f"| {delta:+.4f} | {agree} |"
        )
        print(json.dumps({"a": a, "b": b, **r.__dict__, "offline_delta": delta}))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
