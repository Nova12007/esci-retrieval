"""Score every judged training pair with the cross-encoder, for distillation.

    uv run python scripts/score_teacher.py \
        --ce-checkpoint artifacts/crossencoder_v1/final.pt

Writes artifacts/teacher/<name>.parquet (query_id, product_id, score): one
logit per judged (query, product) pair on the train fold. The student learns
from differences between these scores within a query (margin-MSE), so their
absolute scale never matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from esci.data.splits import add_folds
from esci.models.crossencoder import CrossEncoder, CrossEncoderConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ce-checkpoint", required=True)
    p.add_argument("--name", default="ce_v1_train")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit-queries", type=int, default=None, help="smoke runs")
    args = p.parse_args()

    train = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "train"
    )
    if args.limit_queries:
        keep = train["query_id"].unique().sort().head(args.limit_queries)
        train = train.filter(pl.col("query_id").is_in(keep.implode()))
    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    text = dict(zip(corpus["product_id"], corpus["text"], strict=True))

    queries = train["query"].to_list()
    docs = [text.get(pid, "") for pid in train["product_id"]]
    # Longest first: batches pad to their longest member, so sorting by length
    # removes most padding, and an OOM would show in the first batch.
    order = np.argsort([-(len(q) + len(d)) for q, d in zip(queries, docs, strict=True)])

    ckpt = Path(args.ce_checkpoint)
    ce = CrossEncoder(CrossEncoderConfig.load(ckpt.parent / "ce_config.json"), device="cuda")
    ce.model.load_state_dict(torch.load(ckpt, map_location="cuda"))

    scores = np.empty(len(order), dtype=np.float32)
    for start in tqdm(range(0, len(order), args.batch_size), desc="teacher", unit="batch"):
        rows = order[start : start + args.batch_size]
        scores[rows] = ce.score(
            [queries[i] for i in rows], [docs[i] for i in rows], batch_size=args.batch_size
        )

    out = Path(f"artifacts/teacher/{args.name}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    train.select("query_id", "product_id", "esci_label").with_columns(
        pl.Series("score", scores)
    ).write_parquet(out)
    print(f"{len(scores):,} pairs -> {out}")
    by_label = (
        pl.read_parquet(out).group_by("esci_label").agg(pl.col("score").mean()).sort("esci_label")
    )
    print("mean teacher score by label:", dict(by_label.iter_rows()))


if __name__ == "__main__":
    main()
