"""Train the cross-encoder on (query, product, ESCI gain) pairs.

    # smoke: 500 queries, eval often
    uv run python scripts/train_crossencoder.py --limit-queries 500 \
        --eval-every 50 --out artifacts/ce_smoke

    # the real run, one epoch over 1.26M pairs
    uv run python scripts/train_crossencoder.py --out artifacts/crossencoder_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.aggregate import score_rerank, summarise
from esci.index.rerank import RerankRanker
from esci.models.crossencoder import CrossEncoder, CrossEncoderConfig
from esci.training.ce_loop import CETrainConfig, train_cross_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="run directory; never reuse one")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=5000, help="optimiser steps")
    p.add_argument("--dev-eval-queries", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    train = judgements.filter(pl.col("fold") == "train")
    if args.limit_queries:
        keep = train["query_id"].unique().sort().head(args.limit_queries)
        train = train.filter(pl.col("query_id").is_in(keep.implode()))
    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    text = dict(zip(corpus["product_id"], corpus["text"], strict=True))

    pairs = [
        (q, text.get(p, ""), float(g))
        for q, p, g in zip(train["query"], train["product_id"], train["gain"], strict=True)
    ]
    print(f"{len(pairs):,} training pairs from {train['query_id'].n_unique():,} queries")

    # Same dev slice as the bi-encoder runs: first N dev query_ids by id.
    dev = judgements.filter(pl.col("fold") == "dev")
    dev = dev.filter(
        pl.col("query_id").is_in(
            dev["query_id"].unique().sort().head(args.dev_eval_queries).implode()
        )
    )
    dev_q = dict(zip(dev["query_id"], dev["query"], strict=True))
    dev_c: dict[int, list[str]] = {}
    for qid, pid in zip(dev["query_id"], dev["product_id"], strict=True):
        dev_c.setdefault(qid, []).append(pid)

    model = CrossEncoder(CrossEncoderConfig(max_len=args.max_len), device="cuda")

    def eval_fn(m: CrossEncoder) -> dict[str, float]:
        ranker = RerankRanker(m.score, text, "dev")
        per_query = score_rerank(ranker.rank_candidates(dev_q, dev_c), dev, k=10)
        return summarise(per_query, "dev", "rerank").metrics

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="esci-retrieval", name=f"ce-{Path(args.out).name}", config=vars(args)
        )

    cfg = CETrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        seed=args.seed,
        out_dir=Path(args.out),
    )
    ckpt = train_cross_encoder(model, pairs, cfg, wandb_run=run, eval_fn=eval_fn)
    print("checkpoint:", ckpt)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
