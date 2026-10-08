"""Reciprocal rank fusion of BM25 and a dense retriever, evaluated in both modes.

    uv run python scripts/run_fusion.py --dense biencoder_v3 \
        --dense-checkpoint artifacts/biencoder_v3/final.pt

Each retriever contributes its top 100; RRF (k=60, untuned) fuses them and the
top 100 of the fused list is scored. Hybrid retrieval is worth a row because
the two fail differently: BM25 on paraphrase, dense on exact model numbers and
rare brands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.runner import evaluate_rerank, evaluate_retrieval
from esci.index.dense import DenseRanker
from esci.index.fusion import FusionRanker
from esci.index.lexical import BM25Ranker
from esci.models.biencoder import BiEncoder, EncoderConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dense", default="biencoder_v3", help="cached embeddings name")
    p.add_argument("--dense-checkpoint", default="artifacts/biencoder_v3/final.pt")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--fold", default="test")
    p.add_argument("--limit", type=int, default=None, help="evaluate on N queries only")
    args = p.parse_args()

    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    evalset = judgements.filter(pl.col("fold") == args.fold)
    if args.limit:
        keep = evalset["query_id"].unique().sort().head(args.limit)
        evalset = evalset.filter(pl.col("query_id").is_in(keep.implode()))
    print(f"eval fold '{args.fold}': {evalset['query_id'].n_unique():,} queries")

    t0 = time.perf_counter()
    bm25 = BM25Ranker(corpus["product_id"].to_list(), corpus["text"].to_list(), "bm25")
    print(f"bm25 index built in {time.perf_counter() - t0:.0f}s")

    ckpt = Path(args.dense_checkpoint)
    encoder = BiEncoder(EncoderConfig.load(ckpt.parent / "encoder_config.json"), "cuda")
    encoder.model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    stem = f"artifacts/embeddings/{args.dense}_{args.corpus}"
    ids = Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n")
    assert ids == corpus["product_id"].to_list(), "dense embeddings do not match corpus"
    dense = DenseRanker(
        ids,
        np.load(f"{stem}.npy"),
        lambda qs: encoder.encode(qs, is_query=True),
        args.dense,
        backend="torch",
        device="cuda",
    )

    fused = FusionRanker([bm25, dense], name=f"rrf_bm25_{args.dense}")
    print("\n--- Mode A: RRF over the judged set ---")
    print(evaluate_rerank(fused, evalset, k=10))
    print("\n--- Mode B: RRF of the two top-100s ---")
    print(evaluate_retrieval(fused, evalset, k=100))


if __name__ == "__main__":
    main()
