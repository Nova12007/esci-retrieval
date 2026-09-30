"""Encode the corpus with a bi-encoder and evaluate it in both modes.

    # zero-shot bge-small, same 256-token doc cap as training
    uv run python scripts/run_dense.py --name bge_small_zeroshot

    # a fine-tuned checkpoint; encoder settings come from its encoder_config.json
    uv run python scripts/run_dense.py --name biencoder_v2 \
        --checkpoint artifacts/biencoder_v2/final.pt

Corpus embeddings are cached under artifacts/embeddings/<name>_<corpus>.npy.
Pass --reencode after retraining into the same name.
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
from esci.models.biencoder import BiEncoder, EncoderConfig

EMBEDDINGS = Path("artifacts/embeddings")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="system name in results.jsonl")
    p.add_argument("--checkpoint", default=None, help="final.pt; omit for zero-shot")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--fold", default="test")
    p.add_argument("--limit", type=int, default=None, help="evaluate on N queries only")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--reencode", action="store_true", help="ignore cached embeddings")
    return p.parse_args()


def load_encoder(checkpoint: str | None) -> BiEncoder:
    if checkpoint is None:
        return BiEncoder(EncoderConfig(), device="cuda")
    ckpt = Path(checkpoint)
    cfg = EncoderConfig.load(ckpt.parent / "encoder_config.json")
    encoder = BiEncoder(cfg, device="cuda")
    encoder.model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    return encoder


def corpus_embeddings(
    encoder: BiEncoder, corpus: pl.DataFrame, path: Path, batch_size: int, reencode: bool
) -> np.ndarray:
    ids_path = path.with_suffix(".ids.txt")
    ids = corpus["product_id"].to_list()

    if path.exists() and ids_path.exists() and not reencode:
        cached = ids_path.read_text(encoding="utf-8").split("\n")
        assert cached == ids, f"{ids_path} does not match the corpus; pass --reencode"
        print(f"cached embeddings: {path}")
        return np.load(path, mmap_mode="r")

    t0 = time.perf_counter()
    emb = encoder.encode_to_memmap(
        corpus["text"].to_list(), path, is_query=False, batch_size=batch_size
    )
    ids_path.write_text("\n".join(ids), encoding="utf-8")
    print(f"encoded {len(ids):,} products in {(time.perf_counter() - t0) / 60:.1f} min")
    return emb


def main() -> None:
    args = parse_args()

    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    evalset = judgements.filter(pl.col("fold") == args.fold)
    if args.limit:
        keep = evalset["query_id"].unique().sort().head(args.limit)
        evalset = evalset.filter(pl.col("query_id").is_in(keep))
    print(
        f"corpus: {corpus.height:,} products; eval fold '{args.fold}': "
        f"{evalset['query_id'].n_unique():,} queries"
    )

    encoder = load_encoder(args.checkpoint)
    print(f"encoder: {encoder.cfg}")

    emb = corpus_embeddings(
        encoder,
        corpus,
        EMBEDDINGS / f"{args.name}_{args.corpus}.npy",
        args.batch_size,
        args.reencode,
    )

    t0 = time.perf_counter()
    ranker = DenseRanker(
        product_ids=corpus["product_id"].to_list(),
        embeddings=emb,
        encode_queries=lambda qs: encoder.encode(qs, is_query=True),
        name=args.name,
    )
    print(f"flat index: {emb.nbytes / 1e9:.2f} GB, built in {time.perf_counter() - t0:.1f}s")

    print("\n--- Mode A: rerank judged set ---")
    print(evaluate_rerank(ranker, evalset, k=10))

    print("\n--- Mode B: full-corpus retrieval ---")
    t0 = time.perf_counter()
    print(evaluate_retrieval(ranker, evalset, k=100))
    n = evalset["query_id"].n_unique()
    print(f"Mode B: {n / (time.perf_counter() - t0):.0f} queries/s end to end (encode + search)")


if __name__ == "__main__":
    main()
