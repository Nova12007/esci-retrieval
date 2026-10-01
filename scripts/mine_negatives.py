"""Mine hard negatives for every training query with a trained bi-encoder.

    uv run python scripts/mine_negatives.py --name biencoder_v3 \
        --checkpoint artifacts/biencoder_v3/final.pt

Needs the corpus embeddings run_dense.py cached for the same model. Writes
artifacts/mined/<name>.parquet (query_id, product_id, rank), consumed by
train_biencoder.py --negatives mined|mixed --mined <path>.
"""

from __future__ import annotations

import argparse
import json
from itertools import batched
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from esci.data.pairs import build_query_examples, build_relevance_index
from esci.data.splits import add_folds
from esci.index.dense import DenseRanker
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.training.mining import filter_mined


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="model whose cached embeddings to search")
    p.add_argument("--checkpoint", required=True, help="the same model, to encode queries")
    p.add_argument("--corpus", choices=["t", "tbb"], default="tbb")
    p.add_argument("--depth", type=int, default=100, help="retrieve this many per query")
    p.add_argument("--skip-top", type=int, default=3, help="guard 2: ranks never used")
    p.add_argument("--keep", type=int, default=12, help="guard 3: sampled per query")
    p.add_argument("--mask-threshold", type=float, default=1.0, help="guard 1, as in training")
    p.add_argument("--out", default=None, help="default artifacts/mined/<name>.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out or f"artifacts/mined/{args.name}.parquet")

    judgements = add_folds(pl.read_parquet("data/processed/judgements.parquet"))
    train = judgements.filter(pl.col("fold") == "train")
    examples, _ = build_query_examples(train)
    relevant = build_relevance_index(train, threshold=args.mask_threshold)
    judged = {(q, p) for q, p in zip(train["query_id"], train["product_id"], strict=True)}

    stem = f"artifacts/embeddings/{args.name}_{args.corpus}"
    ids = Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n")
    emb = np.load(f"{stem}.npy")
    ckpt = Path(args.checkpoint)
    encoder = BiEncoder(EncoderConfig.load(ckpt.parent / "encoder_config.json"), "cuda")
    encoder.model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    ranker = DenseRanker(
        ids,
        emb,
        lambda qs: encoder.encode(qs, is_query=True),
        args.name,
        backend="torch",
        device="cuda",
    )

    rows: list[tuple[int, str, int]] = []
    top_unjudged = top_total = kept_unjudged = 0
    for chunk in tqdm(list(batched(examples, 256)), desc="mine", unit="chunk"):
        retrieved = ranker.retrieve({e.query_id: e.query for e in chunk}, k=args.depth)
        for ex in chunk:
            ranked = retrieved[ex.query_id]
            rel = relevant.get(ex.query_id, frozenset())
            for p in ranked[: args.skip_top]:  # what guard 2 throws away
                if p not in rel:
                    top_total += 1
                    top_unjudged += (ex.query_id, p) not in judged
            mined = filter_mined(ranked, rel, ex.query_id, args.skip_top, args.keep)
            kept_unjudged += sum((ex.query_id, p) not in judged for p in mined)
            rows.extend((ex.query_id, p, i) for i, p in enumerate(mined))

    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["query_id", "product_id", "rank"], orient="row").write_parquet(out)
    stats = {
        "queries": len(examples),
        "mined_negatives": len(rows),
        "per_query": round(len(rows) / len(examples), 2),
        "unjudged_share_kept": round(kept_unjudged / max(1, len(rows)), 4),
        "unjudged_share_in_skipped_top": round(top_unjudged / max(1, top_total), 4),
    }
    print(json.dumps(stats, indent=2))
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
