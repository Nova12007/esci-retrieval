"""Dump the queries where the dense model loses most to BM25, for reading by hand.

    uv run python scripts/failure_cases.py

For each of the N worst queries by Mode B Recall@100 (v3 minus BM25): the
query, the Exact products BM25 found that v3 missed, and v3's own top five
with their labels. Writes artifacts/failure_cases.md; the analysis written
from it lives in docs/failure_analysis.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from esci.data.splits import add_folds
from esci.eval.runner import PER_QUERY
from esci.index.dense import DenseRanker
from esci.index.lexical import BM25Ranker
from esci.models.biencoder import BiEncoder, EncoderConfig

OUT = Path("artifacts/failure_cases.md")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--dense", default="biencoder_v3")
    p.add_argument("--dense-checkpoint", default="artifacts/biencoder_v3/final.pt")
    args = p.parse_args()

    test = add_folds(pl.read_parquet("data/processed/judgements.parquet")).filter(
        pl.col("fold") == "test"
    )
    dense_scores = pl.read_parquet(PER_QUERY / f"{args.dense}__retrieval.parquet")
    bm25_scores = pl.read_parquet(PER_QUERY / "bm25_tbb__retrieval.parquet")
    worst = (
        dense_scores.select("query_id", pl.col("recall@100").alias("dense"))
        .join(bm25_scores.select("query_id", pl.col("recall@100").alias("bm25")), on="query_id")
        .with_columns((pl.col("dense") - pl.col("bm25")).alias("diff"))
        .drop_nulls()
        .sort("diff", "query_id")
        .head(args.n)
    )
    qids = worst["query_id"].to_list()
    rows = test.filter(pl.col("query_id").is_in(worst["query_id"].implode()))
    qtext = dict(zip(rows["query_id"], rows["query"], strict=True))
    label = {
        (q, p): lab for q, p, lab in rows.select("query_id", "product_id", "esci_label").iter_rows()
    }

    corpus = pl.read_parquet("data/processed/corpus_tbb.parquet")
    text = dict(zip(corpus["product_id"], corpus["text"], strict=True))
    bm25 = BM25Ranker(corpus["product_id"].to_list(), corpus["text"].to_list(), "bm25")
    ckpt = Path(args.dense_checkpoint)
    enc = BiEncoder(EncoderConfig.load(ckpt.parent / "encoder_config.json"), "cuda")
    enc.model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    stem = f"artifacts/embeddings/{args.dense}_tbb"
    dense = DenseRanker(
        Path(f"{stem}.ids.txt").read_text(encoding="utf-8").split("\n"),
        np.load(f"{stem}.npy"),
        lambda qs: enc.encode(qs, is_query=True),
        args.dense,
        backend="torch",
        device="cuda",
    )
    queries = {q: qtext[q] for q in qids}
    got_bm25 = bm25.retrieve(queries, 100)
    got_dense = dense.retrieve(queries, 100)

    def show(pid: str) -> str:
        return text.get(pid, "")[:140].replace("|", "/").replace("\n", " ")

    out = [f"# {args.n} queries where {args.dense} loses most to BM25 (Recall@100)\n"]
    for r in worst.iter_rows(named=True):
        q = r["query_id"]
        exact = rows.filter((pl.col("query_id") == q) & (pl.col("esci_label") == "E"))
        exact_ids = set(exact["product_id"])
        missed = [p for p in got_bm25[q] if p in exact_ids and p not in set(got_dense[q])]
        out.append(
            f"\n## {qtext[q]!r}  (qid {q}; {len(exact_ids)} Exact; "
            f"dense {r['dense']:.2f}, BM25 {r['bm25']:.2f})\n"
        )
        out.append("BM25 found, dense missed:")
        out += [f"- {show(p)}" for p in missed[:5]] or ["- (none listed)"]
        out.append("\nDense top 5:")
        out += [f"- [{label.get((q, p), 'unjudged')}] {show(p)}" for p in got_dense[q][:5]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
