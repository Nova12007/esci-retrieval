"""Train the bi-encoder.

    # 1. overfit 100 queries -- loss MUST collapse to ~0
    uv run python scripts/train_biencoder.py --limit 100 --epochs 40 \
        --lr 1e-4 --accum 1 --no-dev-eval --no-wandb

    # 2. smoke run
    uv run python scripts/train_biencoder.py --limit 10000 --epochs 1

    # 3. the real run
    uv run python scripts/train_biencoder.py --epochs 2 | tee docs/train_v2.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch

from esci.data.collate import QueryCollator
from esci.data.pairs import build_query_examples, build_relevance_index
from esci.data.splits import add_folds
from esci.eval.aggregate import score_rerank, summarise
from esci.models.biencoder import BiEncoder, EncoderConfig
from esci.training.loop import TrainConfig, train

JUDGEMENTS = Path("data/processed/judgements.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument(
        "--corpus",
        choices=["t", "tbb"],
        default="tbb",
        help="document text variant, matching run_bm25.py",
    )
    p.add_argument("--limit", type=int, default=None, help="cap on TRAIN QUERIES")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--n-positives", type=int, default=1)
    p.add_argument("--n-negatives", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument(
        "--mask-threshold",
        type=float,
        default=1.0,
        help="1.0 masks Exact only; 0.1 also masks Substitutes",
    )
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="artifacts/biencoder_v2")
    p.add_argument("--dev-eval-queries", type=int, default=2000)
    p.add_argument("--no-dev-eval", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", default="", help="run name suffix for W&B")
    return p.parse_args()


def make_dev_eval(encoder_cfg, dev: pl.DataFrame, text_by_id, n_queries: int):
    """Mode A reranking on a dev sample -- no corpus encode needed.

    Cheap enough to run every epoch (~1 min on a 4050), which is the
    difference between watching the model learn and training blind.
    """
    keep = dev["query_id"].unique().sort().head(n_queries)
    sample = dev.filter(pl.col("query_id").is_in(keep))

    queries = dict(zip(sample["query_id"], sample["query"], strict=True))
    candidates: dict[int, list[str]] = {}
    for qid, pid in zip(sample["query_id"], sample["product_id"], strict=True):
        candidates.setdefault(int(qid), []).append(pid)

    def run(encoder: BiEncoder, epoch: int) -> dict[str, float]:
        encoder.model.eval()
        qids = list(queries.keys())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            q_emb = encoder([queries[q] for q in qids], is_query=True).float()

            ranked: dict[int, list[str]] = {}
            for row, qid in enumerate(qids):
                pool = candidates[qid]
                d = encoder([text_by_id.get(p, "") for p in pool], is_query=False).float()
                order = torch.argsort(d @ q_emb[row], descending=True).tolist()
                ranked[qid] = [pool[i] for i in order]

        encoder.model.train()
        per_query = score_rerank(ranked, sample, k=10)
        return summarise(per_query, "dev", "rerank").metrics

    return run


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    judgements = add_folds(pl.read_parquet(JUDGEMENTS))
    train_df = judgements.filter(pl.col("fold") == "train")
    dev_df = judgements.filter(pl.col("fold") == "dev")

    examples, stats = build_query_examples(train_df)
    print("builder stats:", json.dumps(stats, indent=2))

    if args.limit:
        examples = examples[: args.limit]
        kept = {e.query_id for e in examples}
        train_df = train_df.filter(pl.col("query_id").is_in(list(kept)))
        print(f"limited to {len(examples):,} queries")

    relevant = build_relevance_index(train_df, threshold=args.mask_threshold)
    corpus = pl.read_parquet(f"data/processed/corpus_{args.corpus}.parquet")
    text_by_id = dict(zip(corpus["product_id"], corpus["text"], strict=True))

    collator = QueryCollator(
        text_by_id=text_by_id,
        relevant_by_query=relevant,
        n_positives=args.n_positives,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )

    enc_cfg = EncoderConfig(model_name=args.model)
    encoder = BiEncoder(enc_cfg, device="cuda")

    eval_fn = None
    if not args.no_dev_eval and dev_df.height:
        eval_fn = make_dev_eval(enc_cfg, dev_df, text_by_id, args.dev_eval_queries)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="esci-retrieval",
            name=f"biencoder{'-' + args.tag if args.tag else ''}",
            config={**vars(args), **stats},
        )

    cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        temperature=args.temperature,
        out_dir=Path(args.out),
    )

    ckpt = train(encoder, examples, collator, cfg, wandb_run=run, eval_fn=eval_fn)
    print("checkpoint:", ckpt)

    if run:
        run.finish()


if __name__ == "__main__":
    main()
