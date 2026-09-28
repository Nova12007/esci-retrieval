import argparse
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CORPUS_PATH = Path("data/processed/corpus_tbb.parquet")
QUERIES_PATH = Path("data/processed/judgements.parquet")

TEXT_COL = "text"
QUERY_COL = "query"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DOC_LENGTHS = [64, 96, 128, 160, 192, 256, 384, 512]
QUERY_LENGTHS = [16, 24, 32, 48, 64, 96]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_token_lengths(
    texts: list[str],
    prefix: str = "",
    desc: str = "Tokenizing",
) -> np.ndarray:
    """Count tokens including special tokens and prefix."""

    lengths = []

    for text in tqdm(texts, desc=desc, unit="text"):
        encoded = tokenizer(
            prefix + str(text),
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.append(len(encoded["input_ids"]))

    return np.array(lengths)


def print_statistics(
    name: str,
    lengths: np.ndarray,
    candidate_lengths: list[int],
) -> None:

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    percentiles = [50, 75, 90, 95, 99, 99.5, 100]

    for p, value in zip(
        percentiles,
        np.percentile(lengths, percentiles),
        strict=False,
    ):
        print(f"{p:5.1f}th percentile: {value:.0f} tokens")

    print(f"\n{'Max length':>12} {'Truncated':>12} {'Truncation %':>15}")

    for max_len in candidate_lengths:
        truncated = np.sum(lengths > max_len)
        percentage = 100 * truncated / len(lengths)

        print(f"{max_len:12d} {truncated:12d} {percentage:14.2f}%")


def analyze_documents():
    df = pl.read_parquet(CORPUS_PATH)

    texts = df[TEXT_COL].fill_null("").cast(pl.String).to_list()

    lengths = get_token_lengths(
        texts,
        desc="Tokenizing documents",
    )

    print(f"Number of documents: {len(texts)}")

    print_statistics(
        "DOCUMENT TOKEN LENGTHS",
        lengths,
        DOC_LENGTHS,
    )


def analyze_queries():
    df = pl.read_parquet(QUERIES_PATH)

    queries = df[QUERY_COL].fill_null("").cast(pl.String).to_list()

    lengths = get_token_lengths(
        queries,
        prefix=QUERY_PREFIX,
        desc="Tokenizing queries",
    )

    print(f"Number of queries: {len(queries)}")

    print_statistics(
        "QUERY TOKEN LENGTHS (INCLUDING BGE PREFIX)",
        lengths,
        QUERY_LENGTHS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        choices=["documents", "queries", "both"],
        default="both",
        help="What to analyze",
    )

    args = parser.parse_args()

    if args.type in ["documents", "both"]:
        analyze_documents()

    if args.type in ["queries", "both"]:
        analyze_queries()
