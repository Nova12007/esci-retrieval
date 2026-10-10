"""Regression tests: the dense pipeline end to end on a committed fixture.

The fixture is 20 unsaturated test queries with every judged product
(scripts/make_regression_fixture.py). Scores are Mode A nDCG@10 on CPU, fp32.

Why pinned values and not a floor, measured on this fixture (nDCG@10):

    correct pipeline       0.758770
    no bge query prefix    0.768216   (+0.0094: the bug scores HIGHER here)
    mean instead of CLS    0.753921   (-0.0048)

A floor cannot catch a bug that raises the score on a small sample, and
would let a 0.005 drop through at any sensible margin. Both are the silent
failures this test exists for. The models never change and CPU fp32 is
deterministic up to last-digit BLAS differences between machines, so any
drift beyond the tolerance means the pipeline changed. Re-pin deliberately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from esci.eval.metrics import ndcg_at_k
from esci.models.biencoder import BiEncoder, EncoderConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "regression_queries.parquet"
V3 = Path("artifacts/biencoder_v3/final.pt")

ZERO_SHOT_PINNED = 0.758770
V3_PINNED = 0.772805
# Catches both measured bugs (0.0094 and 0.0048 away); wide enough for a
# near-tie flipping on another machine's floating-point arithmetic.
TOLERANCE = 0.003


def fixture_ndcg(cfg: EncoderConfig, checkpoint: Path | None = None) -> float:
    """Mean Mode A nDCG@10 over the fixture's queries, encoded on CPU."""
    torch.manual_seed(0)
    enc = BiEncoder(cfg, device="cpu")
    if checkpoint is not None:
        enc.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    fixture = pl.read_parquet(FIXTURE)
    scores = []
    for _, group in fixture.group_by("query_id", maintain_order=True):
        q = enc.encode([group["query"][0]], is_query=True)[0]
        d = enc.encode(group["text"].to_list(), is_query=False)
        order = np.argsort(-(d @ q), kind="stable")
        gains = group["gain"].to_list()
        value = ndcg_at_k([gains[i] for i in order], gains, 10)
        if value is not None:
            scores.append(value)
    return float(np.mean(scores))


@pytest.mark.regression
def test_zero_shot_pipeline_matches_pinned_value() -> None:
    assert fixture_ndcg(EncoderConfig()) == pytest.approx(ZERO_SHOT_PINNED, abs=TOLERANCE)


@pytest.mark.regression
@pytest.mark.skipif(not V3.exists(), reason="fine-tuned checkpoint is not in git")
def test_fine_tuned_v3_matches_pinned_value() -> None:
    assert fixture_ndcg(EncoderConfig(), V3) == pytest.approx(V3_PINNED, abs=TOLERANCE)
