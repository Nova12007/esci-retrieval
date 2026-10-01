"""Hard negatives mined from the model's own retrieval (ANCE-style).

The model's top-ranked non-relevant products are its current mistakes, so they
make the most informative negatives. The risk is that many are relevant but
unjudged: ESCI judges ~16 products per query, so a highly ranked unjudged
product is often a correct answer. Training against it teaches the model that
right answers are wrong. Three guards, applied in order:

1. drop anything judged relevant to the query (the training mask's threshold);
2. skip the top ranks entirely, where unjudged positives concentrate;
3. sample from the remaining depth instead of taking the hardest.
"""

from __future__ import annotations

import random
from dataclasses import replace
from itertools import chain, zip_longest
from typing import Literal

from esci.data.pairs import QueryExample

NegativeSource = Literal["esci", "mined", "mixed"]


def filter_mined(
    ranked: list[str],
    relevant: frozenset[str],
    query_id: int,
    skip_top: int = 3,
    keep: int = 12,
    seed: int = 13,
) -> list[str]:
    """Apply the three guards to one query's retrieved list, best first.

    Returns up to `keep` products sampled from ranks [skip_top, len(ranked)),
    in rank order, excluding anything in `relevant`. Seeded per query, so a
    re-run mines the same negatives.
    """
    eligible = [p for p in ranked[skip_top:] if p not in relevant]
    if len(eligible) <= keep:
        return eligible
    rng = random.Random(seed * 1_000_003 + query_id)
    chosen = set(rng.sample(range(len(eligible)), keep))
    return [p for i, p in enumerate(eligible) if i in chosen]


def apply_negatives(
    examples: list[QueryExample],
    mined: dict[int, list[str]],
    source: NegativeSource,
) -> list[QueryExample]:
    """Set each example's negative pool from the chosen source.

    esci:  the judged S, C, I products (unchanged)
    mined: the model's filtered mistakes; a query with none mined trains on
           in-batch negatives only
    mixed: the two interleaved, so the collator's front-of-pool sample
           draws from both
    """
    if source == "esci":
        return examples
    out = []
    for ex in examples:
        mined_ids = mined.get(ex.query_id, [])
        if source == "mined":
            negatives = tuple(mined_ids)
        else:
            pairs = zip_longest(ex.negative_ids, mined_ids)
            negatives = tuple(dict.fromkeys(p for p in chain.from_iterable(pairs) if p))
        out.append(replace(ex, negative_ids=negatives))
    return out
