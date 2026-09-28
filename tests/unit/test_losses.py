from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from esci.training.losses import (
    build_duplicate_mask,
    info_nce,
)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def test_perfect_separation_gives_near_zero_loss() -> None:
    """Query identical to its positive, orthogonal to everything else."""
    q = _unit(torch.eye(4))
    pos = q.clone()

    owner = torch.arange(4)

    loss = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        temperature=0.05,
    )

    assert loss.item() < 0.01


def test_uninformative_embeddings_give_log_of_class_count() -> None:
    """All vectors identical: uniform probability over B candidates."""
    b, d = 8, 16

    q = _unit(torch.ones(b, d))
    pos = q.clone()

    owner = torch.arange(b)

    loss = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        temperature=0.05,
    )

    assert loss.item() == pytest.approx(math.log(b), abs=1e-4)


def test_hard_negatives_enlarge_the_class_count() -> None:
    """With N explicit negatives, uniform loss is log(B + N)."""
    b, n, d = 4, 3, 16

    q = _unit(torch.ones(b, d))
    pos = q.clone()
    neg = _unit(torch.ones(b, n, d))

    owner = torch.arange(b)

    loss = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        negatives=neg,
        temperature=0.05,
    )

    assert loss.item() == pytest.approx(math.log(b + n), abs=1e-4)


def test_duplicate_mask_never_masks_the_diagonal() -> None:
    mask = build_duplicate_mask(
        ["a", "a", "b"],
        device="cpu",
    )

    assert not mask[0][0]
    assert not mask[1][1]
    assert not mask[2][2]

    assert mask[0][1]
    assert mask[1][0]

    assert not mask[0][2]


def test_relevance_masking_lowers_the_loss() -> None:
    """A known-relevant cross-query product must not be a negative."""
    q = _unit(torch.eye(3))
    pos = q.clone()

    owner = torch.arange(3)

    # Q1's positive is also relevant to Q0.
    relevance_mask = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    unmasked = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        temperature=0.05,
    )

    masked = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        relevance_mask=relevance_mask,
        temperature=0.05,
    )

    assert masked.item() <= unmasked.item()


def test_multiple_positives_per_query() -> None:
    """Each query can have multiple positives, averaged within query."""
    _b, _d = 2, 4

    q = _unit(torch.eye(4)[:2])

    # Query 0 owns columns 0,1; query 1 owns columns 2,3.
    pos = _unit(torch.eye(4))

    owner = torch.tensor([0, 0, 1, 1])

    loss = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
        temperature=0.05,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_gradients_flow() -> None:
    q = _unit(torch.randn(4, 16)).requires_grad_(True)
    pos = _unit(torch.randn(4, 16))

    owner = torch.arange(4)

    loss = info_nce(
        query=q,
        positives=pos,
        positive_owner=owner,
    )

    loss.backward()

    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
