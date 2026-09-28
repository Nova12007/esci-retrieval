"""Contrastive losses"""

from __future__ import annotations

import torch


def info_nce(
    query: torch.Tensor,  # (B, D)  L2-normalised
    positives: torch.Tensor,  # (M, D)  shared block
    positive_owner: torch.Tensor,  # (M,)    long, row index
    negatives: torch.Tensor | None = None,  # (B, n, D) private block
    relevance_mask: torch.Tensor | None = None,  # (B, M)  True = drop
    negative_mask: torch.Tensor | None = None,  # (B, n)  True = REAL negative
    temperature: float = 0.05,
) -> torch.Tensor:
    b, m, device = query.size(0), positives.size(0), query.device

    shared = (query @ positives.t()) / temperature
    sibling = positive_owner.unsqueeze(0) == torch.arange(b, device=device).unsqueeze(1)
    drop = sibling if relevance_mask is None else (sibling | relevance_mask)
    neg_shared = shared.masked_fill(drop, float("-inf"))

    if negatives is not None and negatives.numel() > 0:
        private = torch.einsum("bd,bnd->bn", query, negatives) / temperature
        if negative_mask is not None:
            private = private.masked_fill(~negative_mask, float("-inf"))
        neg_all = torch.cat([neg_shared, private], dim=1)
    else:
        neg_all = neg_shared

    neg_lse = torch.logsumexp(neg_all, dim=1)

    pos_logit = shared[positive_owner, torch.arange(m, device=device)]
    denom = torch.logaddexp(pos_logit, neg_lse[positive_owner])
    per_positive = denom - pos_logit

    counts = torch.bincount(positive_owner, minlength=b).clamp(min=1)
    per_query = torch.zeros(b, device=device, dtype=per_positive.dtype)
    per_query = per_query.index_add(0, positive_owner, per_positive) / counts
    return per_query.mean()


def build_duplicate_mask(positive_ids: list[str], device: str) -> torch.Tensor:
    """True where two rows in the batch share the same positive product."""
    n = len(positive_ids)
    same = torch.tensor(
        [[positive_ids[i] == positive_ids[j] for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=device,
    )
    same.fill_diagonal_(False)
    return same
