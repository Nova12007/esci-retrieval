"""The training loop"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from esci.data.collate import QueryCollator, build_relevance_mask
from esci.data.pairs import QueryExample
from esci.models.biencoder import BiEncoder
from esci.training.losses import info_nce


class RunLogger(Protocol):
    """The slice of a wandb Run the loop uses, so wandb stays optional."""

    def log(self, data: dict[str, Any], commit: bool | None = None) -> None: ...


class ExampleDataset(Dataset[QueryExample]):
    def __init__(self, examples: list[QueryExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> QueryExample:
        return self.examples[i]


@dataclass
class TrainConfig:
    lr: float = 2e-5
    epochs: int = 2
    batch_size: int = 16  # 6 GB: 16 queries x (1 q + 1 pos + 4 neg) = 96 seqs
    accum_steps: int = 4  # effective optimiser batch 64
    warmup_frac: float = 0.1
    temperature: float = 0.05
    max_grad_norm: float = 1.0
    log_every: int = 50
    ckpt_every: int = 2000
    out_dir: Path = Path("artifacts/biencoder_v2")


def train(
    encoder: BiEncoder,
    examples: list[QueryExample],
    collator: QueryCollator,
    cfg: TrainConfig,
    wandb_run: RunLogger | None = None,
    eval_fn: Callable[[BiEncoder, int], dict[str, float]] | None = None,
) -> Path:
    loader: DataLoader[QueryExample] = DataLoader(
        ExampleDataset(examples),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )

    optimiser = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = max(1, (len(loader) // cfg.accum_steps) * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_frac)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    encoder.cfg.save(cfg.out_dir / "encoder_config.json")

    step = 0
    started = time.perf_counter()

    for epoch in range(cfg.epochs):
        collator.set_epoch(epoch)  # different positive sampled each epoch
        encoder.train()

        for i, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = encoder(batch["queries"], is_query=True)  # (B, D)
                p = encoder(batch["positive_texts"], is_query=False)  # (M, D)

                flat = [t for row in batch["negative_texts"] for t in row]
                n_per = len(batch["negative_texts"][0])
                n_emb = encoder(flat, is_query=False).view(q.size(0), n_per, -1)

                mask = build_relevance_mask(
                    batch["query_ids"],
                    batch["positive_products"],
                    batch["positive_owner"],
                    collator.relevant_by_query,
                    device=str(q.device),
                )

                loss = (
                    info_nce(
                        query=q,
                        positives=p,
                        positive_owner=batch["positive_owner"].to(q.device),
                        negatives=n_emb,
                        relevance_mask=mask,
                        negative_mask=batch["negative_valid"].to(q.device),
                        temperature=cfg.temperature,
                    )
                    / cfg.accum_steps
                )

            loss.backward()  # type: ignore[no-untyped-call]

            if (i + 1) % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.max_grad_norm)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    metrics = {
                        "loss": loss.item() * cfg.accum_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "step": step,
                        "gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                        # if this is always 0, the mask is broken -- see The Diagonal, s7
                        "mask_rate": mask.float().mean().item(),
                        "neg_valid_rate": batch["negative_valid"].float().mean().item(),
                    }
                    wandb_run.log(metrics) if wandb_run else print(metrics)

                if step % cfg.ckpt_every == 0:
                    torch.save(encoder.model.state_dict(), cfg.out_dir / f"step_{step}.pt")

        if eval_fn is not None:
            dev = eval_fn(encoder, epoch)
            print(f"[epoch {epoch}] {dev}")
            if wandb_run:
                wandb_run.log({f"dev/{k}": v for k, v in dev.items()}, commit=False)

    torch.save(encoder.model.state_dict(), cfg.out_dir / "final.pt")
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")
    return cfg.out_dir / "final.pt"
