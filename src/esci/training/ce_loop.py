"""Training loop for the cross-encoder: pointwise, against graded ESCI gains.

Loss is binary cross-entropy with the gain as a soft target, not MSE on a
sigmoid output. MSE through a sigmoid has vanishing gradients exactly where
they are needed: a confidently wrong prediction sits on the flat tail and
barely moves. BCE's gradient with respect to the logit is (p - target), so
confident mistakes get the largest corrections.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from esci.models.crossencoder import CrossEncoder
from esci.training.loop import RunLogger

Pair = tuple[str, str, float]  # (query, product text, gain)


@dataclass
class CETrainConfig:
    lr: float = 2e-5
    epochs: int = 1
    batch_size: int = 64
    warmup_frac: float = 0.1
    max_grad_norm: float = 1.0
    log_every: int = 200
    eval_every: int = 5000
    seed: int = 13
    out_dir: Path = Path("artifacts/crossencoder")


def train_cross_encoder(
    model: CrossEncoder,
    pairs: list[Pair],
    cfg: CETrainConfig,
    wandb_run: RunLogger | None = None,
    eval_fn: Callable[[CrossEncoder], dict[str, float]] | None = None,
) -> Path:
    steps_per_epoch = len(pairs) // cfg.batch_size
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup = int(total_steps * cfg.warmup_frac)

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda s: (
            s / max(1, warmup)
            if s < warmup
            else max(0.0, 1.0 - (s - warmup) / max(1, total_steps - warmup))
        ),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model.cfg.save(cfg.out_dir / "ce_config.json")
    log_path = cfg.out_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)
    vram = torch.cuda.get_device_properties(0).total_memory
    pbar = tqdm(total=total_steps, desc="train", unit="batch", dynamic_ncols=True)

    def emit(record: dict[str, Any]) -> None:
        pbar.write(json.dumps(record))
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if wandb_run:
            wandb_run.log(record)

    def evaluate(step: int) -> None:
        if eval_fn is not None:
            emit({"step": step, **{f"dev/{k}": v for k, v in eval_fn(model).items()}})
            model.model.train()

    evaluate(0)  # step 0 = the MS MARCO model before any ESCI training
    rng = random.Random(cfg.seed)
    step, win_loss, win_n = 0, 0.0, 0
    started = time.perf_counter()

    for _epoch in range(cfg.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        model.model.train()
        for b in range(steps_per_epoch):
            batch = [pairs[i] for i in order[b * cfg.batch_size : (b + 1) * cfg.batch_size]]
            target = torch.tensor([g for _, _, g in batch], device=model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model([q for q, _, _ in batch], [d for _, d, _ in batch])
            loss = loss_fn(logits.float(), target)  # BCE in fp32 for stability
            loss.backward()

            if torch.cuda.max_memory_allocated() > 0.97 * vram:
                raise RuntimeError("peak memory past 97% of VRAM: lower --batch-size")

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1
            win_loss += loss.item()
            win_n += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.log_every == 0:
                emit(
                    {
                        "step": step,
                        "loss": win_loss / win_n,
                        "lr": scheduler.get_last_lr()[0],
                        "gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }
                )
                win_loss, win_n = 0.0, 0
            if step % cfg.eval_every == 0:
                evaluate(step)

    if step % cfg.eval_every:
        evaluate(step)
    pbar.close()
    torch.save(model.model.state_dict(), cfg.out_dir / "final.pt")
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")
    return cfg.out_dir / "final.pt"
