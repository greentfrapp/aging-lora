"""FM-agnostic LoRA training loop with bf16 autocast + per-donor eval.

Designed to be safe for the 2080 Ti: bf16 autocast on, gradient accumulation
configurable, no torch.compile (Turing/CC 7.5 doesn't benefit much and can
trigger long compile waits).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 5e-5
    head_lr: float = 1e-3   # higher LR for the regression head (separate param group)
    weight_decay: float = 0.01
    warmup_pct: float = 0.10
    grad_clip: float = 1.0
    bf16: bool = True
    eval_bf16: bool = False  # disable bf16 in eval to avoid output quantization
    log_every: int = 25
    eval_every: int = 1  # epochs
    num_workers: int = 0  # backed h5ad + Windows: keep 0 to avoid pickling issues
    seq_len: int = 2048
    max_train_steps: int | None = None  # for smoke


def linear_warmup_decay(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


@torch.no_grad()
def evaluate_per_donor(
    model: nn.Module, loader: DataLoader, device: torch.device, bf16: bool
) -> dict:
    model.eval()
    preds: list[float] = []
    ages: list[float] = []
    donors: list[str] = []
    autocast_dtype = torch.bfloat16 if bf16 else torch.float32
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=bf16):
            yhat = model(ids, mask)
        preds.extend(yhat.float().cpu().numpy().tolist())
        ages.extend(batch["age"].numpy().tolist())
        donors.extend(list(batch["donor"]))
    preds = np.asarray(preds)
    ages = np.asarray(ages)
    donors = np.asarray(donors)
    # Per-donor median aggregation (matches LASSO/Pasta convention).
    df = {}
    for d in np.unique(donors):
        m = donors == d
        df[d] = (float(np.median(preds[m])), float(ages[m][0]))
    pred_d = np.array([v[0] for v in df.values()])
    age_d = np.array([v[1] for v in df.values()])
    mae = float(np.median(np.abs(pred_d - age_d)))  # median absolute error
    if len(pred_d) >= 2 and pred_d.std() > 0 and age_d.std() > 0:
        r = float(np.corrcoef(pred_d, age_d)[0, 1])
    else:
        r = float("nan")
    return {
        "mae": mae,
        "pearson_r": r,
        "n_donors": int(len(df)),
        "per_donor_predictions": [
            {"donor": d, "pred_age": p, "true_age": a}
            for d, (p, a) in df.items()
        ],
    }


def _collate(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }
    if "age" in batch[0]:
        out["age"] = torch.stack([b["age"] for b in batch])
    if "donor" in batch[0]:
        out["donor"] = [b["donor"] for b in batch]
    return out


def train(
    model: nn.Module,
    train_ds,
    eval_ds,
    cfg: TrainConfig,
    device: torch.device,
    log_path: Path,
    ckpt_path: Path | None = None,
) -> dict:
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    head_params: list = []
    head_no_decay: list = []
    backbone_params: list = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head." in n:
            (head_no_decay if n.endswith(".bias") else head_params).append(p)
        else:
            backbone_params.append(p)
    param_groups = [
        {"params": backbone_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": head_no_decay, "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    optim = AdamW(param_groups)
    loss_fn = nn.MSELoss()
    trainable = backbone_params + head_params + head_no_decay

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_pct * total_steps))

    autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    history: list[dict] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", buffering=1)

    def log(d):
        log_f.write(json.dumps(d) + "\n")

    log({"event": "config", "config": asdict(cfg)})

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        running_n = 0
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            age = batch["age"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=cfg.bf16):
                yhat = model(ids, mask)
                loss = loss_fn(yhat, age) / cfg.grad_accum
            loss.backward()
            running += float(loss.item()) * cfg.grad_accum * len(age)
            running_n += len(age)

            if (step + 1) % cfg.grad_accum == 0:
                lr_scale = linear_warmup_decay(global_step, total_steps, warmup)
                base_lrs = [cfg.lr, cfg.head_lr, cfg.head_lr]
                for g, base in zip(optim.param_groups, base_lrs):
                    g["lr"] = base * lr_scale
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - t0
                    log(
                        {
                            "event": "step",
                            "epoch": epoch,
                            "step": global_step,
                            "lr_backbone": cfg.lr * lr_scale,
                            "lr_head": cfg.head_lr * lr_scale,
                            "train_mse_running": running / max(running_n, 1),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                    running = 0.0
                    running_n = 0

                if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                    log({"event": "max_steps_hit", "global_step": global_step})
                    break
        if cfg.max_train_steps and global_step >= cfg.max_train_steps:
            break

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate_per_donor(model, eval_loader, device, cfg.eval_bf16)
            metrics["epoch"] = epoch
            metrics["elapsed_s"] = round(time.time() - t0, 1)
            log({"event": "eval", **{k: v for k, v in metrics.items() if k != "per_donor_predictions"}})
            history.append(metrics)

    # Final eval — always run, regardless of max_steps / eval_every.
    # If the per-epoch eval already ran for the final epoch, this re-runs once
    # to keep the contract simple; cost is small relative to a multi-hour run.
    final_metrics = evaluate_per_donor(model, eval_loader, device, cfg.eval_bf16)
    final_metrics["epoch"] = "final"
    final_metrics["elapsed_s"] = round(time.time() - t0, 1)
    log({"event": "final_eval", **{k: v for k, v in final_metrics.items() if k != "per_donor_predictions"}})
    history.append(final_metrics)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # Save ONLY trainable (LoRA + head) state — keeps file small.
        sd = {k: v for k, v in model.state_dict().items() if any(
            (sub in k) for sub in ("lora_", "head.")
        )}
        torch.save(sd, ckpt_path)
        log({"event": "checkpoint", "path": str(ckpt_path), "n_params": len(sd)})

    log({"event": "done", "total_elapsed_s": round(time.time() - t0, 1)})
    log_f.close()
    return {"history": history, "total_elapsed_s": time.time() - t0}
