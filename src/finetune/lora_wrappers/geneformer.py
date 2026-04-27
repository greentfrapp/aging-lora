"""Geneformer LoRA wrapper: BertModel + linear regression head + peft LoRA.

Geneformer V2 (104M) architecture:
    - BertForMaskedLM checkpoint, 12 layers, hidden=768, vocab=20275
    - We strip the MLM head, add a 1-layer regression head over <cls> embedding.
    - LoRA targets: query, value, intermediate.dense, output.dense per kickoff doc.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BertModel


GENEFORMER_CKPT = Path("save/Geneformer/Geneformer-V2-104M")


class GeneformerRegressor(nn.Module):
    """Geneformer + 1-layer regression head over a pooled cell embedding."""

    def __init__(
        self,
        ckpt_path: Path = GENEFORMER_CKPT,
        dropout: float = 0.1,
        bias_init: float = 0.0,
        pool: str = "cls",
    ):
        super().__init__()
        self.backbone = BertModel.from_pretrained(str(ckpt_path), add_pooling_layer=False)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, bias_init)
        if pool not in {"cls", "mean"}:
            raise ValueError(f"unknown pool={pool!r}; expected 'cls' or 'mean'")
        self.pool = pool

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, T, H]
        if self.pool == "cls":
            feat = h[:, 0]
        else:  # "mean": average over attended (non-pad) positions
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            feat = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.head(self.dropout(feat)).squeeze(-1)


def build_geneformer_lora(
    ckpt_path: Path = GENEFORMER_CKPT,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    head_bias_init: float = 0.0,
    pool: str = "cls",
):
    """Construct GeneformerRegressor and wrap backbone with LoRA adapters.

    The regression head is left fully trainable; backbone is frozen except for
    LoRA delta-weights on attention + MLP projections. Gradient checkpointing
    is enabled by default to fit 2048-token sequences on an 11 GB 2080 Ti.

    head_bias_init should be set to mean(y_train) so the model starts at the
    optimal mean predictor; otherwise AdamW + weight decay cannot grow the
    bias from 0 to mean(age) within a few-epoch fine-tune budget.

    pool: "cls" (default, back-compat) or "mean" (E5a ablation — mean over
    attended positions, motivated by Run #2 / v2 cls-only being stuck at
    "predict mean(train)" prediction floor; see notes/phase3_geneformer_convergence.md).
    """
    model = GeneformerRegressor(ckpt_path=ckpt_path, bias_init=head_bias_init, pool=pool)

    # Freeze the entire backbone first; LoRA will inject trainable adapters.
    for p in model.backbone.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["query", "value", "intermediate.dense", "output.dense"],
    )
    model.backbone = get_peft_model(model.backbone, cfg)

    if gradient_checkpointing:
        # peft-wrapped models expose enable_input_require_grads() to keep
        # checkpointing compatible with frozen base weights + LoRA adapters.
        model.backbone.enable_input_require_grads()
        # gradient_checkpointing_enable lives on the underlying BertModel;
        # peft passes through __getattr__.
        try:
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.backbone.gradient_checkpointing_enable()

    # Head stays trainable.
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def trainable_param_summary(model: nn.Module) -> dict:
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(n_total),
        "trainable_params": int(n_train),
        "trainable_pct": round(100.0 * n_train / max(n_total, 1), 4),
    }
