"""ViT backbone with LoRA adapters on attention Q/K/V projections."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from utils.load_dinov3 import (
    DINOV3_REPO,
    DEFAULT_ENCODER_WEIGHTS,
    ENCODER_CHOICES,
    load_encoder,
)


class LoRA(nn.Module):
    """Low-rank update wrapper around a fused QKV linear layer.

    Forward computes ``qkv(x) + [Δq; Δk; Δv]`` where each Δ is ``B @ A @ x``.
    """

    def __init__(
        self,
        qkv: nn.Module,
        w_a_q: nn.Linear,
        w_b_q: nn.Linear,
        w_a_k: nn.Linear,
        w_b_k: nn.Linear,
        w_a_v: nn.Linear,
        w_b_v: nn.Linear,
    ) -> None:
        super().__init__()
        self.qkv = qkv
        self.w_a_q = w_a_q
        self.w_b_q = w_b_q
        self.w_a_k = w_a_k
        self.w_b_k = w_b_k
        self.w_a_v = w_a_v
        self.w_b_v = w_b_v
        self.dim = qkv.in_features
        # Exposed so SelfAttention.compute_attention can read ``self.qkv.in_features``.
        self.in_features = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        new_q = self.w_b_q(self.w_a_q(x))
        new_k = self.w_b_k(self.w_a_k(x))
        new_v = self.w_b_v(self.w_a_v(x))
        qkv[..., : self.dim] += new_q
        qkv[..., self.dim : 2 * self.dim] += new_k
        qkv[..., -self.dim :] += new_v
        return qkv


def _make_lora_pair(dim: int, r: int) -> tuple[nn.Linear, nn.Linear]:
    w_a = nn.Linear(dim, r, bias=False)
    w_b = nn.Linear(r, dim, bias=False)
    nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
    nn.init.zeros_(w_b.weight)
    return w_a, w_b


def add_lora_to_vit(model: nn.Module, r: int = 4) -> nn.Module:
    """Replace each block's fused ``attn.qkv`` with a LoRA-augmented module."""
    if r <= 0:
        raise ValueError(f"LoRA rank r must be positive, got {r}")
    if not hasattr(model, "blocks"):
        raise AttributeError("Expected a ViT with ``model.blocks``")

    for blk in model.blocks:
        old_qkv = blk.attn.qkv
        dim = old_qkv.in_features
        device = old_qkv.weight.device
        dtype = old_qkv.weight.dtype

        w_a_q, w_b_q = _make_lora_pair(dim, r)
        w_a_k, w_b_k = _make_lora_pair(dim, r)
        w_a_v, w_b_v = _make_lora_pair(dim, r)

        lora_qkv = LoRA(old_qkv, w_a_q, w_b_q, w_a_k, w_b_k, w_a_v, w_b_v)
        lora_qkv.to(device=device, dtype=dtype)

        blk.attn.qkv = lora_qkv

    return model


def freeze_non_lora_parameters(model: nn.Module) -> nn.Module:
    """Freeze base weights; keep only LoRA A/B matrices trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = any(
            key in name
            for key in ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
        )
    return model


@torch.no_grad()
def init_lora_parameters(model: nn.Module) -> nn.Module:
    """Restore the zero-initialized LoRA update after backbone initialization."""
    for name, parameter in model.named_parameters():
        if "w_a_" in name:
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        elif "w_b_" in name:
            nn.init.zeros_(parameter)
    return model


def load_vit_with_lora(
    encoder: str = "dinov3",
    *,
    weights: Optional[Path] = None,
    repo_dir: Path = DINOV3_REPO,
    model_name: str = "dinov3_vitb16",
    device: Optional[torch.device] = None,
    r: int = 4,
    freeze_base: bool = True,
) -> nn.Module:
    """Load a pretrained ViT encoder and attach Q/K/V LoRA adapters.

    Parameters
    ----------
    encoder:
        One of ``dinov3 | meddinov3 | braindino | custom``. Defaults to
        the local released DINOv3 ViT-B checkpoint.
    weights:
        Checkpoint path. Defaults to the project encoder weight for ``encoder``.
    r:
        LoRA rank.
    freeze_base:
        If True, only LoRA matrices remain trainable.
    """
    name = encoder.strip().lower()
    if name not in ENCODER_CHOICES:
        raise ValueError(f"Unknown encoder={encoder!r}; choose from {ENCODER_CHOICES}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = Path(weights) if weights is not None else DEFAULT_ENCODER_WEIGHTS[name]
    model = load_encoder(
        name,
        device=device,
        weights=resolved,
        repo_dir=repo_dir,
        model_name=model_name,
    )
    add_lora_to_vit(model, r=r)
    if freeze_base:
        freeze_non_lora_parameters(model)
    return model


def lora_parameters(model: nn.Module):
    """Yield trainable LoRA parameters (for optimizers)."""
    for name, param in model.named_parameters():
        if param.requires_grad and any(
            key in name
            for key in ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")
        ):
            yield param


__all__ = [
    "LoRA",
    "add_lora_to_vit",
    "freeze_non_lora_parameters",
    "init_lora_parameters",
    "load_vit_with_lora",
    "lora_parameters",
]
