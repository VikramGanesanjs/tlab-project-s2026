"""Cross-view decoder components for slice-pair SSL fine-tuning."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .utils import DecoderBlock


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _DecoderRoPE(nn.Module):
    """Axial 2-D rotary position encoding for the CroCo-style decoder.

    ``DecoderBlock`` calls the rotary module with tensors shaped
    ``[B, heads, tokens, head_dim]`` and integer patch coordinates shaped
    ``[B, tokens, 2]``. The coordinate layout mirrors DINOv3's axial RoPE:
    half of the frequencies encode height and half encode width.
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("decoder embed_dim must be divisible by decoder n_heads")
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("decoder head dimension must be divisible by 4 for 2-D RoPE")
        self.head_dim = head_dim
        self.base = float(base)
        periods = base ** (
            2
            * torch.arange(head_dim // 4, dtype=torch.float32)
            / (head_dim // 2)
        )
        self.register_buffer("periods", periods, persistent=False)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        periods = self.base ** (
            2
            * torch.arange(
                self.head_dim // 4,
                device=self.periods.device,
                dtype=self.periods.dtype,
            )
            / (self.head_dim // 2)
        )
        self.periods.copy_(periods)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and x.shape[0] != 1:
            positions = positions.expand(x.shape[0], -1, -1)
        if positions.shape[0] != x.shape[0] or positions.shape[1] != x.shape[2]:
            raise ValueError(
                "RoPE positions must match decoder input: "
                f"x={tuple(x.shape)}, positions={tuple(positions.shape)}"
            )

        pos = positions.to(device=x.device, dtype=self.periods.dtype)
        periods = self.periods.to(device=x.device)
        angles = 2.0 * math.pi * pos[..., :, None] / periods[None, None, None, :]
        # [B, N, 2, D/4] -> [B, N, D/2], then duplicate for rotate_half.
        angles = angles.flatten(-2, -1)
        angles = torch.cat((angles, angles), dim=-1)
        sin = angles.sin().unsqueeze(1)
        cos = angles.cos().unsqueeze(1)
        dtype = x.dtype
        x_float = x.float()
        rotated = (x_float * cos) + (_rotate_half(x_float) * sin)
        return rotated.to(dtype=dtype)


def _patch_positions(batch_size: int, num_patches: int, device: torch.device) -> Tensor:
    """Return row-major ``[B, num_patches, 2]`` patch coordinates."""
    height = int(math.sqrt(num_patches))
    while height > 1 and num_patches % height:
        height -= 1
    width = num_patches // height
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    positions = torch.stack((row_grid.flatten(), col_grid.flatten()), dim=-1)
    return positions.unsqueeze(0).expand(batch_size, -1, -1)


class CrossViewDecoder(nn.Module):
    """CroCo-style decoder for cross-slice masked-patch completion."""

    def __init__(self, cfg: Any, enc_embed_dim: Optional[int] = None) -> None:
        super().__init__()
        self.n_blocks = int(cfg.decoder.n_blocks)
        self.embed_dim = int(cfg.decoder.embed_dim or enc_embed_dim)
        self.enc_embed_dim = int(enc_embed_dim)
        self.n_heads = int(cfg.decoder.n_heads)
        self.mlp_ratio = float(cfg.decoder.mlp_ratio)
        rope_base = float(cfg.decoder.rope_base)
        self.context_mode = str(cfg.decoder.context_mode).lower()
        if self.context_mode not in {"masked", "full"}:
            raise ValueError(
                "decoder.context_mode must be 'masked' or 'full', "
                f"got {self.context_mode!r}"
            )

        self.decoder_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
        self.context_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
        self.rope = _DecoderRoPE(self.embed_dim, self.n_heads, base=rope_base)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    self.embed_dim,
                    self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    rope=self.rope,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.dec_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = (
            nn.Identity()
            if self.embed_dim == self.enc_embed_dim
            else nn.Linear(self.embed_dim, self.enc_embed_dim)
        )

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the decoder after meta-device materialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.rope.reset_parameters()

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_pos: Optional[Tensor] = None,
        context_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if query_tokens.ndim != 3 or context_tokens.ndim != 3:
            raise ValueError("CVD expects query/context tokens shaped [B, patches, dim]")
        if query_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "CVD query/context batches must align: "
                f"query={tuple(query_tokens.shape)}, context={tuple(context_tokens.shape)}"
            )
        batch_size, query_patches, _ = query_tokens.shape
        context_patches = context_tokens.shape[1]
        if query_pos is None:
            query_pos = _patch_positions(batch_size, query_patches, query_tokens.device)
        if context_pos is None:
            context_pos = _patch_positions(batch_size, context_patches, context_tokens.device)

        query = self.decoder_embed(query_tokens)
        context = self.context_embed(context_tokens.detach())
        for block in self.decoder:
            query, _ = block(query, context, query_pos, context_pos)
        return self.output_proj(self.dec_norm(query))
