"""Cross-view decoder components for slice-pair SSL fine-tuning."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from .utils import DecoderBlock


class CrossViewDecoder(nn.Module):
    """Cross-attend one slice's student sequence to the other slice's sequence.

    Both input streams are student encoder outputs, including their CLS token.
    The encoder has already applied its positional treatment, so this decoder
    deliberately does not add a second positional encoding.
    """

    def __init__(self, cfg: Any, enc_embed_dim: Optional[int] = None) -> None:
        super().__init__()
        self.n_blocks = int(cfg.decoder.n_blocks)
        self.embed_dim = int(cfg.decoder.embed_dim or enc_embed_dim)
        self.enc_embed_dim = int(enc_embed_dim)
        self.n_heads = int(cfg.decoder.n_heads)
        self.mlp_ratio = float(cfg.decoder.mlp_ratio)

        if self.embed_dim == self.enc_embed_dim:
            self.decoder_embed = nn.Identity()
            self.context_embed = nn.Identity()
        else:
            self.decoder_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
            self.context_embed = nn.Linear(self.enc_embed_dim, self.embed_dim)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    self.embed_dim,
                    self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
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

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
    ) -> Tensor:
        if query_tokens.ndim != 3 or context_tokens.ndim != 3:
            raise ValueError("CVD expects query/context tokens shaped [B, tokens, dim]")
        if query_tokens.shape[0] != context_tokens.shape[0]:
            raise ValueError(
                "CVD query/context batches must align: "
                f"query={tuple(query_tokens.shape)}, context={tuple(context_tokens.shape)}"
            )

        query = self.decoder_embed(query_tokens)
        context = self.context_embed(context_tokens)
        for block in self.decoder:
            query, _ = block(query, context, None, None)
        return self.output_proj(self.dec_norm(query))
