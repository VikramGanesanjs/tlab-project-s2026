"""Multi-slice DINOv3 classification model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.load_dinov3 import FEATURE_CHOICES  # noqa: E402

AGGREGATOR_CHOICES = ("transformer", "mean")
ENCODER_TRAINING_CHOICES = ("frozen", "lora")
LORA_PARAMETER_NAMES = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")

def is_lora_parameter_name(name: str) -> bool:
    return any(parameter_name in name for parameter_name in LORA_PARAMETER_NAMES)


def set_lora_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, parameter in model.named_parameters():
        if is_lora_parameter_name(name):
            parameter.requires_grad = requires_grad


class AttentionPooling(nn.Module):
    """Pool a token sequence using two-layer learned attention scores."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention_net(tokens)
        attention_weights = F.softmax(attention_scores, dim=1)
        return torch.sum(tokens * attention_weights, dim=1)


class MultiSliceDinoModel(nn.Module):
    """Frozen per-slice encoder followed by a trainable volume aggregator."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        n_slices: Optional[int],
        features: str = "cls",
        n_cls_tokens: int = 1,
        aggregator: str = "transformer",
        d_model: int = 768,
        depth: int = 2,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        num_classes: int = 1,
        encoder_training: str = "frozen",
    ) -> None:
        super().__init__()
        if features not in FEATURE_CHOICES:
            raise ValueError(f"Unknown features={features!r}; choose from {FEATURE_CHOICES}")
        if aggregator not in AGGREGATOR_CHOICES:
            raise ValueError(
                f"Unknown aggregator={aggregator!r}; choose from {AGGREGATOR_CHOICES}"
            )
        if n_slices is not None and n_slices <= 0:
            raise ValueError("n_slices must be positive when provided")
        if n_cls_tokens <= 0:
            raise ValueError("n_cls_tokens must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if aggregator == "transformer":
            if d_model % n_heads:
                raise ValueError("d_model must be divisible by positive n_heads")
            if depth <= 0:
                raise ValueError("depth must be positive")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if encoder_training not in ENCODER_TRAINING_CHOICES:
            raise ValueError(
                f"Unknown encoder_training={encoder_training!r}; "
                f"choose from {ENCODER_TRAINING_CHOICES}"
            )

        self.encoder = encoder
        self.n_slices = int(n_slices) if n_slices is not None else None
        self.features = features
        # Intermediate CLS tokens are meaningful only for the CLS feature mode.
        self.n_cls_tokens = int(n_cls_tokens) if features == "cls" else 1
        self.aggregator = aggregator
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.encoder_training = encoder_training
        if self.encoder_training == "frozen":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        embed_dim = int(encoder.embed_dim)
        pooling_hidden_dim = hidden_dim or embed_dim
        self.patch_pool = (
            AttentionPooling(embed_dim, pooling_hidden_dim)
            if features != "cls"
            else None
        )
        slice_feature_dim = embed_dim * self.n_cls_tokens
        self.slice_projection = (
            nn.Identity()
            if slice_feature_dim == d_model
            else nn.Linear(slice_feature_dim, d_model)
        )
        if aggregator == "transformer":
            self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
            # Fixed-depth runs retain their learned per-slice positional
            # embeddings.  Native-depth CQ500 runs intentionally omit them:
            # this lets the same transformer process any padded sequence
            # length, with ``slice_mask`` determining which slices exist.
            self.position_embedding = (
                nn.Parameter(torch.zeros(1, self.n_slices + 1, d_model))
                if self.n_slices is not None
                else None
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.global_token = None
            self.position_embedding = None
            self.transformer = None
        self.output_norm = nn.LayerNorm(d_model)
        classifier_hidden = hidden_dim or d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.Dropout(0.5),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Linear(classifier_hidden, self.num_classes),
        )
        if self.global_token is not None:
            nn.init.trunc_normal_(self.global_token, std=0.02)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode_slices(
        self, images: torch.Tensor, *, slice_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode ``[B, S, C, H, W]`` into ``[B, S, feature_dim]``.

        When a mask is supplied, only valid slices are sent through the
        encoder.  The returned tensor retains its padded shape so downstream
        aggregation can use the same mask, but its padded embeddings are zero.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected [B, S, C, H, W], got {tuple(images.shape)}")
        batch, n_slices = images.shape[:2]
        if self.n_slices is not None and n_slices != self.n_slices:
            raise ValueError(f"Expected {self.n_slices} slices, got {n_slices}")
        flat = images.reshape(batch * n_slices, *images.shape[2:])
        valid_mask: Optional[torch.Tensor] = None
        if slice_mask is not None:
            if slice_mask.shape != (batch, n_slices):
                raise ValueError(
                    "slice_mask must have shape [B, S], got "
                    f"{tuple(slice_mask.shape)} for images {tuple(images.shape)}"
                )
            valid_mask = slice_mask.to(device=images.device, dtype=torch.bool).reshape(-1)
            if not bool(valid_mask.any()):
                raise ValueError("At least one valid slice is required")
            flat = flat[valid_mask]
        context = (
            torch.enable_grad()
            if self.encoder_training == "lora" and self.training
            else torch.no_grad()
        )
        with context:
            if self.features == "cls" and self.n_cls_tokens > 1:
                intermediate = self.encoder.get_intermediate_layers(
                    flat,
                    n=self.n_cls_tokens,
                    return_class_token=True,
                )
            else:
                output = self.encoder.forward_features(flat)
        if self.features == "cls" and self.n_cls_tokens > 1:
            token = torch.cat(
                [
                    F.normalize(cls_token.float(), p=2, dim=-1)
                    for _, cls_token in intermediate
                ],
                dim=-1,
            )
        else:
            cls = F.normalize(output["x_norm_clstoken"].float(), p=2, dim=-1)
            if self.features == "cls":
                token = cls
            else:
                assert self.patch_pool is not None
                patches = F.normalize(output["x_norm_patchtokens"].float(), p=2, dim=-1)
                if self.features == "both":
                    # Attend over CLS + patch tokens together → single embed_dim vector.
                    tokens = torch.cat([cls.unsqueeze(1), patches], dim=1)
                else:
                    tokens = patches
                token = self.patch_pool(tokens)
        if valid_mask is not None:
            # Scatter only embeddings back into the padded layout; the DINO
            # forward pass above received ``valid_mask.sum()`` images, not B*S.
            padded_token = token.new_zeros((batch * n_slices, token.shape[-1]))
            padded_token[valid_mask] = token
            token = padded_token
        return token.reshape(batch, n_slices, -1)

    def extract_volume_token(
        self, images: torch.Tensor, *, slice_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return one MST volume representation for each input volume.

        For the transformer aggregator this is the output global token.  The
        mean-pool variant has no learned global token, so this instead returns
        its mean-pooled slice representation.  Keeping the aggregation here
        makes inference code able to use the exact representation consumed by
        the classifier without duplicating the MST forward pass.
        """
        slices = self.slice_projection(self.encode_slices(images, slice_mask=slice_mask))
        if self.aggregator == "mean":
            if slice_mask is not None:
                if slice_mask.shape != slices.shape[:2]:
                    raise ValueError(
                        "slice_mask must have shape [B, S], got "
                        f"{tuple(slice_mask.shape)} for slice tokens {tuple(slices.shape)}"
                    )
                weights = slice_mask.to(dtype=slices.dtype, device=slices.device)
                if not bool(weights.any(dim=1).all()):
                    raise ValueError("Every volume must contain at least one valid slice")
                return (slices * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
                    dim=1, keepdim=True
                )
            return slices.mean(dim=1)
        if self.global_token is None or self.transformer is None:
            raise RuntimeError("Transformer aggregator was not initialized")
        global_token = self.global_token.expand(slices.shape[0], -1, -1)
        sequence = torch.cat([global_token, slices], dim=1)
        if self.position_embedding is not None:
            sequence = sequence + self.position_embedding
        padding_mask = None
        if slice_mask is not None:
            if slice_mask.shape != slices.shape[:2]:
                raise ValueError(
                    "slice_mask must have shape [B, S], got "
                    f"{tuple(slice_mask.shape)} for slice tokens {tuple(slices.shape)}"
                )
            valid_slices = slice_mask.to(device=slices.device, dtype=torch.bool)
            if not bool(valid_slices.any(dim=1).all()):
                raise ValueError("Every volume must contain at least one valid slice")
            # The prepended global token is never padding.  Masking is
            # essential for native-depth batches, whose shorter volumes are
            # zero-padded by the collate function.
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        (slices.shape[0], 1), dtype=torch.bool, device=slices.device
                    ),
                    ~valid_slices,
                ],
                dim=1,
            )
        return self.transformer(sequence, src_key_padding_mask=padding_mask)[:, 0]

    def forward(
        self, images: torch.Tensor, *, slice_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        volume = self.extract_volume_token(images, slice_mask=slice_mask)
        logits = self.classifier(self.output_norm(volume))
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    @staticmethod
    def _is_lora_state_name(name: str) -> bool:
        return name.startswith("encoder.") and is_lora_parameter_name(name)

    @staticmethod
    def _is_trainable_state_name(name: str) -> bool:
        return not name.startswith("encoder.") or MultiSliceDinoModel._is_lora_state_name(
            name
        )

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """State for trainable MST components and optional encoder LoRA adapters."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if self._is_trainable_state_name(name)
        }

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        non_encoder_missing = [
            name
            for name in result.missing_keys
            if self._is_trainable_state_name(name)
        ]
        if unexpected or non_encoder_missing:
            raise RuntimeError(
                f"Invalid MST state: missing={non_encoder_missing}, unexpected={unexpected}"
            )
