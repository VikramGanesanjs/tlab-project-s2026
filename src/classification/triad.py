"""Triad 3-D Swin encoder and volume classifier.

Triad is a 3-D MRI-pretrained MONAI Swin Transformer. Its public QuickStart
constructs the same backbone used here and loads the SimMIM checkpoint with a
strict state-dict match. Unlike the multi-slice DINO classifier, this module
passes a complete volume through the encoder and globally average-pools only
the final Swin stage.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAD_CHECKPOINT = (
    _REPOSITORY_ROOT / "opt" / "triad" / "Triad-SwinB-SimMIM.pth"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _triad_swin_transformer(
    *, feature_size: int, drop_path_rate: float, use_checkpoint: bool
) -> nn.Module:
    """Create the exact MONAI Swin-B architecture used by Triad QuickStart."""
    try:
        from monai.networks.nets.swin_unetr import SwinTransformer
    except ImportError as exc:
        raise ImportError(
            "Triad requires MONAI's SwinTransformer. Install the project's MONAI "
            "dependency before using --encoder triad."
        ) from exc
    return SwinTransformer(
        in_chans=1,
        embed_dim=feature_size,
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
        use_v2=True,
    )


class TriadEncoder(nn.Module):
    """MRI-pretrained Triad Swin-B encoder with final-stage average pooling."""

    def __init__(
        self,
        *,
        checkpoint: Optional[Path] = DEFAULT_TRIAD_CHECKPOINT,
        feature_size: int = 48,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if feature_size <= 0:
            raise ValueError(f"feature_size must be positive, got {feature_size}")
        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1], got {drop_path_rate}"
            )
        self.swinViT = _triad_swin_transformer(
            feature_size=int(feature_size),
            drop_path_rate=float(drop_path_rate),
            use_checkpoint=bool(use_checkpoint),
        )
        # MONAI returns the patch embedding plus four encoder stages; the
        # final Triad Swin stage is 16x the initial feature width.
        self.embed_dim = int(feature_size) * 16
        if checkpoint is not None:
            self.load_pretrained(checkpoint)

    def load_pretrained(self, checkpoint: Path) -> None:
        """Strictly load a Triad QuickStart checkpoint into the Swin backbone."""
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Triad checkpoint does not exist: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"Expected a state dict in Triad checkpoint {path}")
        state = state.get("state_dict", state)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state_dict in Triad checkpoint {path}")
        prefix = "backbone.swinViT."
        cleaned = {
            str(name)[len(prefix):] if str(name).startswith(prefix) else str(name): value
            for name, value in state.items()
        }
        self.swinViT.load_state_dict(cleaned, strict=True)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 1, D, H, W]`` volumes into final-stage pooled vectors."""
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(
                "Triad expects one-channel volumes with shape [B, 1, D, H, W], "
                f"got {tuple(volume.shape)}"
            )
        hidden_states = self.swinViT(volume)
        if not isinstance(hidden_states, Sequence) or not hidden_states:
            raise RuntimeError("Triad SwinTransformer did not return encoder stages")
        final_stage = hidden_states[-1]
        return F.adaptive_avg_pool3d(final_stage, output_size=1).flatten(1)


class TriadVolumeClassifier(nn.Module):
    """Classify complete volumes with a Triad encoder and MST-equivalent head.

    The input is the existing classification dataset layout ``[B, D, C, H, W]``.
    It is converted to ``[B, C, D, H, W]``, optionally de-normalized from the
    ImageNet convention used by the current volume datasets, resized to the
    Triad input cube, and projected to Triad's single MRI channel.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        input_channels: int = 3,
        volume_size: int = 96,
        input_normalization: str = "imagenet",
        hidden_dim: Optional[int] = None,
        num_classes: int = 1,
        encoder_training: str = "frozen",
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if volume_size <= 0 or volume_size % 32:
            raise ValueError(
                "volume_size must be a positive multiple of 32 for Triad Swin, "
                f"got {volume_size}"
            )
        if input_normalization not in {"imagenet", "none"}:
            raise ValueError(
                "input_normalization must be 'imagenet' or 'none', got "
                f"{input_normalization!r}"
            )
        if encoder_training not in {"frozen", "finetune"}:
            raise ValueError(
                "Triad encoder_training must be 'frozen' or 'finetune', got "
                f"{encoder_training!r}"
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not hasattr(encoder, "embed_dim"):
            raise TypeError("Triad encoder must expose an embed_dim attribute")

        self.encoder = encoder
        self.input_channels = int(input_channels)
        self.volume_size = int(volume_size)
        self.input_normalization = input_normalization
        self.encoder_training = encoder_training
        self.num_classes = int(num_classes)
        self.input_projection: nn.Module
        if self.input_channels == 1:
            self.input_projection = nn.Identity()
        else:
            projection = nn.Conv3d(self.input_channels, 1, kernel_size=1, bias=False)
            nn.init.constant_(projection.weight, 1.0 / self.input_channels)
            self.input_projection = projection
        if self.encoder_training == "frozen":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        embed_dim = int(encoder.embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        classifier_hidden = int(hidden_dim or embed_dim)
        # Keep this head identical to MultiSliceDinoModel's classifier.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden),
            nn.Dropout(0.5),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def _denormalize_imagenet(self, volume: torch.Tensor) -> torch.Tensor:
        channels = volume.shape[1]
        mean = volume.new_tensor(
            tuple(IMAGENET_MEAN[index % 3] for index in range(channels))
        ).view(1, channels, 1, 1, 1)
        std = volume.new_tensor(
            tuple(IMAGENET_STD[index % 3] for index in range(channels))
        ).view(1, channels, 1, 1, 1)
        return volume * std + mean

    def _resize_volumes(
        self, volumes: torch.Tensor, slice_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        target = (self.volume_size, self.volume_size, self.volume_size)
        if slice_mask is None:
            return F.interpolate(volumes, size=target, mode="trilinear", align_corners=False)
        if slice_mask.shape != (volumes.shape[0], volumes.shape[2]):
            raise ValueError(
                "slice_mask must have shape [B, D], got "
                f"{tuple(slice_mask.shape)} for volumes {tuple(volumes.shape)}"
            )
        valid = slice_mask.to(device=volumes.device, dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every Triad volume must contain at least one valid slice")
        # Native-depth CQ500 batches are padded. Resize each valid volume
        # separately so artificial zero padding cannot change its MRI features.
        return torch.cat(
            [
                F.interpolate(
                    volumes[index : index + 1, :, : int(valid[index].sum())],
                    size=target,
                    mode="trilinear",
                    align_corners=False,
                )
                for index in range(volumes.shape[0])
            ],
            dim=0,
        )

    def _prepare_volume(
        self, images: torch.Tensor, slice_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                "Triad classifier expects dataset volumes shaped [B, D, C, H, W], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {images.shape[2]}"
            )
        volume = images.permute(0, 2, 1, 3, 4).to(torch.float32)
        if self.input_normalization == "imagenet":
            volume = self._denormalize_imagenet(volume)
        volume = self._resize_volumes(volume, slice_mask)
        return self.input_projection(volume)

    def extract_volume_token(
        self,
        images: torch.Tensor,
        *,
        slice_mask: Optional[torch.Tensor] = None,
        profiler: Optional[Any] = None,
    ) -> torch.Tensor:
        volume = self._prepare_volume(images, slice_mask)
        # A frozen encoder normally runs without gradients.  The optional
        # multi-channel input projection sits before it, however, so retain
        # autograd in training when that projection needs to learn how to map
        # DINO-style channels to Triad's single MRI input.
        projection_trainable = any(
            parameter.requires_grad for parameter in self.input_projection.parameters()
        )
        context = (
            torch.enable_grad()
            if self.training and (self.encoder_training == "finetune" or projection_trainable)
            else torch.no_grad()
        )
        with context, (
            profiler.stage("dino_forward_s") if profiler is not None else nullcontext()
        ):
            return self.encoder(volume)

    def forward(
        self,
        images: torch.Tensor,
        *,
        slice_mask: Optional[torch.Tensor] = None,
        profiler: Optional[Any] = None,
    ) -> torch.Tensor:
        token = self.extract_volume_token(
            images, slice_mask=slice_mask, profiler=profiler
        )
        with (
            profiler.stage("rest_network_forward_s")
            if profiler is not None
            else nullcontext()
        ):
            logits = self.classifier(self.output_norm(token))
        return logits.squeeze(-1) if self.num_classes == 1 else logits

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Persist the head/adapter and the backbone when Triad is fine-tuned."""
        if self.encoder_training == "finetune":
            return self.state_dict()
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("encoder.")
        }

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [
            name
            for name in result.missing_keys
            if self.encoder_training == "finetune" or not name.startswith("encoder.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Invalid Triad state: missing={missing}, unexpected={unexpected}"
            )


__all__ = [
    "DEFAULT_TRIAD_CHECKPOINT",
    "TriadEncoder",
    "TriadVolumeClassifier",
]
