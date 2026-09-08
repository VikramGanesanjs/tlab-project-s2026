"""NeuroVFM whole-volume encoder and classification head.

NeuroVFM is kept as a local source checkout under ``opt/neurovfm``.  This
module imports that checkout at runtime and uses its public encoder pipeline;
it deliberately does not require NeuroVFM to be installed as a Python package.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEUROVFM_REPO = _REPOSITORY_ROOT / "opt" / "neurovfm"
DEFAULT_NEUROVFM_WEIGHTS = DEFAULT_NEUROVFM_REPO / "weights"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _neurovfm_components(repo: Path):
    """Import the local NeuroVFM pipeline without a package installation."""
    repo = Path(repo).expanduser().resolve()
    package_root = repo / "neurovfm"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"NeuroVFM source checkout does not exist at {package_root}"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from neurovfm.data.preprocess import tokenize_volume
        from neurovfm.pipelines.encoder import load_encoder
    except ImportError as exc:
        raise ImportError(
            "Could not import NeuroVFM from the local checkout. Its runtime "
            "dependencies (for example SimpleITK) must be available. "
            "FlashAttention is optional because the local checkout includes "
            "a native PyTorch inference fallback."
        ) from exc
    return load_encoder, tokenize_volume


class NeuroVFMEncoder(nn.Module):
    """Frozen local NeuroVFM pipeline that maps complete volumes to embeddings."""

    def __init__(
        self,
        *,
        repo: Path = DEFAULT_NEUROVFM_REPO,
        weights: Path = DEFAULT_NEUROVFM_WEIGHTS,
        device: torch.device,
        modality: str = "mri",
    ) -> None:
        super().__init__()
        if modality not in {"mri", "ct"}:
            raise ValueError(f"modality must be 'mri' or 'ct', got {modality!r}")
        weights = Path(weights).expanduser().resolve()
        if not weights.is_dir():
            raise FileNotFoundError(f"NeuroVFM weights directory does not exist: {weights}")
        load_encoder, self._tokenize_volume = _neurovfm_components(repo)
        # ``EncoderPipeline`` owns the canonical checkpoint construction and
        # normalization logic. Pass a device type because its autocast context
        # expects ``'cpu'`` or ``'cuda'``, rather than e.g. ``'cuda:0'``.
        self.pipeline, _ = load_encoder(str(weights), device=device.type)
        self.model = self.pipeline.model
        self.embed_dim = int(self.model.embed_dim)
        self.modality = modality
        self.device_type = device.type
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _batch_from_volumes(self, volumes: torch.Tensor) -> Dict[str, Any]:
        tokens, coords, lengths, sizes = [], [], [], []
        for index, volume in enumerate(volumes):
            array = volume.squeeze(0).detach().cpu().numpy().astype(np.float32)
            # NeuroVFM's tokenizer is used without background removal so each
            # complete input volume produces an embedding, including scans with
            # little foreground after a dataset-specific crop.
            token, coordinate, _ = self._tokenize_volume(
                array,
                np.ones_like(array, dtype=bool),
                remove_background=False,
            )
            tokens.append(torch.from_numpy(token).float())
            coords.append(torch.from_numpy(coordinate).long())
            lengths.append(len(token))
            sizes.append(tuple(int(size) for size in array.shape))
        if not lengths or min(lengths) <= 0:
            raise ValueError("NeuroVFM requires every volume to contain at least one patch")
        cumulative = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cumulative[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        paths = [
            f"classification_volume_{index}_BrainWindow"
            if self.modality == "ct"
            else f"classification_volume_{index}"
            for index in range(len(lengths))
        ]
        return {
            "img": torch.cat(tokens),
            "coords": torch.cat(coords),
            "series_masks_indices": torch.tensor([], dtype=torch.long),
            "series_cu_seqlens": cumulative,
            "series_max_len": max(lengths),
            "mode": [self.modality] * len(lengths),
            "path": paths,
            "size": sizes,
        }

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 1, D, H, W]`` volumes into one vector per volume."""
        if volumes.ndim != 5 or volumes.shape[1] != 1:
            raise ValueError(
                "NeuroVFM expects one-channel volumes [B, 1, D, H, W], got "
                f"{tuple(volumes.shape)}"
            )
        batch = self._batch_from_volumes(volumes)
        embeddings = self.pipeline.embed(batch, use_amp=self.device_type == "cuda")
        boundaries = batch["series_cu_seqlens"].tolist()
        return torch.stack(
            [embeddings[start:end].mean(dim=0) for start, end in zip(boundaries, boundaries[1:])]
        )


class NeuroVFMVolumeClassifier(nn.Module):
    """Whole-volume NeuroVFM classifier with the multi-slice DINO head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        input_channels: int = 1,
        volume_shape: Sequence[int] = (128, 192, 192),
        input_normalization: str = "none",
        hidden_dim: Optional[int] = None,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError(
                "NeuroVFM consumes native one-channel volumes; "
                f"got input_channels={input_channels}"
            )
        if len(volume_shape) != 3 or any(int(size) <= 0 for size in volume_shape):
            raise ValueError("volume_shape must contain three positive D/H/W dimensions")
        if any(int(size) % patch for size, patch in zip(volume_shape, (4, 16, 16))):
            raise ValueError("NeuroVFM volume_shape must be divisible by (4, 16, 16)")
        if input_normalization != "none":
            raise ValueError(
                "NeuroVFM expects native [0, 1] single-channel volumes; "
                "input_normalization must be 'none'"
            )
        if num_classes <= 0 or not hasattr(encoder, "embed_dim"):
            raise ValueError("NeuroVFM encoder must expose embed_dim and num_classes must be positive")
        self.encoder = encoder
        self.input_channels = int(input_channels)
        self.volume_shape = tuple(int(size) for size in volume_shape)
        self.input_normalization = input_normalization
        self.num_classes = int(num_classes)
        self.input_projection = nn.Identity()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        embed_dim = int(encoder.embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        classifier_hidden = int(hidden_dim or embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden), nn.Dropout(0.5),
            nn.LayerNorm(classifier_hidden), nn.GELU(), nn.Linear(classifier_hidden, self.num_classes),
        )

    def _prepare_volume(self, images: torch.Tensor, slice_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if images.ndim == 4:
            volume = images.unsqueeze(1)  # [B, D, H, W]
        elif images.ndim == 5 and images.shape[1] == 1:
            volume = images  # Native ADNI [B, 1, D, H, W]
        elif images.ndim == 5 and images.shape[2] == 1:
            volume = images.permute(0, 2, 1, 3, 4)  # [B, D, 1, H, W]
        else:
            raise ValueError(
                "NeuroVFM expects a native one-channel volume shaped "
                "[B, 1, D, H, W], [B, D, 1, H, W], or [B, D, H, W]; got "
                f"{tuple(images.shape)}"
            )
        volume = volume.to(torch.float32)
        if slice_mask is not None:
            valid = slice_mask.to(device=volume.device, dtype=torch.bool)
            if valid.shape != (volume.shape[0], volume.shape[2]) or not bool(valid.any(dim=1).all()):
                raise ValueError("slice_mask must retain at least one slice in every volume")
            volume = torch.cat([
                F.interpolate(volume[index:index + 1, :, :int(valid[index].sum())], size=self.volume_shape, mode="trilinear", align_corners=False)
                for index in range(volume.shape[0])
            ])
        else:
            volume = F.interpolate(volume, size=self.volume_shape, mode="trilinear", align_corners=False)
        return self.input_projection(volume)

    def forward(self, images: torch.Tensor, *, slice_mask: Optional[torch.Tensor] = None, profiler: Optional[Any] = None) -> torch.Tensor:
        volume = self._prepare_volume(images, slice_mask)
        with (profiler.stage("dino_forward_s") if profiler is not None else nullcontext()):
            with torch.no_grad():
                token = self.encoder(volume)
        with (profiler.stage("rest_network_forward_s") if profiler is not None else nullcontext()):
            logits = self.classifier(self.output_norm(token))
        return logits.squeeze(-1) if self.num_classes == 1 else logits

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if not name.startswith("encoder.")}

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        missing = [name for name in result.missing_keys if not name.startswith("encoder.")]
        if result.unexpected_keys or missing:
            raise RuntimeError(f"Invalid NeuroVFM state: missing={missing}, unexpected={result.unexpected_keys}")


__all__ = ["DEFAULT_NEUROVFM_REPO", "DEFAULT_NEUROVFM_WEIGHTS", "NeuroVFMEncoder", "NeuroVFMVolumeClassifier"]
