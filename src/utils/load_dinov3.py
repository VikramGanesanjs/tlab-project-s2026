"""Load the project's local DINOv3 ViT backbones through Torch Hub."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DINOV3_REPO = REPO_ROOT / "opt" / "dinov3"
DEFAULT_DINOV3_WEIGHTS = (
    REPO_ROOT / "opt" / "dinov3-weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DEFAULT_MEDDINOV3_WEIGHTS = REPO_ROOT / "opt" / "meddinov3" / "model.pth"
DEFAULT_BRAINDINO_WEIGHTS = REPO_ROOT / "opt" / "braindino" / "brain_dino_weights.pth"
DEFAULT_WEIGHTS = DEFAULT_DINOV3_WEIGHTS
ENCODER_CHOICES = ("dinov3", "meddinov3", "braindino", "custom")
FEATURE_CHOICES = ("both", "cls", "patch")
DEFAULT_ENCODER_WEIGHTS = {
    "dinov3": DEFAULT_DINOV3_WEIGHTS,
    "meddinov3": DEFAULT_MEDDINOV3_WEIGHTS,
    "braindino": DEFAULT_BRAINDINO_WEIGHTS,
    "custom": None,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("non-cancerous", "cancerous")


def _ensure_dinov3_repo(repo_dir: Path) -> Path:
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")
    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    return repo_dir


def load_dinov3_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_DINOV3_WEIGHTS,
    model_name: str = "dinov3_vitb16",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a local DINOv3 encoder and optional local checkpoint via Torch Hub."""
    repo_dir = _ensure_dinov3_repo(repo_dir)
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = torch.hub.load(
        str(repo_dir),
        model=model_name,
        source="local",
        weights=str(weights),
    )
    encoder.to(device).eval()
    LOGGER.info("Loaded DINOv3 encoder=%s weights=%s", model_name, weights)
    return encoder


def _load_dinov3_teacher_backbone(
    *,
    repo_dir: Path,
    weights: Path,
    device: Optional[torch.device],
    label: str,
) -> nn.Module:
    """Load a DINOv3-format teacher checkpoint used by adapted ViT-B models."""
    _ensure_dinov3_repo(repo_dir)
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"{label} weights not found: {weights}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from dinov3.models.vision_transformer import vit_base

    encoder = vit_base(
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("teacher"), dict):
        keys = list(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise KeyError(f"{label} checkpoint missing 'teacher' state dict; got {keys}")
    state = {
        str(name).removeprefix("backbone."): value
        for name, value in checkpoint["teacher"].items()
        if "ibot" not in str(name) and "dino_head" not in str(name)
    }
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    LOGGER.info("Loaded %s encoder weights=%s", label, weights)
    return encoder


def load_meddinov3_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_MEDDINOV3_WEIGHTS,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load the local MedDINOv3 ViT-B teacher checkpoint."""
    return _load_dinov3_teacher_backbone(
        repo_dir=repo_dir,
        weights=weights,
        device=device,
        label="MedDINOv3",
    )


def load_braindino_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_BRAINDINO_WEIGHTS,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load the local BrainDINO ViT-B teacher checkpoint."""
    return _load_dinov3_teacher_backbone(
        repo_dir=repo_dir,
        weights=weights,
        device=device,
        label="BrainDINO",
    )


def load_custom_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Reserved for a user-defined backbone with the DINOv3 feature API."""
    raise NotImplementedError(
        "Custom encoder loading is not implemented; use encoder='dinov3', "
        "'meddinov3', or 'braindino'. "
        f"repo_dir={repo_dir}, weights={weights}, device={device}"
    )


def load_encoder(
    encoder: str = "dinov3",
    *,
    device: Optional[torch.device] = None,
    weights: Optional[Path] = None,
    repo_dir: Path = DINOV3_REPO,
    model_name: str = "dinov3_vitb16",
) -> nn.Module:
    """Load a local DINOv3-family encoder by name."""
    name = encoder.strip().lower()
    if name not in ENCODER_CHOICES:
        raise ValueError(f"Unknown encoder={encoder!r}; choose from {ENCODER_CHOICES}")
    resolved_weights = Path(weights) if weights is not None else DEFAULT_ENCODER_WEIGHTS[name]
    if name == "dinov3":
        assert resolved_weights is not None
        return load_dinov3_encoder(
            repo_dir=repo_dir,
            weights=resolved_weights,
            model_name=model_name,
            device=device,
        )
    if name == "meddinov3":
        assert resolved_weights is not None
        return load_meddinov3_encoder(
            repo_dir=repo_dir,
            weights=resolved_weights,
            device=device,
        )
    if name == "braindino":
        assert resolved_weights is not None
        return load_braindino_encoder(
            repo_dir=repo_dir,
            weights=resolved_weights,
            device=device,
        )
    return load_custom_encoder(repo_dir=repo_dir, weights=resolved_weights, device=device)


def inverse_frequency_weights(
    counts: Mapping[int, int],
    num_classes: int = len(CLASS_NAMES),
) -> torch.Tensor:
    """Compute sklearn-style balanced class weights from class counts."""
    total = float(sum(counts.get(index, 0) for index in range(num_classes)))
    weights = torch.ones(num_classes, dtype=torch.float32)
    for index in range(num_classes):
        count = counts.get(index, 0)
        weights[index] = total / (num_classes * count) if count > 0 and total > 0 else 0.0
    return weights


def patient_strata(raw: Mapping[str, Any]) -> Optional[int]:
    """Map Duke phenotype fields to the unilateral/bilateral split stratum."""
    bilateral = raw.get("bilateral")
    if bilateral is not None and str(bilateral).strip() in {"1", "1.0"}:
        return 2
    if isinstance(bilateral, (int, float)) and float(bilateral) == 1.0:
        return 2
    location = raw.get("tumor_location")
    if location is None:
        return None
    normalized = str(location).strip().upper()
    if normalized in {"L", "LEFT", "0", "0.0"}:
        return 0
    if normalized in {"R", "RIGHT", "1", "1.0"}:
        return 1
    return None


__all__ = [
    "DINOV3_REPO",
    "REPO_ROOT",
    "DEFAULT_DINOV3_WEIGHTS",
    "DEFAULT_MEDDINOV3_WEIGHTS",
    "DEFAULT_BRAINDINO_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "DEFAULT_ENCODER_WEIGHTS",
    "ENCODER_CHOICES",
    "FEATURE_CHOICES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CLASS_NAMES",
    "load_dinov3_encoder",
    "load_meddinov3_encoder",
    "load_braindino_encoder",
    "load_custom_encoder",
    "load_encoder",
    "inverse_frequency_weights",
    "patient_strata",
]
