"""Frozen DINOv3 encoder + attention-pooling head for breast cancer classification."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datasets.duke import DukeClassificationDataset, build_duke_transform  # noqa: E402

DINOV3_REPO = REPO_ROOT / "opt" / "dinov3"
# Full 922-patient conversion (pre / T1). The older nifti tree only has ~20 post_1 vols.
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DEFAULT_DINOV3_WEIGHTS = (
    REPO_ROOT
    / "opt"
    / "dinov3-weights"
    / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DEFAULT_MEDDINOV3_WEIGHTS = REPO_ROOT / "opt" / "meddinov3" / "model.pth"
# Back-compat alias
DEFAULT_WEIGHTS = DEFAULT_DINOV3_WEIGHTS

ENCODER_CHOICES = ("dinov3", "meddinov3", "custom")
FEATURE_CHOICES = ("both", "cls", "patch")
DEFAULT_ENCODER_WEIGHTS = {
    "dinov3": DEFAULT_DINOV3_WEIGHTS,
    "meddinov3": DEFAULT_MEDDINOV3_WEIGHTS,
    "custom": None,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 2
CLASS_NAMES = (
    "non-cancerous",
    "cancerous",
)

# Patient-level strata for train/val split (not model classes).
NUM_STRATA = 3
STRATA_NAMES = (
    "unilateral-L",
    "unilateral-R",
    "bilateral",
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """Softmax attention pool over patch tokens → single vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        weights = self.score(tokens).softmax(dim=1)  # [B, N, 1]
        return (weights * tokens).sum(dim=1)  # [B, D]


class CancerHead(nn.Module):
    """Select CLS and/or attention-pooled patch features, then classify.

    MLP layout: ``Linear → BatchNorm → GELU → Linear``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: Optional[int] = None,
        features: str = "both",
    ) -> None:
        super().__init__()
        if features not in FEATURE_CHOICES:
            raise ValueError(f"Unknown features={features!r}; choose from {FEATURE_CHOICES}")
        self.features = features
        hidden_dim = hidden_dim or embed_dim
        self.pool = AttentionPooling(embed_dim) if features != "cls" else None
        input_dim = 2 * embed_dim if features == "both" else embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.features == "cls":
            features = cls_token
        else:
            assert self.pool is not None
            pooled = self.pool(patch_tokens)
            features = pooled if self.features == "patch" else torch.cat(
                [cls_token, pooled], dim=-1
            )
        return self.mlp(features)


class DinoV3CancerModel(nn.Module):
    """Frozen DINOv3 backbone + trainable cancer-classification head."""

    def __init__(self, encoder: nn.Module, head: CancerHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.encoder.forward_features(images)
        cls = F.normalize(out["x_norm_clstoken"].float(), p=2, dim=-1)
        patch = F.normalize(out["x_norm_patchtokens"].float(), p=2, dim=-1)
        return cls, patch

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls, patch = self.encode(images)
        return self.head(cls, patch)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_transform(
    image_size: int = 224,
    *,
    augment: bool = True,
    crop_scale_min: float = 0.8,
    jitter: float = 0.2,
    rotation_degrees: float = 15.0,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
) -> transforms.Compose:
    return build_duke_transform(
        image_size,
        augment=augment,
        crop_scale_min=crop_scale_min,
        jitter=jitter,
        rotation_degrees=rotation_degrees,
        horizontal_flip_prob=horizontal_flip_prob,
        vertical_flip_prob=vertical_flip_prob,
    )


def collect_slice_labels(dataset: DukeClassificationDataset) -> np.ndarray:
    """Return length-N array of binary class indices."""
    labels = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        labels[i] = int(dataset.get_target(i))
    return labels


def patient_strata(raw: Dict) -> Optional[int]:
    """Map phenotype raw fields to a patient strata index for splitting."""
    bil = raw.get("bilateral")
    if bil is not None and str(bil).strip() in {"1", "1.0"}:
        return 2
    if isinstance(bil, (int, float)) and float(bil) == 1.0:
        return 2
    loc = raw.get("tumor_location")
    if loc is None:
        return None
    s = str(loc).strip().upper()
    if s in {"L", "LEFT", "0", "0.0"}:
        return 0
    if s in {"R", "RIGHT", "1", "1.0"}:
        return 1
    return None


def summarize_class_counts(
    labels: np.ndarray,
    *,
    name: str,
) -> Dict[int, int]:
    """Log and return class counts."""
    counts = Counter(int(c) for c in labels.tolist())
    n = max(sum(counts.values()), 1)
    logger.info("%s class distribution (n=%d):", name, len(labels))
    for c in range(NUM_CLASSES):
        k = counts.get(c, 0)
        logger.info(
            "  %d %-14s %5d  (%5.1f%%)",
            c,
            CLASS_NAMES[c],
            k,
            100.0 * k / n,
        )
    return {c: counts.get(c, 0) for c in range(NUM_CLASSES)}


def inverse_frequency_weights(
    counts: Dict[int, int],
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """sklearn-style balanced weights: ``n_samples / (n_classes * n_c)``."""
    total = float(sum(counts.get(c, 0) for c in range(num_classes)))
    weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c > 0 and total > 0:
            weights[c] = total / (num_classes * n_c)
        else:
            weights[c] = 0.0
    return weights


def patient_level_split(
    dataset: DukeClassificationDataset,
    *,
    val_frac: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset]]:
    """Split by patient so no patient appears in both train and val.

    Stratifies patients by laterality (L / R / bilateral) when possible.
    """
    if val_frac <= 0:
        return Subset(dataset, list(range(len(dataset)))), None

    patient_to_indices: Dict[str, List[int]] = {}
    patient_to_stratum: Dict[str, int] = {}
    for i, (pid, *_rest) in enumerate(dataset._entries):
        patient_to_indices.setdefault(pid, []).append(i)
        if pid not in patient_to_stratum:
            stratum = patient_strata(dataset.get_phenotype_raw(i))
            if stratum is not None:
                patient_to_stratum[pid] = stratum

    rng = np.random.RandomState(seed)
    by_stratum: Dict[int, List[str]] = {c: [] for c in range(NUM_STRATA)}
    unstratified_patients: List[str] = []
    for pid in patient_to_indices:
        if pid in patient_to_stratum:
            by_stratum[patient_to_stratum[pid]].append(pid)
        else:
            unstratified_patients.append(pid)

    train_pids: List[str] = []
    val_pids: List[str] = []
    for c, pids in by_stratum.items():
        rng.shuffle(pids)
        n_val = int(round(len(pids) * val_frac))
        # Keep at least one train patient when a stratum has ≥2 patients.
        if len(pids) >= 2:
            n_val = min(max(n_val, 1) if val_frac > 0 else 0, len(pids) - 1)
        elif len(pids) == 1:
            n_val = 0  # never put a singleton stratum only in val
        val_pids.extend(pids[:n_val])
        train_pids.extend(pids[n_val:])
        logger.info(
            "patient split stratum %d (%s): train=%d val=%d",
            c,
            STRATA_NAMES[c],
            len(pids) - n_val,
            n_val,
        )

    rng.shuffle(unstratified_patients)
    n_val_u = int(round(len(unstratified_patients) * val_frac))
    val_pids.extend(unstratified_patients[:n_val_u])
    train_pids.extend(unstratified_patients[n_val_u:])

    train_idx = [i for pid in train_pids for i in patient_to_indices[pid]]
    val_idx = [i for pid in val_pids for i in patient_to_indices[pid]]
    overlap = set(train_pids) & set(val_pids)
    if overlap:
        raise RuntimeError(f"Patient leakage in split: {sorted(overlap)}")

    logger.info(
        "patient-level split: train_patients=%d val_patients=%d "
        "train_slices=%d val_slices=%d",
        len(train_pids),
        len(val_pids),
        len(train_idx),
        len(val_idx),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if val_idx else None
    return train_ds, val_ds


def build_dataloader(
    *,
    root: Path,
    scan: str = "pre",
    batch_size: int = 32,
    num_workers: int = 4,
    z_min: float = 0.25,
    z_max: float = 0.75,
    image_size: int = 224,
    shuffle: bool = True,
) -> Tuple[DukeClassificationDataset, DataLoader]:
    """Left/right breast axial slices + binary cancer label."""
    dataset = DukeClassificationDataset(
        root=root,
        scan=scan,
        z_min=z_min,
        z_max=z_max,
        transform=build_transform(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# Encoder loading (DINOv3-compatible: forward_features → CLS + patch tokens)
# ---------------------------------------------------------------------------


def _ensure_dinov3_repo(repo_dir: Path) -> Path:
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return repo_dir


def load_dinov3_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_DINOV3_WEIGHTS,
    model_name: str = "dinov3_vitb16",
    device: torch.device,
) -> nn.Module:
    """Natural-image DINOv3 via local torch.hub."""
    repo_dir = _ensure_dinov3_repo(repo_dir)
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")

    encoder = torch.hub.load(
        str(repo_dir),
        model=model_name,
        source="local",
        weights=str(weights),
    )
    encoder.to(device).eval()
    logger.info("Loaded DINOv3 encoder=%s weights=%s", model_name, weights)
    return encoder


def load_meddinov3_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_MEDDINOV3_WEIGHTS,
    device: torch.device,
) -> nn.Module:
    """MedDINOv3 (CT-3M domain-adapted ViT-B/16) on the DINOv3 backbone API."""
    repo_dir = _ensure_dinov3_repo(repo_dir)
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"MedDINOv3 weights not found: {weights}")

    from dinov3.models.vision_transformer import vit_base

    encoder = vit_base(
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if "teacher" not in ckpt:
        raise KeyError(
            f"MedDINOv3 checkpoint missing 'teacher' key; got keys={list(ckpt.keys())}"
        )
    state = {
        k.replace("backbone.", ""): v
        for k, v in ckpt["teacher"].items()
        if "ibot" not in k and "dino_head" not in k
    }
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    logger.info("Loaded MedDINOv3 weights=%s", weights)
    return encoder


def load_custom_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Optional[Path] = None,
    device: torch.device,
) -> nn.Module:
    """Wireframe for a user DINOv3-compatible encoder.

    Expected interface (same as DINOv3 / MedDINOv3):

    * ``encoder.forward_features(images) -> dict`` with
      ``x_norm_clstoken`` ``[B, D]`` and ``x_norm_patchtokens`` ``[B, N, D]``
    * ``encoder.embed_dim: int``
    """
    raise NotImplementedError(
        "Custom encoder is a wireframe only. Implement load_custom_encoder() "
        f"(repo_dir={repo_dir}, weights={weights}, device={device}) or pass "
        "--encoder dinov3|meddinov3."
    )


def load_encoder(
    encoder: str,
    *,
    device: torch.device,
    weights: Optional[Path] = None,
    repo_dir: Path = DINOV3_REPO,
    model_name: str = "dinov3_vitb16",
) -> nn.Module:
    """Dispatch to a DINOv3-framework encoder by name."""
    name = encoder.strip().lower()
    if name not in ENCODER_CHOICES:
        raise ValueError(
            f"Unknown encoder={encoder!r}; choose from {ENCODER_CHOICES}"
        )
    resolved = Path(weights) if weights is not None else DEFAULT_ENCODER_WEIGHTS[name]

    if name == "dinov3":
        if resolved is None:
            raise ValueError("--weights is required for encoder=dinov3")
        return load_dinov3_encoder(
            repo_dir=repo_dir,
            weights=resolved,
            model_name=model_name,
            device=device,
        )
    if name == "meddinov3":
        if resolved is None:
            raise ValueError("--weights is required for encoder=meddinov3")
        return load_meddinov3_encoder(
            repo_dir=repo_dir,
            weights=resolved,
            device=device,
        )
    return load_custom_encoder(
        repo_dir=repo_dir,
        weights=resolved,
        device=device,
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: DinoV3CancerModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Return mean CE loss and binary F1 (positive = cancerous)."""
    model.eval()
    total_loss = 0.0
    total = 0
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    weight = None if class_weights is None else class_weights.to(device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        logits = model(images)
        n = int(targets.numel())
        if n == 0:
            continue
        loss = F.cross_entropy(logits, targets, weight=weight)
        total_loss += float(loss.item()) * n
        total += n
        ys.append(targets.cpu().numpy())
        preds.append(logits.argmax(dim=-1).cpu().numpy())

    if total == 0:
        return float("nan"), float("nan")
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    val_f1 = float(
        f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    )
    return total_loss / total, val_f1


@torch.no_grad()
def collect_predictions(
    model: DinoV3CancerModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``y_true``, ``y_pred``, ``y_prob`` (P(cancerous))."""
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        logits = model(images)
        p = logits.softmax(dim=-1)
        ys.append(targets.cpu().numpy())
        preds.append(p.argmax(dim=-1).cpu().numpy())
        probs.append(p[:, 1].cpu().numpy())
    if not ys:
        empty = np.zeros((0,), dtype=np.int64)
        empty_p = np.zeros((0,), dtype=np.float64)
        return empty, empty, empty_p
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
    )


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary AUROC; NaN if a class is absent."""
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        logger.warning("AUROC undefined: %s", exc)
        return float("nan")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> np.ndarray:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(CLASS_NAMES),
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Cancer classification — validation confusion matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote confusion matrix → %s", out_path)
    return cm


def train(
    *,
    data_root: Path,
    scan: str,
    z_min: float,
    z_max: float,
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    val_frac: float,
    seed: int,
    device: torch.device,
    encoder_name: str,
    features: str,
    weights: Optional[Path],
    repo_dir: Path,
    model_name: str,
    image_size: int,
    augment: bool,
    crop_scale_min: float,
    jitter: float,
    rotation_degrees: float,
    horizontal_flip_prob: float,
    vertical_flip_prob: float,
    include_bilateral: bool,
    checkpoint_dir: Path,
) -> None:
    torch.manual_seed(seed)

    full_ds = DukeClassificationDataset(
        root=data_root,
        scan=scan,
        z_min=z_min,
        z_max=z_max,
        include_bilateral=include_bilateral,
        transform=build_transform(
            image_size,
            augment=augment,
            crop_scale_min=crop_scale_min,
            jitter=jitter,
            rotation_degrees=rotation_degrees,
            horizontal_flip_prob=horizontal_flip_prob,
            vertical_flip_prob=vertical_flip_prob,
        ),
    )
    all_labels = collect_slice_labels(full_ds)
    summarize_class_counts(all_labels, name="full dataset (slice)")

    train_ds, val_ds = patient_level_split(
        full_ds,
        val_frac=val_frac,
        seed=seed,
    )

    train_labels = all_labels[train_ds.indices]
    train_counts = summarize_class_counts(train_labels, name="train (slice)")
    class_weights = inverse_frequency_weights(train_counts)
    logger.info(
        "CE class weights (inverse frequency): %s",
        {
            CLASS_NAMES[c]: round(float(class_weights[c]), 4)
            for c in range(NUM_CLASSES)
        },
    )

    if val_ds is not None:
        summarize_class_counts(all_labels[val_ds.indices], name="val (slice)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    encoder = load_encoder(
        encoder_name,
        device=device,
        weights=weights,
        repo_dir=repo_dir,
        model_name=model_name,
    )
    embed_dim = int(encoder.embed_dim)
    head = CancerHead(
        embed_dim=embed_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim or embed_dim,
        features=features,
    )
    model = DinoV3CancerModel(encoder, head).to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    logger.info(
        "encoder=%s features=%s train_slices=%d val_slices=%d classes=%s embed_dim=%d",
        encoder_name,
        features,
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        list(CLASS_NAMES),
        embed_dim,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()  # keep frozen BN/dropout off
        running = 0.0
        correct = 0
        n_seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            n = int(targets.numel())
            if n == 0:
                continue

            logits = model(images)
            loss = F.cross_entropy(logits, targets, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * n
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            n_seen += n

        train_loss = running / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)
        if val_loader is not None:
            val_loss, val_f1 = evaluate(
                model, val_loader, device, class_weights=class_weights
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                ckpt_path = checkpoint_dir / "best_head.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder": encoder_name,
                        "features": features,
                        "head": model.head.state_dict(),
                        "num_classes": NUM_CLASSES,
                        "class_names": list(CLASS_NAMES),
                        "class_weights": class_weights.detach().cpu(),
                        "val_loss": val_loss,
                        "val_f1": val_f1,
                    },
                    ckpt_path,
                )
            logger.info(
                "epoch %d/%d train_ce=%.5f train_acc=%.3f val_ce=%.5f val_f1=%.3f%s",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_f1,
                " *" if improved else "",
            )
        else:
            logger.info(
                "epoch %d/%d train_ce=%.5f train_acc=%.3f",
                epoch,
                epochs,
                train_loss,
                train_acc,
            )

    if val_loader is not None:
        y_true, y_pred, y_prob = collect_predictions(model, val_loader, device)
        auroc = compute_auroc(y_true, y_prob)
        final_f1 = float(
            f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        ) if len(y_true) else float("nan")
        cm_path = checkpoint_dir / "val_confusion_matrix.png"
        cm = save_confusion_matrix(y_true, y_pred, cm_path)
        logger.info(
            "Final val metrics: n=%d f1=%.4f auroc=%.4f",
            len(y_true),
            final_f1,
            auroc,
        )
        logger.info("Confusion matrix:\n%s", cm)
    else:
        auroc = None

    last_path = checkpoint_dir / "last_head.pt"
    torch.save(
        {
            "epoch": epochs,
            "encoder": encoder_name,
            "features": features,
            "head": model.head.state_dict(),
            "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES),
            "class_weights": class_weights.detach().cpu(),
            "val_loss": best_val if val_loader is not None else None,
            "val_auroc": auroc,
        },
        last_path,
    )
    logger.info("Wrote %s", last_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--scan", type=str, default="pre")
    p.add_argument(
        "--z-min",
        type=float,
        default=0.25,
        help="Min fractional z (inclusive). Middle 50%% → 0.25",
    )
    p.add_argument(
        "--z-max",
        type=float,
        default=0.75,
        help="Max fractional z (exclusive after rounding). Middle 50%% → 0.75",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=0, help="0 → use embed_dim")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--include-bilateral",
        action="store_true",
        help="Include bilateral cases (excluded by default)",
    )
    augmentation = p.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        default=True,
        help="Enable random crop, jitter, rotation, and flips (default)",
    )
    augmentation.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Use deterministic resize preprocessing",
    )
    p.add_argument("--crop-scale-min", type=float, default=0.8)
    p.add_argument("--jitter", type=float, default=0.2)
    p.add_argument("--rotation-degrees", type=float, default=15.0)
    p.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    p.add_argument("--vertical-flip-prob", type=float, default=0.5)
    p.add_argument(
        "--encoder",
        type=str,
        choices=list(ENCODER_CHOICES),
        default="dinov3",
        help="Frozen backbone: dinov3 | meddinov3 | custom (wireframe)",
    )
    p.add_argument(
        "--features",
        type=str,
        choices=list(FEATURE_CHOICES),
        default="both",
        help="Head input: both CLS + pooled patches, CLS only, or pooled patches only",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vitb16",
        help="torch.hub model name (used when --encoder dinov3)",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Encoder checkpoint. Defaults: dinov3 → opt/dinov3-weights/...; "
            "meddinov3 → opt/meddinov3/model.pth"
        ),
    )
    p.add_argument(
        "--dinov3-repo",
        type=Path,
        default=DINOV3_REPO,
        help="Local DINOv3 repo (architecture source for dinov3/meddinov3)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Subfolder under runs/ for checkpoints and confusion matrix. "
            "Defaults to --encoder (e.g. runs/dinov3)."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <repo>/runs/<run-name>)",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def resolve_checkpoint_dir(
    *,
    run_name: Optional[str],
    encoder: str,
    checkpoint_dir: Optional[Path],
) -> Path:
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    name = (run_name or encoder).strip()
    if not name:
        raise ValueError("--run-name must be a non-empty string")
    return REPO_ROOT / "runs" / name


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_dir = resolve_checkpoint_dir(
        run_name=args.run_name,
        encoder=args.encoder,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info("Run output directory: %s", checkpoint_dir)
    train(
        data_root=args.data_root,
        scan=args.scan,
        z_min=args.z_min,
        z_max=args.z_max,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        encoder_name=args.encoder,
        features=args.features,
        weights=args.weights,
        repo_dir=args.dinov3_repo,
        model_name=args.model_name,
        image_size=args.image_size,
        augment=args.augment,
        crop_scale_min=args.crop_scale_min,
        jitter=args.jitter,
        rotation_degrees=args.rotation_degrees,
        horizontal_flip_prob=args.horizontal_flip_prob,
        vertical_flip_prob=args.vertical_flip_prob,
        include_bilateral=args.include_bilateral,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
