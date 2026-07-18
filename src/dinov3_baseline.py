"""Frozen DINOv3 encoder + attention-pooling head for molecular subtype."""

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
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from duke import PHENOTYPE_SENTINEL, DukeBreastMRIDataset  # noqa: E402

DINOV3_REPO = REPO_ROOT / "opt" / "dinov3"
# Full 922-patient conversion (pre / T1). The older nifti tree only has ~20 post_1 vols.
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DEFAULT_WEIGHTS = (
    REPO_ROOT
    / "opt"
    / "dinov3-weights"
    / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Schema: 0=luminal-like, 1=ER/PR+ HER2+, 2=her2, 3=trip neg
MOL_SUBTYPE_COLUMN = "mol_subtype"
NUM_MOL_SUBTYPES = 4
MOL_SUBTYPE_NAMES = (
    "luminal-like",
    "ER/PR+ HER2+",
    "her2",
    "trip neg",
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


class MolSubtypeHead(nn.Module):
    """Attention-pool patch tokens, concat CLS, then a 2-layer MLP → 4 logits.

    MLP layout: ``Linear → BatchNorm → GELU → Linear``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = NUM_MOL_SUBTYPES,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.pool = AttentionPooling(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(patch_tokens)
        return self.mlp(torch.cat([cls_token, pooled], dim=-1))


class DinoV3MolSubtypeModel(nn.Module):
    """Frozen DINOv3 backbone + trainable molecular-subtype head."""

    def __init__(self, encoder: nn.Module, head: MolSubtypeHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.encoder.forward_features(images)
        cls = out["x_norm_clstoken"].float()
        patch = out["x_norm_patchtokens"].float()
        return cls, patch

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls, patch = self.encode(images)
        return self.head(cls, patch)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def mol_subtype_target(vector: np.ndarray) -> int:
    """Map dataset float vector ``[mol_subtype]`` → class index (or sentinel)."""
    value = float(np.asarray(vector).reshape(-1)[0])
    if value == PHENOTYPE_SENTINEL or np.isnan(value):
        return int(PHENOTYPE_SENTINEL)
    return int(value)


def collect_slice_labels(dataset: DukeBreastMRIDataset) -> np.ndarray:
    """Return length-N array of class indices (sentinel for missing)."""
    labels = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        labels[i] = mol_subtype_target(dataset.get_target(i))
    return labels


def summarize_class_counts(
    labels: np.ndarray,
    *,
    name: str,
) -> Dict[int, int]:
    """Log and return labeled class counts (excludes sentinel)."""
    labeled = labels[labels != int(PHENOTYPE_SENTINEL)]
    counts = Counter(int(c) for c in labeled.tolist())
    n = max(sum(counts.values()), 1)
    logger.info("%s class distribution (n=%d labeled):", name, len(labeled))
    for c in range(NUM_MOL_SUBTYPES):
        k = counts.get(c, 0)
        logger.info(
            "  %d %-14s %5d  (%5.1f%%)",
            c,
            MOL_SUBTYPE_NAMES[c],
            k,
            100.0 * k / n,
        )
    n_missing = int((labels == int(PHENOTYPE_SENTINEL)).sum())
    if n_missing:
        logger.info("  missing=%d", n_missing)
    return {c: counts.get(c, 0) for c in range(NUM_MOL_SUBTYPES)}


def inverse_frequency_weights(
    counts: Dict[int, int],
    num_classes: int = NUM_MOL_SUBTYPES,
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
    dataset: DukeBreastMRIDataset,
    labels: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset]]:
    """Split by patient so no patient appears in both train and val.

    Stratifies patients by their mol_subtype when possible.
    """
    if val_frac <= 0:
        return Subset(dataset, list(range(len(dataset)))), None

    patient_to_indices: Dict[str, List[int]] = {}
    patient_to_label: Dict[str, int] = {}
    for i, (pid, *_rest) in enumerate(dataset._entries):
        patient_to_indices.setdefault(pid, []).append(i)
        y = int(labels[i])
        if y != int(PHENOTYPE_SENTINEL):
            patient_to_label[pid] = y

    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[str]] = {c: [] for c in range(NUM_MOL_SUBTYPES)}
    unlabeled_patients: List[str] = []
    for pid in patient_to_indices:
        if pid in patient_to_label:
            by_class[patient_to_label[pid]].append(pid)
        else:
            unlabeled_patients.append(pid)

    train_pids: List[str] = []
    val_pids: List[str] = []
    for c, pids in by_class.items():
        rng.shuffle(pids)
        n_val = int(round(len(pids) * val_frac))
        # Keep at least one train patient when a class has ≥2 patients.
        if len(pids) >= 2:
            n_val = min(max(n_val, 1) if val_frac > 0 else 0, len(pids) - 1)
        elif len(pids) == 1:
            n_val = 0  # never put a singleton class only in val
        val_pids.extend(pids[:n_val])
        train_pids.extend(pids[n_val:])
        logger.info(
            "patient split class %d (%s): train=%d val=%d",
            c,
            MOL_SUBTYPE_NAMES[c],
            len(pids) - n_val,
            n_val,
        )

    rng.shuffle(unlabeled_patients)
    n_val_u = int(round(len(unlabeled_patients) * val_frac))
    val_pids.extend(unlabeled_patients[:n_val_u])
    train_pids.extend(unlabeled_patients[n_val_u:])

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
    seed: int = 0,
) -> Tuple[DukeBreastMRIDataset, DataLoader]:
    """Unpaired middle-band axial slices + molecular subtype class index."""
    dataset = DukeBreastMRIDataset(
        root=root,
        scan=scan,
        return_pair=False,
        z_min=z_min,
        z_max=z_max,
        phenotype_columns=[MOL_SUBTYPE_COLUMN],
        transform=build_transform(image_size),
        target_transform=mol_subtype_target,
        seed=seed,
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
# DINOv3 loading
# ---------------------------------------------------------------------------


def load_dinov3_encoder(
    *,
    repo_dir: Path = DINOV3_REPO,
    weights: Path = DEFAULT_WEIGHTS,
    model_name: str = "dinov3_vitb16",
    device: torch.device,
) -> nn.Module:
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    encoder = torch.hub.load(
        str(repo_dir),
        model=model_name,
        source="local",
        weights=str(weights),
    )
    encoder.to(device).eval()
    return encoder


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: DinoV3MolSubtypeModel,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Return mean CE loss and accuracy over labeled (non-sentinel) samples."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_labeled = 0
    weight = None if class_weights is None else class_weights.to(device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        logits = model(images)

        labeled = targets != int(PHENOTYPE_SENTINEL)
        n_labeled = int(labeled.sum().item())
        if n_labeled == 0:
            continue
        loss = F.cross_entropy(logits[labeled], targets[labeled], weight=weight)
        total_loss += float(loss.item()) * n_labeled
        total_correct += int(
            (logits[labeled].argmax(dim=-1) == targets[labeled]).sum().item()
        )
        total_labeled += n_labeled

    if total_labeled == 0:
        return float("nan"), float("nan")
    return total_loss / total_labeled, total_correct / total_labeled


@torch.no_grad()
def collect_predictions(
    model: DinoV3MolSubtypeModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``y_true``, ``y_pred``, ``y_prob`` for labeled samples only."""
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        logits = model(images)
        labeled = targets != int(PHENOTYPE_SENTINEL)
        if not labeled.any():
            continue
        t = targets[labeled]
        p = logits[labeled].softmax(dim=-1)
        ys.append(t.cpu().numpy())
        preds.append(p.argmax(dim=-1).cpu().numpy())
        probs.append(p.cpu().numpy())
    if not ys:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_p = np.zeros((0, NUM_MOL_SUBTYPES), dtype=np.float64)
        return empty_i, empty_i, empty_p
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(probs, axis=0),
    )


def compute_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC; NaN if a class is absent."""
    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(NUM_MOL_SUBTYPES)),
            )
        )
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
        labels=list(range(NUM_MOL_SUBTYPES)),
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(MOL_SUBTYPE_NAMES),
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Molecular subtype — validation confusion matrix")
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
    weights: Path,
    repo_dir: Path,
    model_name: str,
    image_size: int,
    checkpoint_dir: Path,
) -> None:
    torch.manual_seed(seed)

    full_ds = DukeBreastMRIDataset(
        root=data_root,
        scan=scan,
        return_pair=False,
        z_min=z_min,
        z_max=z_max,
        phenotype_columns=[MOL_SUBTYPE_COLUMN],
        transform=build_transform(image_size),
        target_transform=mol_subtype_target,
        seed=seed,
    )
    all_labels = collect_slice_labels(full_ds)
    summarize_class_counts(all_labels, name="full dataset (slice)")

    train_ds, val_ds = patient_level_split(
        full_ds,
        all_labels,
        val_frac=val_frac,
        seed=seed,
    )

    train_labels = all_labels[train_ds.indices]
    train_counts = summarize_class_counts(train_labels, name="train (slice)")
    class_weights = inverse_frequency_weights(train_counts)
    logger.info(
        "CE class weights (inverse frequency): %s",
        {
            MOL_SUBTYPE_NAMES[c]: round(float(class_weights[c]), 4)
            for c in range(NUM_MOL_SUBTYPES)
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

    encoder = load_dinov3_encoder(
        repo_dir=repo_dir,
        weights=weights,
        model_name=model_name,
        device=device,
    )
    embed_dim = int(encoder.embed_dim)
    head = MolSubtypeHead(
        embed_dim=embed_dim,
        num_classes=NUM_MOL_SUBTYPES,
        hidden_dim=hidden_dim or embed_dim,
    )
    model = DinoV3MolSubtypeModel(encoder, head).to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    logger.info(
        "Train slices=%d val_slices=%d classes=%s embed_dim=%d",
        len(train_ds),
        len(val_ds) if val_ds is not None else 0,
        list(MOL_SUBTYPE_NAMES),
        embed_dim,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()  # keep frozen BN/dropout off
        running = 0.0
        correct = 0
        labeled_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            mask = targets != int(PHENOTYPE_SENTINEL)
            n = int(mask.sum().item())
            if n == 0:
                continue

            logits = model(images)
            loss = F.cross_entropy(
                logits[mask],
                targets[mask],
                weight=class_weights,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * n
            correct += int((logits[mask].argmax(dim=-1) == targets[mask]).sum().item())
            labeled_count += n

        train_loss = running / max(labeled_count, 1)
        train_acc = correct / max(labeled_count, 1)
        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model, val_loader, device, class_weights=class_weights
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                ckpt_path = checkpoint_dir / "best_head.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "head": model.head.state_dict(),
                        "num_classes": NUM_MOL_SUBTYPES,
                        "class_names": list(MOL_SUBTYPE_NAMES),
                        "class_weights": class_weights.detach().cpu(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    ckpt_path,
                )
            logger.info(
                "epoch %d/%d train_ce=%.5f train_acc=%.3f val_ce=%.5f val_acc=%.3f%s",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
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
        auroc = compute_macro_auroc(y_true, y_prob)
        cm_path = checkpoint_dir / "val_confusion_matrix.png"
        cm = save_confusion_matrix(y_true, y_pred, cm_path)
        logger.info(
            "Final val metrics: n=%d acc=%.4f macro_auroc=%.4f",
            len(y_true),
            float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
            auroc,
        )
        logger.info("Confusion matrix:\n%s", cm)
    else:
        auroc = None

    last_path = checkpoint_dir / "last_head.pt"
    torch.save(
        {
            "epoch": epochs,
            "head": model.head.state_dict(),
            "num_classes": NUM_MOL_SUBTYPES,
            "class_names": list(MOL_SUBTYPE_NAMES),
            "class_weights": class_weights.detach().cpu(),
            "val_loss": best_val if val_loader is not None else None,
            "val_macro_auroc": auroc,
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
    p.add_argument("--model-name", type=str, default="dinov3_vitb16")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "dinov3_baseline",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


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
        weights=args.weights,
        repo_dir=args.dinov3_repo,
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
