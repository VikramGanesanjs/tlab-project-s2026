"""Frozen DINOv3 encoder + multi-slice transformer classification."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dinov3_baseline import (  # noqa: E402
    CLASS_NAMES as DUKE_CLASS_NAMES,
    DINOV3_REPO,
    ENCODER_CHOICES,
    FEATURE_CHOICES,
    REPO_ROOT,
    build_transform,
    inverse_frequency_weights,
    load_encoder,
    patient_strata,
)
from datasets.adni import (  # noqa: E402
    ADNIMultiSliceDataset,
    ADNI_TASK_CHOICES,
    DEFAULT_ADNI_TASK,
    DEFAULT_ROOT as ADNI_DEFAULT_ROOT,
    resolve_adni_task,
    build_adni_transform,
)
from datasets.duke import DukeMultiSliceDataset  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DATASET_CHOICES = ("duke", "adni")
MultiSliceDataset = Union[DukeMultiSliceDataset, ADNIMultiSliceDataset]


class AttentionPooling(nn.Module):
    """Pool patch tokens using two-layer learned attention scores."""

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
    """Frozen per-slice encoder followed by a trainable volume transformer."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        n_slices: int,
        features: str = "cls",
        d_model: int = 768,
        depth: int = 2,
        n_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if features not in FEATURE_CHOICES:
            raise ValueError(f"Unknown features={features!r}; choose from {FEATURE_CHOICES}")
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
        if d_model <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.encoder = encoder
        self.n_slices = int(n_slices)
        self.features = features
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        embed_dim = int(encoder.embed_dim)
        pooling_hidden_dim = hidden_dim or embed_dim
        self.patch_pool = (
            AttentionPooling(embed_dim, pooling_hidden_dim)
            if features != "cls"
            else None
        )
        slice_dim = 2 * embed_dim if features == "both" else embed_dim
        self.slice_projection = (
            nn.Identity() if slice_dim == d_model else nn.Linear(slice_dim, d_model)
        )
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_slices + 1, d_model)
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
        self.output_norm = nn.LayerNorm(d_model)
        classifier_hidden = hidden_dim or d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.Dropout(0.5),
            nn.BatchNorm1d(classifier_hidden),
            nn.GELU(),
            nn.Linear(classifier_hidden, self.num_classes),
        )
        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def encode_slices(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, S, C, H, W]`` into ``[B, S, feature_dim]``."""
        if images.ndim != 5:
            raise ValueError(f"Expected [B, S, C, H, W], got {tuple(images.shape)}")
        batch, n_slices = images.shape[:2]
        if n_slices != self.n_slices:
            raise ValueError(f"Expected {self.n_slices} slices, got {n_slices}")
        flat = images.reshape(batch * n_slices, *images.shape[2:])
        with torch.no_grad():
            output = self.encoder.forward_features(flat)
        cls = F.normalize(output["x_norm_clstoken"].float(), p=2, dim=-1)
        if self.features == "cls":
            token = cls
        else:
            assert self.patch_pool is not None
            patches = F.normalize(output["x_norm_patchtokens"].float(), p=2, dim=-1)
            pooled = self.patch_pool(patches)
            token = pooled if self.features == "patch" else torch.cat([cls, pooled], dim=-1)
        return token.reshape(batch, n_slices, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        slices = self.slice_projection(self.encode_slices(images))
        global_token = self.global_token.expand(slices.shape[0], -1, -1)
        sequence = torch.cat([global_token, slices], dim=1)
        sequence = sequence + self.position_embedding
        encoded = self.transformer(sequence)
        logits = self.classifier(self.output_norm(encoded[:, 0]))
        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """State for all trainable MST components, excluding the frozen encoder."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("encoder.")
        }

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        non_encoder_missing = [
            name for name in result.missing_keys if not name.startswith("encoder.")
        ]
        if unexpected or non_encoder_missing:
            raise RuntimeError(
                f"Invalid MST state: missing={non_encoder_missing}, unexpected={unexpected}"
            )


def is_binary_task(num_classes: int) -> bool:
    """Duke uses a single logit (``num_classes=1``) with BCE."""
    return int(num_classes) == 1


def task_config(
    dataset_name: str,
    *,
    adni_task: str = DEFAULT_ADNI_TASK,
) -> Tuple[int, Tuple[str, ...], str]:
    """Return ``(num_logits, class_names, loss_name)`` for a dataset/task."""
    if dataset_name == "duke":
        # Single logit + BCE; class_names remain the binary labels used in metrics.
        return 1, DUKE_CLASS_NAMES, "BCEWithLogitsLoss"
    if dataset_name == "adni":
        spec = resolve_adni_task(adni_task)
        loss_name = "BCEWithLogitsLoss" if spec.binary else "CrossEntropyLoss"
        return spec.num_logits, spec.class_names, loss_name
    raise ValueError(f"Unknown dataset={dataset_name!r}")


def collect_labels(dataset: Dataset) -> np.ndarray:
    return np.asarray(
        [int(dataset.get_target(index)) for index in range(len(dataset))],  # type: ignore[attr-defined]
        dtype=np.int64,
    )


def summarize_class_counts(
    labels: np.ndarray,
    *,
    name: str,
    class_names: Sequence[str],
) -> Dict[int, int]:
    counts = Counter(int(label) for label in labels.tolist())
    total = max(sum(counts.values()), 1)
    logger.info("%s class distribution (n=%d):", name, len(labels))
    for class_index, class_name in enumerate(class_names):
        count = counts.get(class_index, 0)
        logger.info(
            "  %d %-14s %5d  (%5.1f%%)",
            class_index,
            class_name,
            count,
            100.0 * count / total,
        )
    return {index: counts.get(index, 0) for index in range(len(class_names))}


def compute_auroc(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    num_classes: int,
) -> float:
    if is_binary_task(num_classes):
        if np.unique(y_true).size < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y_true, y_probability))
        except ValueError as exc:
            logger.warning("AUROC undefined: %s", exc)
            return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                y_true,
                y_probability,
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except ValueError as exc:
        logger.warning("AUROC undefined: %s", exc)
        return float("nan")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    class_names: Sequence[str],
) -> np.ndarray:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    display.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote confusion matrix → %s", out_path)
    return matrix


def classification_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    if is_binary_task(num_classes):
        return float(
            f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        )
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )


def should_early_stop(
    *,
    epoch: int,
    best_epoch: int,
    min_epochs: int,
    patience: int,
) -> bool:
    """Return whether validation loss has failed to improve long enough."""
    return (
        best_epoch > 0
        and epoch >= min_epochs
        and epoch - best_epoch >= patience
    )


def _duke_patient_stratum(dataset: DukeMultiSliceDataset, index: int) -> Optional[int]:
    return patient_strata(dataset.get_phenotype_raw(index))


def _adni_patient_stratum(dataset: ADNIMultiSliceDataset, index: int) -> int:
    return int(dataset.get_target(index))


def _held_out_counts(
    n_patients: int,
    *,
    val_frac: float,
    test_frac: float,
) -> Tuple[int, int]:
    """Choose per-stratum val/test sizes, leaving at least one train patient when possible."""
    n_val = int(round(n_patients * val_frac)) if val_frac > 0 else 0
    n_test = int(round(n_patients * test_frac)) if test_frac > 0 else 0
    if n_patients >= 3 and val_frac > 0 and test_frac > 0:
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
        overflow = n_val + n_test - (n_patients - 1)
        if overflow > 0:
            reduce_test = min(overflow, max(n_test - 1, 0))
            n_test -= reduce_test
            overflow -= reduce_test
            n_val -= overflow
    elif n_patients >= 2:
        # Tiny strata can support only one held-out patient while keeping train.
        if val_frac > 0:
            n_val = min(max(n_val, 1), n_patients - 1)
            n_test = 0
        elif test_frac > 0:
            n_test = min(max(n_test, 1), n_patients - 1)
            n_val = 0
        else:
            n_val = n_test = 0
    else:
        n_val = n_test = 0
    return n_val, n_test


def patient_level_split(
    dataset: MultiSliceDataset,
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
    get_stratum: Callable[[Any, int], Optional[int]],
    unit_name: str = "samples",
) -> Tuple[Subset, Optional[Subset], Optional[Subset]]:
    """Split samples by patient into train / validation / test with optional stratification.

    ``test_frac`` patients are reserved and never used for training or model selection.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac must be < 1, got {val_frac} + {test_frac}"
        )
    if val_frac == 0 and test_frac == 0:
        return Subset(dataset, list(range(len(dataset)))), None, None

    patient_indices: Dict[str, List[int]] = defaultdict(list)
    strata: Dict[str, Optional[int]] = {}
    for index in range(len(dataset)):
        patient_id = dataset.get_patient_id(index)
        patient_indices[patient_id].append(index)
        if patient_id not in strata:
            strata[patient_id] = get_stratum(dataset, index)

    rng = np.random.RandomState(seed)
    grouped: Dict[Optional[int], List[str]] = defaultdict(list)
    for patient_id in patient_indices:
        grouped[strata.get(patient_id)].append(patient_id)

    train_patients: List[str] = []
    val_patients: List[str] = []
    test_patients: List[str] = []
    for stratum, patients in grouped.items():
        rng.shuffle(patients)
        n_val, n_test = _held_out_counts(
            len(patients), val_frac=val_frac, test_frac=test_frac
        )
        test_patients.extend(patients[:n_test])
        val_patients.extend(patients[n_test : n_test + n_val])
        train_patients.extend(patients[n_test + n_val :])
        logger.info(
            "patient split stratum=%s train=%d val=%d test=%d",
            stratum,
            len(patients) - n_val - n_test,
            n_val,
            n_test,
        )

    train_indices = [
        index for patient_id in train_patients for index in patient_indices[patient_id]
    ]
    val_indices = [
        index for patient_id in val_patients for index in patient_indices[patient_id]
    ]
    test_indices = [
        index for patient_id in test_patients for index in patient_indices[patient_id]
    ]
    disjoint = (
        set(train_patients).isdisjoint(val_patients)
        and set(train_patients).isdisjoint(test_patients)
        and set(val_patients).isdisjoint(test_patients)
    )
    if not disjoint:
        raise RuntimeError("Patient leakage across train/validation/test split")
    logger.info(
        "patient-level split: train_patients=%d val_patients=%d test_patients=%d "
        "train_%s=%d val_%s=%d test_%s=%d",
        len(train_patients),
        len(val_patients),
        len(test_patients),
        unit_name,
        len(train_indices),
        unit_name,
        len(val_indices),
        unit_name,
        len(test_indices),
    )
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices) if val_indices else None,
        Subset(dataset, test_indices) if test_indices else None,
    )


@torch.no_grad()
def evaluate(
    model: MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
    *,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Return mean validation loss and F1 (binary or macro)."""
    model.eval()
    total_loss = 0.0
    total = 0
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    binary = is_binary_task(model.num_classes)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if binary:
            targets = targets.to(device, non_blocking=True).float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            predictions = (logits >= 0).long()
            target_labels = targets.long()
        else:
            targets = targets.to(device, non_blocking=True).long()
            loss = F.cross_entropy(logits, targets, weight=class_weights)
            predictions = logits.argmax(dim=-1)
            target_labels = targets
        n_items = int(targets.numel())
        total_loss += float(loss.item()) * n_items
        total += n_items
        targets_all.append(target_labels.cpu().numpy())
        predictions_all.append(predictions.cpu().numpy())

    if total == 0:
        return float("nan"), float("nan")
    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(predictions_all)
    return total_loss / total, classification_f1(
        y_true, y_pred, num_classes=model.num_classes
    )


@torch.no_grad()
def collect_predictions(
    model: MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return targets, predictions, and class probabilities."""
    model.eval()
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    binary = is_binary_task(model.num_classes)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if binary:
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()
        else:
            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
        targets_all.append(targets.long().cpu().numpy())
        predictions_all.append(predictions.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
    if not targets_all:
        empty = np.zeros((0,), dtype=np.int64)
        empty_prob = np.zeros((0,) if binary else (0, model.num_classes), dtype=np.float64)
        return empty, empty, empty_prob
    return (
        np.concatenate(targets_all),
        np.concatenate(predictions_all),
        np.concatenate(probabilities_all),
    )


def _checkpoint_payload(
    model: MultiSliceDinoModel,
    *,
    epoch: int,
    args: argparse.Namespace,
    class_names: Sequence[str],
    loss_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **metrics: object,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "epoch": epoch,
        "model": model.trainable_state_dict(),
        "dataset": args.dataset,
        "adni_task": args.adni_task if args.dataset == "adni" else None,
        "encoder": args.encoder,
        "model_name": args.model_name,
        "weights": str(args.weights) if args.weights is not None else None,
        "features": args.features,
        "n_slices": args.n_slices,
        "minimum_z_index_distance": args.minimum_z_index_distance,
        "include_bilateral": args.include_bilateral,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "image_size": args.image_size,
        "augment": args.augment,
        "crop_scale_min": args.crop_scale_min,
        "jitter": args.jitter,
        "rotation_degrees": args.rotation_degrees,
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "vertical_flip_prob": args.vertical_flip_prob,
        "d_model": args.d_model,
        "mst_depth": args.mst_depth,
        "mst_heads": args.mst_heads,
        "mst_ffn_dim": args.mst_ffn_dim,
        "mst_dropout": args.mst_dropout,
        "hidden_dim": args.hidden_dim,
        "min_epochs": args.min_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "cosine_lr": args.cosine_lr,
        "min_lr": args.min_lr,
        "num_classes": model.num_classes,
        "class_names": list(class_names),
        "loss": loss_name,
        **metrics,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def build_dataset(
    args: argparse.Namespace,
    *,
    augment: Optional[bool] = None,
    slice_sampling: str = "random",
) -> MultiSliceDataset:
    use_augment = args.augment if augment is None else bool(augment)
    transform_kwargs = dict(
        image_size=args.image_size,
        augment=use_augment,
        crop_scale_min=args.crop_scale_min,
        jitter=args.jitter,
        rotation_degrees=args.rotation_degrees,
        horizontal_flip_prob=args.horizontal_flip_prob,
        vertical_flip_prob=args.vertical_flip_prob,
    )
    if args.dataset == "duke":
        return DukeMultiSliceDataset(
            root=args.data_root,
            n_slices=args.n_slices,
            minimum_z_index_distance=args.minimum_z_index_distance,
            include_bilateral=args.include_bilateral,
            scan=args.scan,
            z_min=args.z_min,
            z_max=args.z_max,
            transform=build_transform(**transform_kwargs),
            slice_sampling=slice_sampling,
        )
    if args.dataset == "adni":
        return ADNIMultiSliceDataset(
            root=args.data_root,
            csv_path=args.csv_path,
            task=args.adni_task,
            n_slices=args.n_slices,
            minimum_z_index_distance=args.minimum_z_index_distance,
            z_min=args.z_min,
            z_max=args.z_max,
            transform=build_adni_transform(**transform_kwargs),
            slice_sampling=slice_sampling,
        )
    raise ValueError(f"Unknown dataset={args.dataset!r}")


def train(args: argparse.Namespace, device: torch.device, checkpoint_dir: Path) -> None:
    torch.manual_seed(args.seed)
    num_classes, class_names, loss_name = task_config(
        args.dataset, adni_task=args.adni_task
    )
    binary = is_binary_task(num_classes)
    unit_name = "breasts" if args.dataset == "duke" else "scans"
    # Train keeps stochastic augmentations + random slice sampling.
    # Validation/test are deterministic: no augmentations + evenly spaced slices.
    train_full = build_dataset(args, augment=args.augment, slice_sampling="random")
    labels = collect_labels(train_full)
    summarize_class_counts(labels, name=f"full dataset ({unit_name})", class_names=class_names)

    get_stratum: Callable[[Any, int], Optional[int]]
    if args.dataset == "duke":
        get_stratum = _duke_patient_stratum
    else:
        get_stratum = _adni_patient_stratum

    train_dataset, train_val_subset, train_test_subset = patient_level_split(
        train_full,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        get_stratum=get_stratum,
        unit_name=unit_name,
    )
    val_dataset: Optional[Subset] = None
    test_dataset: Optional[Subset] = None
    if train_val_subset is not None or train_test_subset is not None:
        eval_full = build_dataset(args, augment=False, slice_sampling="even")
        if len(eval_full) != len(train_full):
            raise RuntimeError(
                "Train/eval dataset index mismatch: "
                f"train_full={len(train_full)} eval_full={len(eval_full)}"
            )
        if train_val_subset is not None:
            val_dataset = Subset(eval_full, list(train_val_subset.indices))
            logger.info(
                "Validation uses augment=False and evenly spaced slices "
                "(n_val=%d)",
                len(val_dataset),
            )
        if train_test_subset is not None:
            test_dataset = Subset(eval_full, list(train_test_subset.indices))
            logger.info(
                "Test uses augment=False and evenly spaced slices "
                "(n_test=%d)",
                len(test_dataset),
            )

    train_labels = labels[train_dataset.indices]
    train_counts = summarize_class_counts(
        train_labels, name=f"train ({unit_name})", class_names=class_names
    )
    if val_dataset is not None:
        summarize_class_counts(
            labels[val_dataset.indices],
            name=f"val ({unit_name})",
            class_names=class_names,
        )
    if test_dataset is not None:
        summarize_class_counts(
            labels[test_dataset.indices],
            name=f"test ({unit_name})",
            class_names=class_names,
        )

    class_weights: Optional[torch.Tensor] = None
    if not binary:
        class_weights = inverse_frequency_weights(
            train_counts, num_classes=num_classes
        ).to(device)
        logger.info(
            "ADNI class weights (inverse frequency): %s",
            {
                class_names[index]: float(class_weights[index])
                for index in range(num_classes)
            },
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if test_dataset is not None
        else None
    )

    encoder = load_encoder(
        args.encoder,
        device=device,
        weights=args.weights,
        repo_dir=args.dinov3_repo,
        model_name=args.model_name,
    )
    model = MultiSliceDinoModel(
        encoder,
        n_slices=args.n_slices,
        features=args.features,
        d_model=args.d_model,
        depth=args.mst_depth,
        n_heads=args.mst_heads,
        ffn_dim=args.mst_ffn_dim,
        dropout=args.mst_dropout,
        hidden_dim=args.hidden_dim or args.d_model,
        num_classes=num_classes,
    ).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        if args.cosine_lr
        else None
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    best_path = checkpoint_dir / "best_mst.pt"
    epochs_trained = 0
    stopped_early = False
    loss_tag = "bce" if binary else "ce"

    logger.info(
        "dataset=%s adni_task=%s encoder=%s features=%s n_slices=%d num_classes=%d "
        "train_%s=%d val_%s=%d test_%s=%d embed_dim=%d d_model=%d loss=%s",
        args.dataset,
        args.adni_task if args.dataset == "adni" else None,
        args.encoder,
        args.features,
        args.n_slices,
        num_classes,
        unit_name,
        len(train_dataset),
        unit_name,
        len(val_dataset) if val_dataset is not None else 0,
        unit_name,
        len(test_dataset) if test_dataset is not None else 0,
        int(encoder.embed_dim),
        args.d_model,
        loss_name,
    )
    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        model.encoder.eval()
        running_loss = 0.0
        correct = 0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if binary:
                targets = targets.to(device, non_blocking=True).float()
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                predictions = (logits >= 0).float()
                correct += int((predictions == targets).sum().item())
            else:
                targets = targets.to(device, non_blocking=True).long()
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                correct += int((logits.argmax(dim=-1) == targets).sum().item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_items = int(targets.numel())
            running_loss += float(loss.item()) * n_items
            seen += n_items

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)
        if val_loader is None:
            logger.info(
                "epoch %d/%d lr=%.6g train_%s=%.5f train_acc=%.3f",
                epoch,
                args.epochs,
                current_lr,
                loss_tag,
                train_loss,
                train_accuracy,
            )
            if scheduler is not None:
                scheduler.step()
            continue

        val_loss, val_f1 = evaluate(
            model, val_loader, device, class_weights=class_weights
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model,
                    epoch=epoch,
                    args=args,
                    class_names=class_names,
                    loss_name=loss_name,
                    optimizer=optimizer,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    class_weights=(
                        class_weights.detach().cpu().tolist()
                        if class_weights is not None
                        else None
                    ),
                ),
                best_path,
            )
        logger.info(
            "epoch %d/%d lr=%.6g train_%s=%.5f train_acc=%.3f "
            "val_%s=%.5f val_f1=%.3f%s",
            epoch,
            args.epochs,
            current_lr,
            loss_tag,
            train_loss,
            train_accuracy,
            loss_tag,
            val_loss,
            val_f1,
            " *" if improved else "",
        )
        epochs_without_improvement = epoch - best_epoch
        if should_early_stop(
            epoch=epoch,
            best_epoch=best_epoch,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
        ):
            stopped_early = True
            logger.info(
                "Early stopping at epoch %d: validation loss has not decreased "
                "for %d epochs (best epoch=%d, best val_%s=%.5f)",
                epoch,
                epochs_without_improvement,
                best_epoch,
                loss_tag,
                best_val,
            )
            break
        if scheduler is not None:
            scheduler.step()

    if stopped_early:
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_trainable_state_dict(best["model"])
        optimizer.load_state_dict(best["optimizer"])
        logger.info("Restored best model from epoch %d", best["epoch"])

    last_path = checkpoint_dir / "last_mst.pt"
    torch.save(
        _checkpoint_payload(
            model,
            epoch=epochs_trained,
            args=args,
            class_names=class_names,
            loss_name=loss_name,
            optimizer=optimizer,
            best_val_loss=best_val if val_loader is not None else None,
            best_epoch=best_epoch if val_loader is not None else None,
            stopped_early=stopped_early,
            class_weights=(
                class_weights.detach().cpu().tolist()
                if class_weights is not None
                else None
            ),
        ),
        last_path,
    )
    logger.info("Wrote %s", last_path)

    if val_loader is not None and best_path.is_file():
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_trainable_state_dict(best["model"])
        y_true, y_pred, y_probability = collect_predictions(model, val_loader, device)
        auroc = compute_auroc(y_true, y_probability, num_classes=num_classes)
        final_f1 = classification_f1(y_true, y_pred, num_classes=num_classes)
        confusion = save_confusion_matrix(
            y_true,
            y_pred,
            checkpoint_dir / "val_confusion_matrix.png",
            class_names=class_names,
        )
        logger.info(
            "Best-checkpoint val metrics: n=%d f1=%.4f auroc=%.4f",
            len(y_true),
            final_f1,
            auroc,
        )
        logger.info("Val confusion matrix:\n%s", confusion)

    if test_loader is not None:
        if best_path.is_file():
            best = torch.load(best_path, map_location=device, weights_only=False)
            model.load_trainable_state_dict(best["model"])
        y_true, y_pred, y_probability = collect_predictions(model, test_loader, device)
        auroc = compute_auroc(y_true, y_probability, num_classes=num_classes)
        final_f1 = classification_f1(y_true, y_pred, num_classes=num_classes)
        confusion = save_confusion_matrix(
            y_true,
            y_pred,
            checkpoint_dir / "test_confusion_matrix.png",
            class_names=class_names,
        )
        logger.info(
            "Best-checkpoint test metrics: n=%d f1=%.4f auroc=%.4f",
            len(y_true),
            final_f1,
            auroc,
        )
        logger.info("Test confusion matrix:\n%s", confusion)


def _yaml_defaults(path: Path, parser: argparse.ArgumentParser) -> Dict[str, object]:
    """Load and type-check parser defaults from a YAML mapping."""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"could not read YAML params file {path}: {exc}")

    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        parser.error(f"YAML params file {path} must contain a top-level mapping")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "params_file"}
    }
    defaults: Dict[str, object] = {}
    for raw_key, value in raw_config.items():
        if not isinstance(raw_key, str):
            parser.error(f"YAML parameter names must be strings, got {raw_key!r}")
        key = raw_key.replace("-", "_")
        if key in defaults:
            parser.error(f"duplicate YAML parameter after normalization: {raw_key!r}")
        action = actions.get(key)
        if action is None:
            parser.error(f"unknown YAML parameter: {raw_key!r}")
        is_boolean_flag = action.nargs == 0 and isinstance(action.const, bool)
        if is_boolean_flag:
            if not isinstance(value, bool):
                parser.error(f"YAML parameter {raw_key!r} must be true or false")
        elif value is not None and action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                parser.error(f"invalid value for YAML parameter {raw_key!r}: {exc}")
        if action.choices is not None and value not in action.choices:
            parser.error(
                f"invalid value for YAML parameter {raw_key!r}: {value!r} "
                f"(choose from {', '.join(map(str, action.choices))})"
            )
        defaults[key] = value
    return defaults


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-file",
        "--config",
        type=Path,
        default=None,
        help="YAML file containing argument values; explicit CLI arguments override it",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CHOICES),
        default="duke",
        help="duke: binary breast cancer; adni: diagnosis task selected by --adni-task",
    )
    parser.add_argument(
        "--adni-task",
        choices=list(ADNI_TASK_CHOICES),
        default=DEFAULT_ADNI_TASK,
        help=(
            "ADNI label configuration (ignored for Duke). "
            "cn_mci_ad: three-class CE; cn_ad/cn_mci/mci_ad: binary BCE like Duke "
            "(second diagnosis is the positive class)"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (defaults to the processed Duke or ADNI path)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="ADNI metadata CSV (defaults to the CSV inside --data-root)",
    )
    parser.add_argument("--scan", type=str, default="pre")
    parser.add_argument("--z-min", type=float, default=0)
    parser.add_argument("--z-max", type=float, default=1)
    parser.add_argument("--n-slices", type=int, default=8)
    parser.add_argument(
        "--minimum-z-index-distance",
        "--min-slices-dist",
        dest="minimum_z_index_distance",
        type=int,
        default=None,
        help=(
            "Minimum index difference between sampled slices. "
            "Default: floor(eligible_slices / n_slices) - 1 per volume"
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--include-bilateral",
        action="store_true",
        help="Include bilateral Duke cases (excluded by default; ignored for ADNI)",
    )
    augmentation = parser.add_mutually_exclusive_group()
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
    parser.add_argument("--crop-scale-min", type=float, default=0.8)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument(
        "--vertical-flip-prob",
        type=float,
        default=None,
        help="Vertical flip probability (default: 0.5 for Duke, 0.0 for ADNI)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=30,
        help="Minimum epochs before validation-loss early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without lower validation loss",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help="Anneal the learning rate with a cosine schedule (disabled by default)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Final learning rate for --cosine-lr (default: 0)",
    )
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--test-frac",
        type=float,
        default=None,
        help=(
            "Patient-level held-out test fraction (never used for training or "
            "early stopping). Defaults to --val-frac"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--mst-depth", type=int, default=2)
    parser.add_argument("--mst-heads", type=int, default=12)
    parser.add_argument("--mst-ffn-dim", type=int, default=3072)
    parser.add_argument("--mst-dropout", type=float, default=0.1)
    parser.add_argument(
        "--encoder", choices=list(ENCODER_CHOICES), default="dinov3"
    )
    parser.add_argument(
        "--features",
        choices=list(FEATURE_CHOICES),
        default="cls",
        help="Per-slice token: CLS, attention-pooled patches, or both",
    )
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--params-file", "--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.params_file is not None:
        parser.set_defaults(**_yaml_defaults(config_args.params_file, parser))
    args = parser.parse_args(argv)
    if args.data_root is None:
        args.data_root = (
            ADNI_DEFAULT_ROOT if args.dataset == "adni" else DEFAULT_DATA_ROOT
        )
    if args.vertical_flip_prob is None:
        args.vertical_flip_prob = 0.0 if args.dataset == "adni" else 0.5
    if args.test_frac is None:
        args.test_frac = args.val_frac
    if args.n_slices <= 0:
        parser.error("--n-slices must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.min_epochs < 0:
        parser.error("--min-epochs must be non-negative")
    if args.early_stopping_patience <= 0:
        parser.error("--early-stopping-patience must be positive")
    if args.lr < 0:
        parser.error("--lr must be non-negative")
    if args.min_lr < 0:
        parser.error("--min-lr must be non-negative")
    if args.cosine_lr and args.min_lr > args.lr:
        parser.error("--min-lr cannot exceed --lr when --cosine-lr is enabled")
    if (
        args.minimum_z_index_distance is not None
        and args.minimum_z_index_distance < 0
    ):
        parser.error("--minimum-z-index-distance must be non-negative")
    if args.mst_depth <= 0:
        parser.error("--mst-depth must be positive")
    if args.mst_heads <= 0 or args.d_model % args.mst_heads:
        parser.error("--d-model must be divisible by positive --mst-heads")
    if not 0.0 <= args.vertical_flip_prob <= 1.0:
        parser.error("--vertical-flip-prob must be in [0, 1]")
    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val-frac must be in [0, 1)")
    if not 0.0 <= args.test_frac < 1.0:
        parser.error("--test-frac must be in [0, 1)")
    if args.val_frac + args.test_frac >= 1.0:
        parser.error("--val-frac + --test-frac must be < 1")
    return args


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
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else REPO_ROOT / "runs" / (args.run_name or f"{args.encoder}_mst_{args.dataset}")
    )
    logger.info("Run output directory: %s", checkpoint_dir)
    train(args, device, Path(checkpoint_dir))


if __name__ == "__main__":
    main()
