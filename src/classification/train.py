"""Frozen DINOv3 encoder + multi-slice transformer (or mean-pool) classification."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
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
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dinov3_baseline import (  # noqa: E402
    CLASS_NAMES as DUKE_CLASS_NAMES,
    DINOV3_REPO,
    ENCODER_CHOICES,
    FEATURE_CHOICES,
    REPO_ROOT,
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
    build_adni_volume_transform,
)
from datasets.duke import (  # noqa: E402
    DukeMultiSliceDataset,
    build_duke_volume_transform,
)
from ssl_finetuning.splits import patient_level_stratified_split  # noqa: E402
from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from utils.vit_lora import LoRA, add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402
from classification.model import MultiSliceDinoModel, set_lora_requires_grad  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DATASET_CHOICES = ("duke", "adni")
AGGREGATOR_CHOICES = ("transformer", "mean")
ENCODER_TRAINING_CHOICES = ("frozen", "lora")
EARLY_STOPPING_METRIC_CHOICES = ("bce_loss", "f1", "auroc")
EARLY_STOPPING_MIN_IMPROVEMENT = 0.005
MultiSliceDataset = Union[DukeMultiSliceDataset, ADNIMultiSliceDataset]
LORA_PARAMETER_NAMES = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")

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
    title: str = "Confusion matrix",
) -> np.ndarray:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    display.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title)
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


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, float]:
    """Compute scalar classification metrics for a held-out split."""
    if len(y_true) == 0:
        return {
            "f1": float("nan"),
            "auroc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    labels = list(range(num_classes if not is_binary_task(num_classes) else 2))
    if is_binary_task(num_classes):
        precision = float(
            precision_score(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
        )
        recall = float(
            recall_score(
                y_true, y_pred, average="binary", pos_label=1, zero_division=0
            )
        )
    else:
        precision = float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        )
    return {
        "f1": classification_f1(y_true, y_pred, num_classes=num_classes),
        "auroc": compute_auroc(y_true, y_probability, num_classes=num_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
    }


def save_auroc_plot(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    out_path: Path,
    *,
    num_classes: int,
    class_names: Sequence[str],
    title: str = "ROC curve",
) -> None:
    """Save a binary or one-vs-rest multiclass ROC curve plot."""
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        logger.warning("Skipping AUROC plot (need ≥2 classes in targets): %s", out_path)
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")

    if is_binary_task(num_classes):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)
        curve_auc = float(auc(false_positive_rate, true_positive_rate))
        positive_name = class_names[1] if len(class_names) > 1 else "positive"
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{positive_name} (AUC = {curve_auc:.3f})",
        )
    else:
        # One-vs-rest ROC for each class, plus a macro-average curve.
        n_classes = int(num_classes)
        mean_false_positive_rate = np.linspace(0, 1, 101)
        true_positive_rates: List[np.ndarray] = []
        for class_index, class_name in enumerate(class_names):
            binary_true = (y_true == class_index).astype(np.int64)
            if np.unique(binary_true).size < 2:
                continue
            scores = y_probability[:, class_index]
            false_positive_rate, true_positive_rate, _ = roc_curve(binary_true, scores)
            curve_auc = float(auc(false_positive_rate, true_positive_rate))
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=1.5,
                label=f"{class_name} (AUC = {curve_auc:.3f})",
            )
            interpolated = np.interp(
                mean_false_positive_rate, false_positive_rate, true_positive_rate
            )
            true_positive_rates.append(interpolated)
        if true_positive_rates:
            mean_true_positive_rate = np.mean(np.stack(true_positive_rates, axis=0), axis=0)
            macro_auc = float(auc(mean_false_positive_rate, mean_true_positive_rate))
            ax.plot(
                mean_false_positive_rate,
                mean_true_positive_rate,
                color="black",
                linewidth=2,
                label=f"macro-average (AUC = {macro_auc:.3f})",
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote AUROC plot → %s", out_path)


def _json_safe(value: object) -> object:
    """Convert values to JSON-serializable forms (Paths → str, NaN → null)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def args_to_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {key: _json_safe(value) for key, value in vars(args).items()}


def save_run_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    metrics_by_split: Dict[str, Dict[str, object]],
) -> None:
    """Write run parameters and split metrics to a JSON summary file."""
    payload = {
        "parameters": args_to_dict(args),
        "metrics": _json_safe(metrics_by_split),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("Wrote run summary → %s", out_path)


def should_early_stop(
    *,
    epoch: int,
    best_epoch: int,
    min_epochs: int,
    patience: int,
) -> bool:
    """Return whether the selected validation metric has failed to improve."""
    return (
        best_epoch > 0
        and epoch >= min_epochs
        and epoch - best_epoch >= patience
    )


def _duke_patient_stratum(dataset: DukeMultiSliceDataset, index: int) -> Optional[int]:
    return patient_strata(dataset.get_phenotype_raw(index))


def _adni_patient_stratum(dataset: ADNIMultiSliceDataset, index: int) -> int:
    return int(dataset.get_target(index))



def _dataset_patient_ids(dataset: Dataset) -> set[str]:
    """Return the patient IDs represented by a dataset."""
    get_patient_id = getattr(dataset, "get_patient_id", None)
    if get_patient_id is None:
        raise TypeError(
            f"Dataset {type(dataset).__name__} must provide get_patient_id()"
        )
    return {str(get_patient_id(index)) for index in range(len(dataset))}


def _require_dataset_patient_ids(
    dataset: Dataset,
    expected_patient_ids: Sequence[str],
    *,
    split_name: str,
) -> None:
    """Fail loudly if a split dataset contains anything but the requested patients."""
    actual_patients = _dataset_patient_ids(dataset)
    expected_patients = {str(patient_id) for patient_id in expected_patient_ids}
    if actual_patients != expected_patients:
        raise RuntimeError(
            f"{split_name} dataset patient IDs do not match requested split: "
            f"missing={sorted(expected_patients - actual_patients)[:5]}, "
            f"extra={sorted(actual_patients - expected_patients)[:5]}"
        )


def _assert_loader_patient_disjoint(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
) -> None:
    """Assert that no patient is represented in more than one split loader."""
    train_patients = _dataset_patient_ids(train_loader.dataset)
    val_patients = (
        _dataset_patient_ids(val_loader.dataset) if val_loader is not None else set()
    )
    test_patients = (
        _dataset_patient_ids(test_loader.dataset) if test_loader is not None else set()
    )
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    if train_val_overlap:
        raise RuntimeError(
            "Patient leakage between train and validation loaders: "
            f"{sorted(train_val_overlap)[:5]}"
        )
    if train_test_overlap:
        raise RuntimeError(
            "Patient leakage between train and test loaders: "
            f"{sorted(train_test_overlap)[:5]}"
        )
    if val_test_overlap:
        raise RuntimeError(
            "Patient leakage between validation and test loaders: "
            f"{sorted(val_test_overlap)[:5]}"
        )
    logger.info(
        "Verified mutually exclusive loader patients: train=%d val=%d test=%d",
        len(train_patients),
        len(val_patients),
        len(test_patients),
    )


@torch.no_grad()
def evaluate(
    model: MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
    *,
    class_weights: Optional[torch.Tensor] = None,
    bce_pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Return mean validation loss, F1, and AUROC."""
    model.eval()
    total_loss = 0.0
    total = 0
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    binary = is_binary_task(model.num_classes)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if binary:
            targets = targets.to(device, non_blocking=True).float()
            if bce_pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=bce_pos_weight
                )
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()
            target_labels = targets.long()
        else:
            targets = targets.to(device, non_blocking=True).long()
            loss = F.cross_entropy(logits, targets, weight=class_weights)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            target_labels = targets
        n_items = int(targets.numel())
        total_loss += float(loss.item()) * n_items
        total += n_items
        targets_all.append(target_labels.cpu().numpy())
        predictions_all.append(predictions.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())

    if total == 0:
        return float("nan"), float("nan"), float("nan")
    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(predictions_all)
    y_probability = np.concatenate(probabilities_all)
    return (
        total_loss / total,
        classification_f1(y_true, y_pred, num_classes=model.num_classes),
        compute_auroc(y_true, y_probability, num_classes=model.num_classes),
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
        "encoder_training": args.encoder_training,
        "lora_r": args.lora_r if args.encoder_training == "lora" else None,
        "freeze_epochs": args.freeze_epochs,
        "model_name": args.model_name,
        "weights": str(args.weights) if args.weights is not None else None,
        "features": args.features,
        "n_cls_tokens": args.n_cls_tokens,
        "slice_aggregator": args.slice_aggregator,
        "weight_ce_loss": args.weight_ce_loss,
        "early_stopping": args.early_stopping,
        "early_stopping_metric": args.early_stopping_metric,
        "n_slices": args.n_slices,
        "include_bilateral": args.include_bilateral,
        "image_size": args.image_size,
        "augment": args.augment,
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
    patient_ids: Optional[Sequence[str]] = None,
) -> MultiSliceDataset:
    use_augment = args.augment if augment is None else bool(augment)
    if args.dataset == "duke":
        return DukeMultiSliceDataset(
            root=args.data_root,
            n_slices=args.n_slices,
            include_bilateral=args.include_bilateral,
            scan=args.scan,
            patient_ids=patient_ids,
            augment=use_augment,
            image_size=args.image_size,
            transform=build_duke_volume_transform(augment=use_augment),
        )
    if args.dataset == "adni":
        return ADNIMultiSliceDataset(
            root=args.data_root,
            csv_path=args.csv_path,
            task=args.adni_task,
            n_slices=args.n_slices,
            patient_ids=patient_ids,
            augment=use_augment,
            image_size=args.image_size,
            transform=build_adni_volume_transform(augment=use_augment),
        )
    raise ValueError(f"Unknown dataset={args.dataset!r}")


def train(args: argparse.Namespace, device: torch.device, checkpoint_dir: Path) -> None:
    torch.manual_seed(args.seed)
    num_classes, class_names, loss_name = task_config(
        args.dataset, adni_task=args.adni_task
    )
    binary = is_binary_task(num_classes)
    unit_name = "breasts" if args.dataset == "duke" else "scans"
    # Generate or load patient IDs once, then construct each split directly.
    split_dataset = build_dataset(args, augment=False)
    labels = collect_labels(split_dataset)
    summarize_class_counts(labels, name=f"full dataset ({unit_name})", class_names=class_names)

    get_stratum: Callable[[Any, int], Optional[int]] = (
        _duke_patient_stratum if args.dataset == "duke" else _adni_patient_stratum
    )
    _, split_metadata = patient_level_stratified_split(
        split_dataset,
        dataset_name=args.dataset,
        train_fraction=1.0 - args.val_frac - args.test_frac,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        stratum_fn=get_stratum,
        split_file=args.splits_file,
        use_saved_split_config=False,
        # A saved ADNI split can contain all diagnoses. The active task uses
        # only its allowed diagnoses, so discard the other patient IDs while
        # still requiring every task-eligible patient to be assigned once.
        allow_saved_patient_superset=args.dataset == "adni",
        # ADNI task label IDs depend on the pair of diagnoses, whereas the
        # patient assignments remain valid across task-specific subsets.
        validate_saved_patient_strata=args.dataset != "adni",
    )
    split_patients = split_metadata["splits"]
    train_patient_ids = [str(patient_id) for patient_id in split_patients["train"]]
    val_patient_ids = [str(patient_id) for patient_id in split_patients["val"]]
    test_patient_ids = [str(patient_id) for patient_id in split_patients["test"]]
    if args.dataset == "adni":
        task_patient_ids = _dataset_patient_ids(split_dataset)
        split_patient_lists = (train_patient_ids, val_patient_ids, test_patient_ids)
        excluded_patients = sum(
            patient_id not in task_patient_ids
            for patient_ids in split_patient_lists
            for patient_id in patient_ids
        )
        train_patient_ids = [
            patient_id for patient_id in train_patient_ids if patient_id in task_patient_ids
        ]
        val_patient_ids = [
            patient_id for patient_id in val_patient_ids if patient_id in task_patient_ids
        ]
        test_patient_ids = [
            patient_id for patient_id in test_patient_ids if patient_id in task_patient_ids
        ]
        if excluded_patients:
            logger.info(
                "Excluded %d patients outside ADNI task %s from the saved split",
                excluded_patients,
                resolve_adni_task(args.adni_task).name,
            )

    train_dataset = build_dataset(
        args, augment=args.augment, patient_ids=train_patient_ids
    )
    val_dataset: Optional[MultiSliceDataset] = (
        build_dataset(args, augment=False, patient_ids=val_patient_ids)
        if val_patient_ids
        else None
    )
    test_dataset: Optional[MultiSliceDataset] = (
        build_dataset(args, augment=False, patient_ids=test_patient_ids)
        if test_patient_ids
        else None
    )
    _require_dataset_patient_ids(
        train_dataset, train_patient_ids, split_name="train"
    )
    if val_dataset is not None:
        _require_dataset_patient_ids(
            val_dataset, val_patient_ids, split_name="validation"
        )
    if test_dataset is not None:
        _require_dataset_patient_ids(
            test_dataset, test_patient_ids, split_name="test"
        )
    logger.info(
        "Datasets: train=%d patients/%d %s, val=%d patients/%d %s, "
        "test=%d patients/%d %s",
        len(train_patient_ids),
        len(train_dataset),
        unit_name,
        len(val_patient_ids),
        len(val_dataset) if val_dataset is not None else 0,
        unit_name,
        len(test_patient_ids),
        len(test_dataset) if test_dataset is not None else 0,
        unit_name,
    )

    train_labels = collect_labels(train_dataset)
    train_counts = summarize_class_counts(
        train_labels, name=f"train ({unit_name})", class_names=class_names
    )
    if val_dataset is not None:
        summarize_class_counts(
            collect_labels(val_dataset),
            name=f"val ({unit_name})",
            class_names=class_names,
        )
    if test_dataset is not None:
        summarize_class_counts(
            collect_labels(test_dataset),
            name=f"test ({unit_name})",
            class_names=class_names,
        )

    class_weights: Optional[torch.Tensor] = None
    bce_pos_weight: Optional[torch.Tensor] = None
    if binary and args.weight_ce_loss:
        negative_count = train_counts.get(0, 0)
        positive_count = train_counts.get(1, 0)
        if negative_count <= 0 or positive_count <= 0:
            raise ValueError(
                "Cannot compute a weighted BCE loss without both binary classes "
                f"in the training split: negative={negative_count}, "
                f"positive={positive_count}"
            )
        bce_pos_weight = torch.tensor(
            [negative_count / positive_count], dtype=torch.float32, device=device
        )
        logger.info(
            "BCE positive-class weight: %.6f (negative=%d positive=%d)",
            float(bce_pos_weight.item()),
            negative_count,
            positive_count,
        )
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
    _assert_loader_patient_disjoint(train_loader, val_loader, test_loader)

    if args.weights is not None and args.encoder == "dinov3":
        encoder = load_custom_dinov3_encoder(
            checkpoint=Path(args.weights),
            repo_dir=args.dinov3_repo,
            device=device,
            encoder_training=args.encoder_training,
            lora_rank=args.lora_r,
        )
    else:
        encoder = load_encoder(
            args.encoder,
            device=device,
            weights=args.weights,
            repo_dir=args.dinov3_repo,
            model_name=args.model_name,
        )
        if args.encoder_training == "lora":
            add_lora_to_vit(encoder, r=args.lora_r)
            freeze_non_lora_parameters(encoder)
    if args.encoder_training == "lora" and args.freeze_epochs > 0:
        set_lora_requires_grad(encoder, False)
    model = MultiSliceDinoModel(
        encoder,
        n_slices=args.n_slices,
        features=args.features,
        n_cls_tokens=args.n_cls_tokens,
        aggregator=args.slice_aggregator,
        d_model=args.d_model,
        depth=args.mst_depth,
        n_heads=args.mst_heads,
        ffn_dim=args.mst_ffn_dim,
        dropout=args.mst_dropout,
        hidden_dim=args.hidden_dim or args.d_model,
        num_classes=num_classes,
        encoder_training=args.encoder_training,
    ).to(device)
    trainable_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if MultiSliceDinoModel._is_trainable_state_name(name)
    ]
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
    best_metric_value = (
        float("inf")
        if args.early_stopping_metric == "bce_loss"
        else float("-inf")
    )
    best_checkpoint_val_loss = float("inf")
    best_epoch = 0
    best_path = checkpoint_dir / "best_mst.pt"
    epochs_trained = 0
    stopped_early = False
    loss_tag = "bce" if binary else "ce"
    lora_unfrozen = not (
        args.encoder_training == "lora" and args.freeze_epochs > 0
    )

    logger.info(
        "dataset=%s adni_task=%s encoder=%s features=%s n_cls_tokens=%d aggregator=%s "
        "n_slices=%d "
        "encoder_training=%s lora_r=%s freeze_epochs=%d num_classes=%d "
        "train_%s=%d val_%s=%d test_%s=%d embed_dim=%d d_model=%d loss=%s",
        args.dataset,
        args.adni_task if args.dataset == "adni" else None,
        args.encoder,
        args.features,
        args.n_cls_tokens,
        args.slice_aggregator,
        args.n_slices,
        args.encoder_training,
        args.lora_r if args.encoder_training == "lora" else None,
        args.freeze_epochs,
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
        if (
            args.encoder_training == "lora"
            and not lora_unfrozen
            and epoch > args.freeze_epochs
        ):
            set_lora_requires_grad(model.encoder, True)
            lora_unfrozen = True
            logger.info("Unfroze LoRA adapters at epoch %d", epoch)
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
                if bce_pos_weight is None:
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=bce_pos_weight
                    )
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

        val_loss, val_f1, val_auroc = evaluate(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
        )
        validation_metrics = {
            "bce_loss": val_loss,
            "f1": val_f1,
            "auroc": val_auroc,
        }
        current_metric_value = validation_metrics[args.early_stopping_metric]
        if math.isnan(current_metric_value):
            improved = False
        elif args.early_stopping_metric == "bce_loss":
            improved = (
                current_metric_value
                < best_metric_value - EARLY_STOPPING_MIN_IMPROVEMENT
            )
        else:
            improved = (
                current_metric_value
                > best_metric_value + EARLY_STOPPING_MIN_IMPROVEMENT
            )
        if improved:
            best_metric_value = current_metric_value
            best_checkpoint_val_loss = val_loss
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
                    val_auroc=val_auroc,
                    class_weights=(
                        class_weights.detach().cpu().tolist()
                        if class_weights is not None
                        else None
                    ),
                    early_stopping_value=current_metric_value,
                ),
                best_path,
            )
        logger.info(
            "epoch %d/%d lr=%.6g train_%s=%.5f train_acc=%.3f "
            "val_%s=%.5f val_f1=%.3f val_auroc=%.3f "
            "early_stopping_%s=%.5f%s",
            epoch,
            args.epochs,
            current_lr,
            loss_tag,
            train_loss,
            train_accuracy,
            loss_tag,
            val_loss,
            val_f1,
            val_auroc,
            args.early_stopping_metric,
            current_metric_value,
            " *" if improved else "",
        )
        epochs_without_improvement = epoch - best_epoch
        if args.early_stopping and should_early_stop(
            epoch=epoch,
            best_epoch=best_epoch,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
        ):
            stopped_early = True
            logger.info(
                "Early stopping at epoch %d: validation %s has not improved "
                "for %d epochs (best epoch=%d, best value=%.5f)",
                epoch,
                args.early_stopping_metric,
                epochs_without_improvement,
                best_epoch,
                best_metric_value,
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
            best_val_loss=(
                best_checkpoint_val_loss if val_loader is not None else None
            ),
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

    metrics_by_split: Dict[str, Dict[str, object]] = {}

    if val_loader is not None:
        if best_path.is_file():
            best = torch.load(best_path, map_location=device, weights_only=False)
            model.load_trainable_state_dict(best["model"])
            logger.info("Evaluating final validation metrics from best epoch %d", best["epoch"])
        y_true, y_pred, y_probability = collect_predictions(model, val_loader, device)
        final_val_loss, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
        )
        val_metrics = compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=num_classes
        )
        confusion = save_confusion_matrix(
            y_true,
            y_pred,
            checkpoint_dir / "val_confusion_matrix.png",
            class_names=class_names,
            title="Validation confusion matrix",
        )
        metrics_by_split["val"] = {
            "n": int(len(y_true)),
            "loss": final_val_loss,
            **val_metrics,
        }
        logger.info(
            "Final val metrics: n=%d loss=%.5f f1=%.4f auroc=%.4f "
            "acc=%.4f precision=%.4f recall=%.4f",
            len(y_true),
            final_val_loss,
            val_metrics["f1"],
            val_metrics["auroc"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
        )
        logger.info("Val confusion matrix:\n%s", confusion)

    if test_loader is not None:
        if best_path.is_file():
            best = torch.load(best_path, map_location=device, weights_only=False)
            model.load_trainable_state_dict(best["model"])
        y_true, y_pred, y_probability = collect_predictions(model, test_loader, device)
        final_test_loss, _, _ = evaluate(
            model,
            test_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
        )
        test_metrics = compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=num_classes
        )
        confusion = save_confusion_matrix(
            y_true,
            y_pred,
            checkpoint_dir / "test_confusion_matrix.png",
            class_names=class_names,
            title="Test confusion matrix",
        )
        save_auroc_plot(
            y_true,
            y_probability,
            checkpoint_dir / "test_auroc.png",
            num_classes=num_classes,
            class_names=class_names,
            title="Test ROC curve",
        )
        metrics_by_split["test"] = {
            "n": int(len(y_true)),
            "loss": final_test_loss,
            **test_metrics,
        }
        logger.info(
            "Final test metrics: n=%d loss=%.5f f1=%.4f auroc=%.4f "
            "acc=%.4f precision=%.4f recall=%.4f",
            len(y_true),
            final_test_loss,
            test_metrics["f1"],
            test_metrics["auroc"],
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
        )
        logger.info("Test confusion matrix:\n%s", confusion)

    save_run_summary(
        checkpoint_dir / "run_summary.json",
        args=args,
        metrics_by_split=metrics_by_split,
    )
