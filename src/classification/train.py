"""Frozen DINOv3 encoder + multi-slice transformer (or mean-pool) classification."""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
from torch.utils.data import DataLoader, Dataset, default_collate

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.load_dinov3 import (  # noqa: E402
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
from datasets.cq500 import (  # noqa: E402
    CQ500MultiSliceDataset,
    build_cq500_volume_transform,
    resolve_cq500_task,
)
from datasets.organmnist3d import (  # noqa: E402
    ORGANMNIST3D_CLASS_NAMES,
    OrganMNIST3DMultiSliceDataset,
    build_organmnist3d_volume_transform,
)
from datasets.breastdm import (  # noqa: E402
    BREASTDM_CLASS_NAMES,
    BreastDMMultiSliceDataset,
    DEFAULT_ROOT as BREASTDM_DEFAULT_ROOT,
    build_breastdm_volume_transform,
)
from utils.fold_cv import make_dataset_patient_folds  # noqa: E402
from utils.splits import patient_level_stratified_split  # noqa: E402
from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from utils.vit_lora import LoRA, add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402
from classification.model import MultiSliceDinoModel, set_lora_requires_grad  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DATASET_CHOICES = ("duke", "adni", "cq500", "organmnist3d", "breastdm")
AGGREGATOR_CHOICES = ("transformer", "mean")
ENCODER_TRAINING_CHOICES = ("frozen", "lora")
EARLY_STOPPING_METRIC_CHOICES = ("bce_loss", "f1", "auroc")
EARLY_STOPPING_MIN_IMPROVEMENT = 0.005
MultiSliceDataset = Union[
    DukeMultiSliceDataset,
    ADNIMultiSliceDataset,
    CQ500MultiSliceDataset,
    OrganMNIST3DMultiSliceDataset,
    BreastDMMultiSliceDataset,
]
LORA_PARAMETER_NAMES = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")


class EpochProfiler:
    """Accumulate low-overhead, GPU-accurate timings for one training epoch.

    CUDA kernels are asynchronous, so CUDA events are deliberately used for
    compute sections.  The DataLoader wait is measured on the host because it
    represents the time the training loop could not obtain the next batch.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._cuda = device.type == "cuda"
        self._cpu_seconds: Dict[str, float] = {}
        self._cuda_events: Dict[
            str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a CPU section or its CUDA work, depending on the active device."""
        if self._cuda:
            with torch.cuda.device(self.device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    yield
                finally:
                    end.record()
            self._cuda_events.setdefault(name, []).append((start, end))
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_seconds[name] = self._cpu_seconds.get(name, 0.0) + (
                time.perf_counter() - start_time
            )

    def add_host_time(self, name: str, seconds: float) -> None:
        """Add a wall-clock duration, such as waiting for the next batch."""
        self._cpu_seconds[name] = self._cpu_seconds.get(name, 0.0) + seconds

    def finish(self) -> Dict[str, float]:
        """Synchronize once, then return accumulated timing values in seconds."""
        if self._cuda:
            torch.cuda.synchronize(self.device)
            timings = dict(self._cpu_seconds)
            for name, events in self._cuda_events.items():
                timings[name] = sum(
                    start.elapsed_time(end) / 1_000.0 for start, end in events
                )
            return timings
        return dict(self._cpu_seconds)


def _process_peak_rss_mb() -> float:
    """Return process peak RSS in MiB (a process-wide, not per-epoch, value)."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.  The project runs on Linux in
    # production, but accepting both makes local runs less surprising.
    if sys.platform == "darwin":
        return float(peak_rss) / (1024.0 * 1024.0)
    return float(peak_rss) / 1024.0


def _gpu_memory_metrics(device: torch.device) -> Dict[str, float]:
    """Return allocator state and epoch peak GPU memory in MiB, if applicable."""
    if device.type != "cuda":
        return {}
    to_mebibytes = 1024.0 * 1024.0
    return {
        "gpu_memory_allocated_mb_end": (
            torch.cuda.memory_allocated(device) / to_mebibytes
        ),
        "gpu_memory_reserved_mb_end": (
            torch.cuda.memory_reserved(device) / to_mebibytes
        ),
        "gpu_memory_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / to_mebibytes
        ),
        "gpu_memory_peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / to_mebibytes
        ),
    }

def is_binary_task(num_classes: int) -> bool:
    """Duke uses a single logit (``num_classes=1``) with BCE."""
    return int(num_classes) == 1


def task_config(
    dataset_name: str,
    *,
    adni_task: str = DEFAULT_ADNI_TASK,
    cq500_task: str = "ich",
) -> Tuple[int, Tuple[str, ...], str]:
    """Return ``(num_logits, class_names, loss_name)`` for a dataset/task."""
    if dataset_name == "duke":
        # Single logit + BCE; class_names remain the binary labels used in metrics.
        return 1, DUKE_CLASS_NAMES, "BCEWithLogitsLoss"
    if dataset_name == "adni":
        spec = resolve_adni_task(adni_task)
        loss_name = "BCEWithLogitsLoss" if spec.binary else "CrossEntropyLoss"
        return spec.num_logits, spec.class_names, loss_name
    if dataset_name == "cq500":
        spec = resolve_cq500_task(cq500_task)
        return spec.num_logits, spec.class_names, "BCEWithLogitsLoss"
    if dataset_name == "organmnist3d":
        return len(ORGANMNIST3D_CLASS_NAMES), ORGANMNIST3D_CLASS_NAMES, "CrossEntropyLoss"
    if dataset_name == "breastdm":
        return 1, BREASTDM_CLASS_NAMES, "BCEWithLogitsLoss"
    raise ValueError(f"Unknown dataset={dataset_name!r}")


def is_multilabel_task(dataset_name: str, *, cq500_task: str = "ich") -> bool:
    """Return whether a task has independent binary labels per output logit."""
    return dataset_name == "cq500" and resolve_cq500_task(cq500_task).multi_label


def collect_labels(dataset: Dataset) -> np.ndarray:
    return np.asarray([dataset.get_target(index) for index in range(len(dataset))])  # type: ignore[attr-defined]


def collate_variable_depth_volumes(
    batch: Sequence[Tuple[torch.Tensor, Any]],
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    """Pad ``[D,C,H,W]`` volumes and return a mask for valid depth entries."""
    images, targets = zip(*batch)
    if not images:
        raise ValueError("Cannot collate an empty batch")
    if any(image.ndim != 4 for image in images):
        raise ValueError("Expected each variable-depth image to have shape [D,C,H,W]")
    channels_and_size = images[0].shape[1:]
    if any(image.shape[1:] != channels_and_size for image in images):
        raise ValueError("All variable-depth volumes must have matching [C,H,W]")
    depths = [int(image.shape[0]) for image in images]
    if min(depths) <= 0:
        raise ValueError("Every volume must contain at least one slice")
    padded = images[0].new_zeros((len(images), max(depths), *channels_and_size))
    slice_mask = torch.zeros((len(images), max(depths)), dtype=torch.bool)
    for index, image in enumerate(images):
        padded[index, : image.shape[0]] = image
        slice_mask[index, : image.shape[0]] = True
    return padded, default_collate(list(targets)), slice_mask


def _unpack_volume_batch(
    batch: Sequence[Any],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Accept the regular two-item batch or variable-depth three-item batch."""
    if len(batch) == 2:
        images, targets = batch
        return images, targets, None
    if len(batch) == 3:
        images, targets, slice_mask = batch
        return images, targets, slice_mask
    raise ValueError(f"Expected a 2- or 3-item batch, got {len(batch)} items")


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
    multi_label: bool = False,
) -> float:
    if multi_label:
        try:
            return float(roc_auc_score(y_true, y_probability, average="macro"))
        except ValueError as exc:
            logger.warning("AUROC undefined: %s", exc)
            return float("nan")
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
    multi_label: bool = False,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    if multi_label:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
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
    multi_label: bool = False,
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
    if multi_label:
        return {
            "f1": classification_f1(y_true, y_pred, num_classes=num_classes, multi_label=True),
            "auroc": compute_auroc(y_true, y_probability, num_classes=num_classes, multi_label=True),
            "accuracy": float((y_true == y_pred).mean()),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
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
    epoch_benchmarks: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    """Write run parameters and split metrics to a JSON summary file."""
    payload = {
        "parameters": args_to_dict(args),
        "metrics": _json_safe(metrics_by_split),
    }
    if epoch_benchmarks is not None:
        payload["epoch_benchmarks"] = _json_safe(epoch_benchmarks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("Wrote run summary → %s", out_path)


def save_epoch_benchmarks(
    out_path: Path, epoch_benchmarks: Sequence[Dict[str, object]]
) -> None:
    """Persist completed epoch measurements so an interrupted run keeps its data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"epoch_benchmarks": _json_safe(epoch_benchmarks)}, handle, indent=2)
        handle.write("\n")


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


def _cq500_patient_stratum(dataset: CQ500MultiSliceDataset, index: int) -> Union[int, Tuple[int, ...]]:
    """Return a hashable CQ500 label for patient-level fold stratification."""
    target = dataset.get_target(index)
    if isinstance(target, np.ndarray):
        return tuple(int(value) for value in target.tolist())
    return int(target)


def saved_split_patient_ids(
    args: argparse.Namespace, dataset: Dataset
) -> Tuple[list[str], list[str], list[str]]:
    """Load legacy explicit patient splits for a classification dataset."""
    splits_file = getattr(args, "splits_file", None)
    if splits_file is None:
        raise ValueError("saved_split_patient_ids requires args.splits_file")
    splits_file = Path(splits_file)
    if not splits_file.is_file():
        raise FileNotFoundError(f"Saved split file does not exist: {splits_file}")
    get_stratum = (
        _cq500_patient_stratum
        if args.dataset == "cq500"
        else _adni_patient_stratum
        if args.dataset == "adni"
        else _duke_patient_stratum
    )
    _, metadata = patient_level_stratified_split(
        dataset,
        dataset_name=args.dataset,
        # These fields are required by the legacy loader but are ignored when
        # loading the explicit patient assignments below.
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.data_seed,
        stratum_fn=get_stratum,
        split_file=splits_file,
        use_saved_split_config=False,
        # A saved ADNI split may include diagnoses excluded by the active task.
        allow_saved_patient_superset=args.dataset == "adni",
        validate_saved_patient_strata=args.dataset != "adni",
    )
    splits = metadata["splits"]
    return (
        [str(patient_id) for patient_id in splits["train"]],
        [str(patient_id) for patient_id in splits["val"]],
        [str(patient_id) for patient_id in splits["test"]],
    )



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
    multi_label: bool = False,
) -> Tuple[float, float, float]:
    """Return mean validation loss, F1, and AUROC."""
    model.eval()
    total_loss = 0.0
    total = 0
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    binary = is_binary_task(model.num_classes)
    for batch in loader:
        images, targets, slice_mask = _unpack_volume_batch(batch)
        images = images.to(device, non_blocking=True)
        logits = model(
            images,
            slice_mask=(
                slice_mask.to(device, non_blocking=True)
                if slice_mask is not None
                else None
            ),
        )
        if binary or multi_label:
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
        classification_f1(y_true, y_pred, num_classes=model.num_classes, multi_label=multi_label),
        compute_auroc(y_true, y_probability, num_classes=model.num_classes, multi_label=multi_label),
    )


@torch.no_grad()
def collect_predictions(
    model: MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
    *,
    multi_label: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return targets, predictions, and class probabilities."""
    model.eval()
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    binary = is_binary_task(model.num_classes)
    for batch in loader:
        images, targets, slice_mask = _unpack_volume_batch(batch)
        images = images.to(device, non_blocking=True)
        logits = model(
            images,
            slice_mask=(
                slice_mask.to(device, non_blocking=True)
                if slice_mask is not None
                else None
            ),
        )
        if binary or multi_label:
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
        "cq500_task": args.cq500_task if args.dataset == "cq500" else None,
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
        "benchmark": getattr(args, "benchmark", False),
        "early_stopping": args.early_stopping,
        "early_stopping_metric": args.early_stopping_metric,
        "n_slices": args.n_slices,
        "cq500_max_slices": getattr(args, "cq500_max_slices", 128),
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
        "n_folds": args.n_folds,
        "fold": args.fold,
        "data_seed": args.data_seed,
        "train_ratio": args.train_ratio,
        "splits_file": (
            str(args.splits_file)
            if getattr(args, "splits_file", None) is not None
            else None
        ),
        "cosine_lr": args.cosine_lr,
        "min_lr": args.min_lr,
        "lr_scheduler": (
            "reduce_on_plateau"
            if getattr(args, "reduce_lr_on_plateau", True)
            else "none"
        ),
        "reduce_lr_on_plateau": getattr(args, "reduce_lr_on_plateau", True),
        "lr_plateau_factor": getattr(args, "lr_plateau_factor", 0.1),
        "lr_plateau_patience": getattr(args, "lr_plateau_patience", 3),
        "lr_plateau_threshold": getattr(args, "lr_plateau_threshold", 0.005),
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
    split: Optional[str] = None,
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
    if args.dataset == "cq500":
        return CQ500MultiSliceDataset(
            root=args.data_root,
            csv_path=args.csv_path,
            task=args.cq500_task,
            n_slices=args.n_slices,
            cq500_max_slices=getattr(args, "cq500_max_slices", 128),
            patient_ids=patient_ids,
            augment=use_augment,
            image_size=args.image_size,
            transform=build_cq500_volume_transform(augment=use_augment),
        )
    if args.dataset == "organmnist3d":
        if patient_ids is not None:
            raise ValueError(
                "OrganMNIST3D uses its official train/val/test arrays and does not "
                "support patient-level filtering"
            )
        return OrganMNIST3DMultiSliceDataset(
            root=args.data_root,
            split=split or "train",
            n_slices=args.n_slices,
            augment=use_augment,
            image_size=args.image_size,
            transform=build_organmnist3d_volume_transform(augment=use_augment),
        )
    if args.dataset == "breastdm":
        return BreastDMMultiSliceDataset(
            root=args.data_root,
            split=split or "train",
            n_slices=args.n_slices,
            augment=use_augment,
            image_size=args.image_size,
            transform=build_breastdm_volume_transform(augment=use_augment),
        )
    raise ValueError(f"Unknown dataset={args.dataset!r}")


def train(args: argparse.Namespace, device: torch.device, checkpoint_dir: Path) -> None:
    torch.manual_seed(args.seed)
    cq500_task = getattr(args, "cq500_task", "ich")
    splits_file = getattr(args, "splits_file", None)
    if args.n_slices is None and args.dataset != "cq500":
        raise ValueError("n_slices=None is supported only for CQ500")
    num_classes, class_names, loss_name = task_config(
        args.dataset, adni_task=args.adni_task, cq500_task=cq500_task
    )
    binary = is_binary_task(num_classes)
    multi_label = is_multilabel_task(args.dataset, cq500_task=cq500_task)
    unit_name = {
        "duke": "breasts",
        "adni": "scans",
        "cq500": "volumes",
        "organmnist3d": "volumes",
        "breastdm": "volumes",
    }[args.dataset]
    if args.dataset in {"organmnist3d", "breastdm"}:
        train_dataset = build_dataset(args, augment=args.augment, split="train")
        val_dataset: Optional[MultiSliceDataset] = build_dataset(args, augment=False, split="val")
        test_dataset: Optional[MultiSliceDataset] = build_dataset(args, augment=False, split="test")
        logger.info(
            "Using supplied %s splits: train=%d %s, val=%d %s, test=%d %s",
            args.dataset,
            len(train_dataset), unit_name, len(val_dataset), unit_name,
            len(test_dataset), unit_name,
        )
    else:
        split_dataset = build_dataset(args, augment=False)
        labels = collect_labels(split_dataset)
        summarize_class_counts(
            labels, name=f"full dataset ({unit_name})", class_names=class_names
        )
        if splits_file is not None:
            train_patient_ids, val_patient_ids, test_patient_ids = saved_split_patient_ids(
                args, split_dataset
            )
        else:
            folds = make_dataset_patient_folds(
                split_dataset,
                n_folds=args.n_folds,
                seed=args.data_seed,
                target_fn=_cq500_patient_stratum if args.dataset == "cq500" else None,
            )
            train_patient_ids, val_patient_ids, test_patient_ids = folds.get_split(
                args.fold, train_ratio=args.train_ratio, train_seed=args.data_seed
            )
        train_dataset = build_dataset(args, augment=args.augment, patient_ids=train_patient_ids)
        val_dataset = build_dataset(args, augment=False, patient_ids=val_patient_ids)
        test_dataset = build_dataset(args, augment=False, patient_ids=test_patient_ids)
        _require_dataset_patient_ids(train_dataset, train_patient_ids, split_name="train")
        _require_dataset_patient_ids(val_dataset, val_patient_ids, split_name="validation")
        _require_dataset_patient_ids(test_dataset, test_patient_ids, split_name="test")
        if splits_file is not None:
            logger.info(
                "Saved split %s: train=%d patients/%d %s, val=%d patients/%d %s, "
                "test=%d patients/%d %s",
                splits_file,
                len(train_patient_ids), len(train_dataset), unit_name,
                len(val_patient_ids), len(val_dataset), unit_name,
                len(test_patient_ids), len(test_dataset), unit_name,
            )
        else:
            logger.info(
                "Fold %d/%d (seed=%d): train=%d patients/%d %s (ratio=%.3f), "
                "val=%d patients/%d %s, test=%d patients/%d %s",
                args.fold, args.n_folds, args.data_seed,
                len(train_patient_ids), len(train_dataset), unit_name, args.train_ratio,
                len(val_patient_ids), len(val_dataset), unit_name,
                len(test_patient_ids), len(test_dataset), unit_name,
            )

    train_labels = collect_labels(train_dataset)
    if multi_label:
        train_counts = {}
        logger.info(
            "train (%s) subtype positives: %s",
            unit_name,
            dict(zip(class_names, train_labels.sum(axis=0).astype(int).tolist())),
        )
    else:
        train_counts = summarize_class_counts(
            train_labels, name=f"train ({unit_name})", class_names=class_names
        )
        if val_dataset is not None:
            summarize_class_counts(
                collect_labels(val_dataset), name=f"val ({unit_name})", class_names=class_names
            )
        if test_dataset is not None:
            summarize_class_counts(
                collect_labels(test_dataset), name=f"test ({unit_name})", class_names=class_names
            )

    class_weights: Optional[torch.Tensor] = None
    bce_pos_weight: Optional[torch.Tensor] = None
    if multi_label and args.weight_ce_loss:
        positives = train_labels.sum(axis=0)
        negatives = len(train_labels) - positives
        if np.any(positives == 0) or np.any(negatives == 0):
            raise ValueError("Cannot compute weighted multi-label BCE without both values per subtype")
        bce_pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    elif binary and args.weight_ce_loss:
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
    if not binary and not multi_label:
        class_weights = inverse_frequency_weights(
            train_counts, num_classes=num_classes
        ).to(device)
        logger.info(
            "%s class weights (inverse frequency): %s",
            args.dataset,
            {
                class_names[index]: float(class_weights[index])
                for index in range(num_classes)
            },
        )

    collate_fn = (
        collate_variable_depth_volumes
        if args.dataset == "cq500" and args.n_slices is None
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )
        if test_dataset is not None
        else None
    )
    if args.dataset != "organmnist3d":
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
    use_plateau_scheduler = bool(getattr(args, "reduce_lr_on_plateau", True))
    scheduler_metric = args.early_stopping_metric
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if scheduler_metric == "bce_loss" else "max",
            factor=getattr(args, "lr_plateau_factor", 0.1),
            patience=getattr(args, "lr_plateau_patience", 3),
            threshold=getattr(args, "lr_plateau_threshold", 0.005),
            threshold_mode="abs",
            min_lr=args.min_lr,
        )
        if use_plateau_scheduler
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
    loss_tag = "bce" if binary or multi_label else "ce"
    lora_unfrozen = not (
        args.encoder_training == "lora" and args.freeze_epochs > 0
    )
    benchmark_enabled = bool(getattr(args, "benchmark", False))
    epoch_benchmarks: List[Dict[str, object]] = []

    logger.info(
        "dataset=%s adni_task=%s encoder=%s features=%s n_cls_tokens=%d aggregator=%s "
        "n_slices=%s "
        "encoder_training=%s lora_r=%s freeze_epochs=%d num_classes=%d "
        "train_%s=%d val_%s=%d test_%s=%d embed_dim=%d d_model=%d loss=%s",
        args.dataset,
        args.adni_task if args.dataset == "adni" else cq500_task if args.dataset == "cq500" else None,
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
    if scheduler is not None:
        logger.info(
            "LR scheduler=ReduceLROnPlateau metric=%s factor=%.3g patience=%d "
            "threshold=%.3g min_lr=%.3g",
            scheduler_metric,
            getattr(args, "lr_plateau_factor", 0.1),
            getattr(args, "lr_plateau_patience", 3),
            getattr(args, "lr_plateau_threshold", 0.005),
            args.min_lr,
        )
    if benchmark_enabled:
        logger.info(
            "Epoch benchmarking enabled: CUDA event timings measure GPU work; "
            "data_loading_s measures DataLoader wait time."
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
        profiler = EpochProfiler(device) if benchmark_enabled else None
        if benchmark_enabled and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start_time = time.perf_counter()
        train_start_time = epoch_start_time
        train_iterator = iter(train_loader)
        batch_wait_start = time.perf_counter()
        while True:
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            if profiler is not None:
                profiler.add_host_time(
                    "data_loading_s", time.perf_counter() - batch_wait_start
                )
            images, targets, slice_mask = _unpack_volume_batch(batch)
            with (
                profiler.stage("host_to_device_s")
                if profiler is not None
                else nullcontext()
            ):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                device_slice_mask = (
                    slice_mask.to(device, non_blocking=True)
                    if slice_mask is not None
                    else None
                )
            logits = model(
                images, slice_mask=device_slice_mask, profiler=profiler
            )
            if binary or multi_label:
                targets = targets.float()
                if bce_pos_weight is None:
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=bce_pos_weight
                    )
                predictions = (logits >= 0).float()
                correct += int((predictions == targets).sum().item())
            else:
                targets = targets.long()
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                correct += int((logits.argmax(dim=-1) == targets).sum().item())
            with (
                profiler.stage("backward_optimizer_s")
                if profiler is not None
                else nullcontext()
            ):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n_items = int(targets.numel())
            running_loss += float(loss.item()) * n_items
            seen += n_items
            batch_wait_start = time.perf_counter()

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)
        train_timings = profiler.finish() if profiler is not None else {}
        train_wall_time = time.perf_counter() - train_start_time
        if val_loader is None:
            if profiler is not None:
                epoch_benchmark: Dict[str, object] = {
                    "epoch": epoch,
                    "train_wall_s": train_wall_time,
                    "validation_s": 0.0,
                    "epoch_wall_s": time.perf_counter() - epoch_start_time,
                    "process_peak_rss_mb": _process_peak_rss_mb(),
                    **train_timings,
                    **_gpu_memory_metrics(device),
                }
                epoch_benchmarks.append(epoch_benchmark)
                save_epoch_benchmarks(
                    checkpoint_dir / "epoch_benchmarks.json", epoch_benchmarks
                )
            logger.info(
                "epoch %d/%d lr=%.6g train_%s=%.5f train_acc=%.3f%s",
                epoch,
                args.epochs,
                current_lr,
                loss_tag,
                train_loss,
                train_accuracy,
                (
                    " data=%.2fs dino=%.2fs head=%.2fs backward=%.2fs"
                    % (
                        train_timings.get("data_loading_s", 0.0),
                        train_timings.get("dino_forward_s", 0.0),
                        train_timings.get("rest_network_forward_s", 0.0),
                        train_timings.get("backward_optimizer_s", 0.0),
                    )
                    if profiler is not None
                    else ""
                ),
            )
            continue

        validation_start_time = time.perf_counter()
        val_loss, val_f1, val_auroc = evaluate(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
            multi_label=multi_label,
        )
        if profiler is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_wall_time = time.perf_counter() - validation_start_time
        if profiler is not None:
            epoch_benchmark = {
                "epoch": epoch,
                "train_wall_s": train_wall_time,
                "validation_s": validation_wall_time,
                "epoch_wall_s": time.perf_counter() - epoch_start_time,
                "process_peak_rss_mb": _process_peak_rss_mb(),
                **train_timings,
                **_gpu_memory_metrics(device),
            }
            epoch_benchmarks.append(epoch_benchmark)
            save_epoch_benchmarks(
                checkpoint_dir / "epoch_benchmarks.json", epoch_benchmarks
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
            "early_stopping_%s=%.5f%s%s",
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
            (
                " | time: data=%.2fs transfer=%.2fs dino=%.2fs head=%.2fs "
                "backward=%.2fs val=%.2fs epoch=%.2fs"
                % (
                    train_timings.get("data_loading_s", 0.0),
                    train_timings.get("host_to_device_s", 0.0),
                    train_timings.get("dino_forward_s", 0.0),
                    train_timings.get("rest_network_forward_s", 0.0),
                    train_timings.get("backward_optimizer_s", 0.0),
                    validation_wall_time,
                    epoch_benchmark["epoch_wall_s"] if profiler is not None else 0.0,
                )
                if profiler is not None
                else ""
            ),
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
            scheduler_value = validation_metrics[scheduler_metric]
            if math.isfinite(scheduler_value):
                lr_before_step = optimizer.param_groups[0]["lr"]
                scheduler.step(scheduler_value)
                lr_after_step = optimizer.param_groups[0]["lr"]
                if lr_after_step < lr_before_step:
                    logger.info(
                        "Reduced learning rate after validation %s plateau: %.6g → %.6g",
                        scheduler_metric,
                        lr_before_step,
                        lr_after_step,
                    )
            else:
                logger.warning(
                    "Skipping ReduceLROnPlateau step because validation %s is NaN",
                    scheduler_metric,
                )

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
        y_true, y_pred, y_probability = collect_predictions(
            model, val_loader, device, multi_label=multi_label
        )
        final_val_loss, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
            multi_label=multi_label,
        )
        val_metrics = compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=num_classes, multi_label=multi_label
        )
        confusion = None if multi_label else save_confusion_matrix(
            y_true, y_pred, checkpoint_dir / "val_confusion_matrix.png",
            class_names=class_names, title="Validation confusion matrix",
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
        y_true, y_pred, y_probability = collect_predictions(
            model, test_loader, device, multi_label=multi_label
        )
        final_test_loss, _, _ = evaluate(
            model,
            test_loader,
            device,
            class_weights=class_weights,
            bce_pos_weight=bce_pos_weight,
            multi_label=multi_label,
        )
        test_metrics = compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=num_classes, multi_label=multi_label
        )
        confusion = None if multi_label else save_confusion_matrix(
            y_true, y_pred, checkpoint_dir / "test_confusion_matrix.png",
            class_names=class_names, title="Test confusion matrix",
        )
        if not multi_label:
            save_auroc_plot(
                y_true, y_probability, checkpoint_dir / "test_auroc.png",
                num_classes=num_classes, class_names=class_names, title="Test ROC curve",
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
        epoch_benchmarks=epoch_benchmarks if benchmark_enabled else None,
    )
