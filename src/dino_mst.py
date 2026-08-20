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

_SRC_DIR = Path(__file__).resolve().parent
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
from merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from vit_lora import LoRA, add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DATASET_CHOICES = ("duke", "adni")
AGGREGATOR_CHOICES = ("transformer", "mean")
ENCODER_TRAINING_CHOICES = ("frozen", "lora")
EARLY_STOPPING_METRIC_CHOICES = ("bce_loss", "f1", "auroc")
EARLY_STOPPING_MIN_IMPROVEMENT = 0.005
MultiSliceDataset = Union[DukeMultiSliceDataset, ADNIMultiSliceDataset]
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
        n_slices: int,
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
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
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
        self.n_slices = int(n_slices)
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

    def encode_slices(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, S, C, H, W]`` into ``[B, S, feature_dim]``."""
        if images.ndim != 5:
            raise ValueError(f"Expected [B, S, C, H, W], got {tuple(images.shape)}")
        batch, n_slices = images.shape[:2]
        if n_slices != self.n_slices:
            raise ValueError(f"Expected {self.n_slices} slices, got {n_slices}")
        flat = images.reshape(batch * n_slices, *images.shape[2:])
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
        return token.reshape(batch, n_slices, -1)

    def extract_volume_token(self, images: torch.Tensor) -> torch.Tensor:
        """Return one MST volume representation for each input volume.

        For the transformer aggregator this is the output global token.  The
        mean-pool variant has no learned global token, so this instead returns
        its mean-pooled slice representation.  Keeping the aggregation here
        makes inference code able to use the exact representation consumed by
        the classifier without duplicating the MST forward pass.
        """
        slices = self.slice_projection(self.encode_slices(images))
        if self.aggregator == "mean":
            return slices.mean(dim=1)
        assert self.global_token is not None and self.transformer is not None
        global_token = self.global_token.expand(slices.shape[0], -1, -1)
        sequence = torch.cat([global_token, slices], dim=1)
        sequence = sequence + self.position_embedding
        return self.transformer(sequence)[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        volume = self.extract_volume_token(images)
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
    parser.add_argument("--n-slices", type=int, default=8)
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
        help="Enable random MONAI 3-D volume transforms (default)",
    )
    augmentation.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable random volume transforms",
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
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable validation-metric early stopping (default); use "
            "--no-early-stopping to train for all epochs"
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many epochs without improving the selected metric",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=list(EARLY_STOPPING_METRIC_CHOICES),
        default="bce_loss",
        help=(
            "Validation metric for best-checkpoint selection and early stopping: "
            "bce_loss (minimize), f1 (maximize), or auroc (maximize)"
        ),
    )
    ce_weight = parser.add_mutually_exclusive_group()
    ce_weight.add_argument(
        "--weight-ce-loss",
        "--bce-loss-weight",
        dest="weight_ce_loss",
        action="store_true",
        default=False,
        help="Weight binary BCE positives by training class imbalance",
    )
    ce_weight.add_argument(
        "--no-weight-ce-loss",
        "--no-bce-loss-weight",
        dest="weight_ce_loss",
        action="store_false",
        help="Disable binary BCE class weighting (default)",
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
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing saved patient IDs under train, val, "
            "and test; patients absent from the current dataset are ignored"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--slice-aggregator",
        choices=list(AGGREGATOR_CHOICES),
        default="transformer",
        help=(
            "How per-slice embeddings become a volume embedding: multi-slice "
            "transformer (default) or mean pooling, which ignores the --mst-* options"
        ),
    )
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--mst-depth", type=int, default=2)
    parser.add_argument("--mst-heads", type=int, default=12)
    parser.add_argument("--mst-ffn-dim", type=int, default=3072)
    parser.add_argument("--mst-dropout", type=float, default=0.1)
    parser.add_argument(
        "--encoder",
        choices=list(ENCODER_CHOICES),
        default="dinov3",
        help="Frozen backbone: dinov3 | meddinov3 | braindino | custom (wireframe)",
    )
    parser.add_argument(
        "--encoder-training",
        choices=list(ENCODER_TRAINING_CHOICES),
        default="frozen",
        help=(
            "Backbone training mode: frozen keeps the encoder fixed; lora freezes "
            "the base encoder and trains low-rank Q/K/V adapters"
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank used when --encoder-training=lora",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help=(
            "For --encoder-training=lora, keep LoRA adapters frozen for the first "
            "N epochs while training only the MST/head"
        ),
    )
    parser.add_argument(
        "--features",
        choices=list(FEATURE_CHOICES),
        default="cls",
        help=(
            "Per-slice token: cls (CLS only), patch (attention-pooled patches), "
            "or both (attention pool over CLS + patch tokens together)"
        ),
    )
    parser.add_argument(
        "--n-cls-tokens",
        type=int,
        default=1,
        help=(
            "Number of final encoder-layer CLS tokens to concatenate per slice; "
            "used only when --features=cls"
        ),
    )
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Encoder checkpoint. For encoder=dinov3, a .pth file or DINOv3 "
            "distributed-checkpoint directory loads only the ViT-B teacher "
            "backbone. Defaults: dinov3 → opt/dinov3-weights/...; "
            "meddinov3 → opt/meddinov3/model.pth; "
            "braindino → opt/braindino/brain_dino_weights.pth"
        ),
    )
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=DINOV3_REPO,
        help="Local DINOv3 repo (architecture source for dinov3/meddinov3/braindino)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Subfolder under runs/<dataset>/ for checkpoints and plots. "
            "Defaults to <encoder>_<mst|meanpool>_<dataset>."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <repo>/runs/<dataset>/<run-name>)",
    )
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
    if args.d_model <= 0:
        parser.error("--d-model must be positive")
    if args.lora_r <= 0:
        parser.error("--lora-r must be positive")
    if args.freeze_epochs < 0:
        parser.error("--freeze-epochs must be non-negative")
    if args.encoder_training != "lora" and args.freeze_epochs > 0:
        parser.error("--freeze-epochs is only valid with --encoder-training=lora")
    if args.slice_aggregator == "transformer":
        if args.mst_depth <= 0:
            parser.error("--mst-depth must be positive")
        if args.mst_heads <= 0 or args.d_model % args.mst_heads:
            parser.error("--d-model must be divisible by positive --mst-heads")
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
        else REPO_ROOT
        / "runs"
        / args.dataset
        / (
            args.run_name
            or "{}_{}_{}".format(
                args.encoder,
                "mst" if args.slice_aggregator == "transformer" else "meanpool",
                args.dataset,
            )
        )
    )
    logger.info("Run output directory: %s", checkpoint_dir)
    train(args, device, Path(checkpoint_dir))


if __name__ == "__main__":
    main()
