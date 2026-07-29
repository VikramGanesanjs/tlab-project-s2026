"""Supervised contrastive LoRA fine-tuning for single MRI slices."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    ADNI_TASK_CHOICES,
    DEFAULT_ADNI_TASK,
    DEFAULT_ROOT as ADNI_DEFAULT_ROOT,
    resolve_adni_task,
    build_adni_transform,
)
from datasets.duke import DukeClassificationDataset  # noqa: E402
from dinov3_baseline import (  # noqa: E402
    CLASS_NAMES as DUKE_CLASS_NAMES,
    DINOV3_REPO,
    ENCODER_CHOICES,
    REPO_ROOT,
    build_transform,
    load_encoder,
)
from vit_lora import add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_CHOICES = ("duke", "adni")
DEFAULT_DUKE_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
LORA_PARAMETER_NAMES = ("w_a_q", "w_b_q", "w_a_k", "w_b_k", "w_a_v", "w_b_v")


def is_lora_parameter_name(name: str) -> bool:
    return any(parameter_name in name for parameter_name in LORA_PARAMETER_NAMES)


def lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if is_lora_parameter_name(name)
    }


def lora_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and is_lora_parameter_name(name)
    ]


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Single-view supervised contrastive loss over a batch of normalized features."""
    if features.ndim != 2:
        raise ValueError(f"Expected [B, D] features, got {tuple(features.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    labels = labels.reshape(-1)
    batch_size = int(features.shape[0])
    if batch_size <= 1:
        return features.sum() * 0.0

    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return features.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positive_mask, 0.0).sum(
        dim=1
    ) / positive_mask.sum(dim=1).clamp_min(1)
    return -positive_log_prob[valid_anchor].mean()


def collect_labels(dataset: Dataset) -> np.ndarray:
    return np.asarray(
        [int(dataset.get_target(index)) for index in range(len(dataset))],  # type: ignore[attr-defined]
        dtype=np.int64,
    )


def get_patient_id(dataset: Dataset, index: int) -> str:
    if hasattr(dataset, "get_patient_id"):
        return str(dataset.get_patient_id(index))  # type: ignore[attr-defined]
    if hasattr(dataset, "_entries"):
        return str(dataset._entries[index][0])  # type: ignore[attr-defined]
    return str(index)


def patient_level_split(
    dataset: Dataset,
    *,
    val_frac: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset]]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if val_frac == 0:
        return Subset(dataset, list(range(len(dataset)))), None

    patient_indices: Dict[str, List[int]] = defaultdict(list)
    patient_labels: Dict[str, int] = {}
    for index in range(len(dataset)):
        patient_id = get_patient_id(dataset, index)
        patient_indices[patient_id].append(index)
        patient_labels.setdefault(patient_id, int(dataset.get_target(index)))  # type: ignore[attr-defined]

    rng = np.random.RandomState(seed)
    grouped: Dict[int, List[str]] = defaultdict(list)
    for patient_id, label in patient_labels.items():
        grouped[label].append(patient_id)

    train_patients: List[str] = []
    val_patients: List[str] = []
    for label, patients in grouped.items():
        rng.shuffle(patients)
        n_val = int(round(len(patients) * val_frac))
        if len(patients) >= 2:
            n_val = min(max(n_val, 1), len(patients) - 1)
        else:
            n_val = 0
        val_patients.extend(patients[:n_val])
        train_patients.extend(patients[n_val:])
        logger.info(
            "patient split label=%s train=%d val=%d",
            label,
            len(patients) - n_val,
            n_val,
        )

    train_indices = [
        index for patient_id in train_patients for index in patient_indices[patient_id]
    ]
    val_indices = [
        index for patient_id in val_patients for index in patient_indices[patient_id]
    ]
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices) if val_indices else None,
    )


def class_names_for(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.dataset == "duke":
        return DUKE_CLASS_NAMES
    return resolve_adni_task(args.adni_task).class_names


def build_dataset(args: argparse.Namespace, *, augment: bool) -> Dataset:
    transform_kwargs = dict(
        image_size=args.image_size,
        augment=augment,
        crop_scale_min=args.crop_scale_min,
        jitter=args.jitter,
        rotation_degrees=args.rotation_degrees,
        horizontal_flip_prob=args.horizontal_flip_prob,
        vertical_flip_prob=args.vertical_flip_prob,
    )
    if args.dataset == "duke":
        return DukeClassificationDataset(
            root=args.data_root,
            scan=args.scan,
            z_min=args.z_min,
            z_max=args.z_max,
            include_bilateral=args.include_bilateral,
            transform=build_transform(**transform_kwargs),
        )
    if args.dataset == "adni":
        return ADNIClassificationDataset(
            root=args.data_root,
            csv_path=args.csv_path,
            task=args.adni_task,
            z_min=args.z_min,
            z_max=args.z_max,
            transform=build_adni_transform(**transform_kwargs),
        )
    raise ValueError(f"Unknown dataset={args.dataset!r}")


def summarize_class_counts(
    labels: np.ndarray,
    class_names: Sequence[str],
    name: str,
) -> None:
    counts = Counter(int(label) for label in labels.tolist())
    total = max(sum(counts.values()), 1)
    logger.info("%s class distribution (n=%d):", name, len(labels))
    for index, class_name in enumerate(class_names):
        count = counts.get(index, 0)
        logger.info(
            "  %d %-14s %5d  (%5.1f%%)",
            index,
            class_name,
            count,
            100.0 * count / total,
        )


@torch.no_grad()
def evaluate(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float,
) -> float:
    encoder.eval()
    total_loss = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        output = encoder.forward_features(images)
        features = F.normalize(output["x_norm_clstoken"].float(), p=2, dim=-1)
        loss = supcon_loss(features, targets, temperature)
        n_items = int(targets.numel())
        total_loss += float(loss.item()) * n_items
        total += n_items
    return total_loss / max(total, 1)


def checkpoint_payload(
    encoder: torch.nn.Module,
    *,
    epoch: int,
    args: argparse.Namespace,
    **metrics: object,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "encoder": args.encoder,
        "model_name": args.model_name,
        "weights": str(args.weights) if args.weights is not None else None,
        "dataset": args.dataset,
        "adni_task": args.adni_task if args.dataset == "adni" else None,
        "lora_r": args.lora_r,
        "temperature": args.temperature,
        "image_size": args.image_size,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "model": lora_state_dict(encoder),
        **metrics,
    }


def train(args: argparse.Namespace, device: torch.device, checkpoint_dir: Path) -> None:
    torch.manual_seed(args.seed)
    class_names = class_names_for(args)
    train_full = build_dataset(args, augment=args.augment)
    labels = collect_labels(train_full)
    summarize_class_counts(labels, class_names, "full dataset (slice)")
    train_subset, val_subset = patient_level_split(
        train_full,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    eval_full = build_dataset(args, augment=False) if val_subset is not None else None
    val_dataset = (
        Subset(eval_full, list(val_subset.indices))
        if eval_full is not None and val_subset is not None
        else None
    )
    summarize_class_counts(labels[train_subset.indices], class_names, "train (slice)")
    if val_dataset is not None:
        summarize_class_counts(labels[val_dataset.indices], class_names, "val (slice)")

    train_loader = DataLoader(
        train_subset,
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

    encoder = load_encoder(
        args.encoder,
        device=device,
        weights=args.weights,
        repo_dir=args.dinov3_repo,
        model_name=args.model_name,
    )
    add_lora_to_vit(encoder, r=args.lora_r)
    freeze_non_lora_parameters(encoder)
    encoder.to(device)
    optimizer = torch.optim.AdamW(
        lora_parameters(encoder),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
        if args.cosine_lr
        else None
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    metrics: Dict[str, object] = {}
    logger.info(
        "encoder=%s lora_r=%d dataset=%s adni_task=%s train_slices=%d "
        "val_slices=%d temperature=%.4f",
        args.encoder,
        args.lora_r,
        args.dataset,
        args.adni_task if args.dataset == "adni" else None,
        len(train_subset),
        len(val_dataset) if val_dataset is not None else 0,
        args.temperature,
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        running_loss = 0.0
        seen = 0
        skipped_batches = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            output = encoder.forward_features(images)
            features = F.normalize(output["x_norm_clstoken"].float(), p=2, dim=-1)
            loss = supcon_loss(features, targets, args.temperature)
            if not loss.requires_grad:
                skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_items = int(targets.numel())
            running_loss += float(loss.item()) * n_items
            seen += n_items

        train_loss = running_loss / max(seen, 1)
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss = evaluate(
                encoder,
                val_loader,
                device,
                temperature=args.temperature,
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                torch.save(
                    checkpoint_payload(
                        encoder,
                        epoch=epoch,
                        args=args,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    ),
                    checkpoint_dir / "best_supcon_lora.pt",
                )
            logger.info(
                "epoch %d/%d train_supcon=%.5f val_supcon=%.5f skipped_batches=%d%s",
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                skipped_batches,
                " *" if improved else "",
            )
            metrics = {"best_val_loss": best_val}
        else:
            logger.info(
                "epoch %d/%d train_supcon=%.5f skipped_batches=%d",
                epoch,
                args.epochs,
                train_loss,
                skipped_batches,
            )

    torch.save(
        checkpoint_payload(
            encoder,
            epoch=args.epochs,
            args=args,
            train_loss=train_loss,
            **metrics,
        ),
        checkpoint_dir / "last_supcon_lora.pt",
    )
    with (checkpoint_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parameters": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "metrics": metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _yaml_defaults(path: Path, parser: argparse.ArgumentParser) -> Dict[str, object]:
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
    parser.add_argument("--params-file", "--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=list(DATASET_CHOICES), default="adni")
    parser.add_argument("--adni-task", choices=list(ADNI_TASK_CHOICES), default=DEFAULT_ADNI_TASK)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--scan", type=str, default="pre")
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--include-bilateral", action="store_true")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augment", dest="augment", action="store_true", default=True)
    augmentation.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--crop-scale-min", type=float, default=0.8)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--rotation-degrees", type=float, default=15.0)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--vertical-flip-prob", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", choices=list(ENCODER_CHOICES), default="dinov3")
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    parser.add_argument("--lora-r", type=int, default=16)
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
        args.data_root = ADNI_DEFAULT_ROOT if args.dataset == "adni" else DEFAULT_DUKE_ROOT
    if args.vertical_flip_prob is None:
        args.vertical_flip_prob = 0.0 if args.dataset == "adni" else 0.5
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.batch_size <= 1:
        parser.error("--batch-size must be greater than 1 for SupCon")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.lr < 0:
        parser.error("--lr must be non-negative")
    if args.min_lr < 0:
        parser.error("--min-lr must be non-negative")
    if args.cosine_lr and args.min_lr > args.lr:
        parser.error("--min-lr cannot exceed --lr when --cosine-lr is enabled")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.lora_r <= 0:
        parser.error("--lora-r must be positive")
    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val-frac must be in [0, 1)")
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
        else REPO_ROOT / "runs" / "supcon" / (args.run_name or f"{args.encoder}_{args.dataset}_lora")
    )
    logger.info("Run output directory: %s", checkpoint_dir)
    train(args, device, Path(checkpoint_dir))


if __name__ == "__main__":
    main()
