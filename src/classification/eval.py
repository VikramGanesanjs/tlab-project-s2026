"""Evaluate a saved classification MST head on its deterministic CV folds."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from classification.model import MultiSliceDinoModel  # noqa: E402
from classification.neurovfm import (  # noqa: E402
    DEFAULT_NEUROVFM_REPO,
    DEFAULT_NEUROVFM_WEIGHTS,
    NeuroVFMEncoder,
    NeuroVFMVolumeClassifier,
)
from classification.train import (  # noqa: E402
    _assert_loader_patient_disjoint,
    _cq500_patient_stratum,
    _require_dataset_patient_ids,
    build_dataset,
    collate_variable_depth_volumes,
    collect_labels,
    collect_predictions,
    compute_classification_metrics,
    evaluate,
    is_multilabel_task,
    is_binary_task,
    save_run_summary,
    saved_split_patient_ids,
    task_config,
)
from datasets.adni import DEFAULT_ROOT as ADNI_DEFAULT_ROOT  # noqa: E402
from datasets.cq500 import DEFAULT_ROOT as CQ500_DEFAULT_ROOT  # noqa: E402
from datasets.organmnist3d import DEFAULT_ROOT as ORGANMNIST3D_DEFAULT_ROOT  # noqa: E402
from datasets.breastdm import DEFAULT_ROOT as BREASTDM_DEFAULT_ROOT  # noqa: E402
from utils.load_dinov3 import (  # noqa: E402
    DINOV3_REPO,
    REPO_ROOT,
    inverse_frequency_weights,
    load_encoder,
)
from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from utils.fold_cv import make_dataset_patient_folds  # noqa: E402
from utils.vit_lora import add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_DUKE_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head", type=Path, help="Path to last_mst.pt or best_mst.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <head directory>/metrics_summary.json).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument(
        "--adni-manifest-path",
        type=Path,
        default=None,
        help="Override the ADNI patient-to-NIfTI JSON manifest path",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Override the checkpoint's legacy JSON patient split file",
    )
    parser.add_argument("--scan", default="pre")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.splits_file is not None and not args.splits_file.is_file():
        parser.error(f"--splits-file does not exist or is not a file: {args.splits_file}")
    return args


def _value(checkpoint: dict[str, Any], name: str, default: Any = None) -> Any:
    """Read a checkpoint setting, failing clearly when architecture data is absent."""
    value = checkpoint.get(name, default)
    if value is None:
        raise ValueError(f"Checkpoint does not contain required setting {name!r}")
    return value


def _evaluation_args(cli_args: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    """Construct the data/model arguments recorded in a classification checkpoint."""
    dataset = _value(checkpoint, "dataset")
    data_root = cli_args.data_root or {
        "duke": DEFAULT_DUKE_DATA_ROOT,
        "adni": ADNI_DEFAULT_ROOT,
        "cq500": CQ500_DEFAULT_ROOT,
        "organmnist3d": ORGANMNIST3D_DEFAULT_ROOT,
        "breastdm": BREASTDM_DEFAULT_ROOT,
    }.get(dataset)
    if data_root is None:
        raise ValueError(f"Unsupported dataset in checkpoint: {dataset!r}")
    if "n_slices" not in checkpoint:
        raise ValueError("Checkpoint does not contain required setting 'n_slices'")
    n_slices = checkpoint["n_slices"]
    if n_slices is not None:
        n_slices = int(n_slices)
    if n_slices is None and dataset != "cq500":
        raise ValueError("n_slices=None checkpoints are supported only for CQ500")
    return argparse.Namespace(
        head=cli_args.head,
        output=cli_args.output,
        data_root=data_root,
        csv_path=cli_args.csv_path,
        adni_manifest_path=cli_args.adni_manifest_path,
        scan=cli_args.scan,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        dinov3_repo=cli_args.dinov3_repo,
        device=cli_args.device,
        dataset=dataset,
        adni_task=checkpoint.get("adni_task") or "cn_ad",
        cq500_task=checkpoint.get("cq500_task") or "ich",
        n_slices=n_slices,
        cq500_max_slices=int(checkpoint.get("cq500_max_slices", 128)),
        image_size=int(_value(checkpoint, "image_size")),
        include_bilateral=bool(checkpoint.get("include_bilateral", False)),
        augment=False,
        encoder=_value(checkpoint, "encoder"),
        encoder_training=checkpoint.get("encoder_training", "frozen"),
        lora_r=int(checkpoint.get("lora_r") or 16),
        model_name=_value(checkpoint, "model_name"),
        weights=(
            Path(checkpoint["weights"])
            if checkpoint.get("weights") is not None
            else None
        ),
        features=_value(checkpoint, "features"),
        n_cls_tokens=int(checkpoint.get("n_cls_tokens", 1)),
        slice_aggregator=_value(checkpoint, "slice_aggregator"),
        d_model=int(_value(checkpoint, "d_model")),
        mst_depth=int(_value(checkpoint, "mst_depth")),
        mst_heads=int(_value(checkpoint, "mst_heads")),
        mst_ffn_dim=int(_value(checkpoint, "mst_ffn_dim")),
        mst_dropout=float(_value(checkpoint, "mst_dropout")),
        hidden_dim=int(_value(checkpoint, "hidden_dim")),
        neurovfm_repo=Path(checkpoint.get("neurovfm_repo") or DEFAULT_NEUROVFM_REPO),
        neurovfm_input_channels=int(checkpoint.get("neurovfm_input_channels") or 1),
        neurovfm_volume_shape=tuple(checkpoint.get("neurovfm_volume_shape") or (128, 192, 192)),
        neurovfm_input_normalization=checkpoint.get("neurovfm_input_normalization") or "none",
        neurovfm_modality=checkpoint.get("neurovfm_modality") or "auto",
        weight_ce_loss=bool(checkpoint.get("weight_ce_loss", False)),
        n_folds=int(checkpoint.get("n_folds", 5)),
        fold=int(checkpoint.get("fold", 0)),
        data_seed=int(checkpoint.get("data_seed", 0)),
        train_ratio=float(checkpoint.get("train_ratio", 1.0)),
        splits_file=(
            cli_args.splits_file
            if cli_args.splits_file is not None
            else Path(checkpoint["splits_file"])
            if checkpoint.get("splits_file") is not None
            else None
        ),
        seed=0,
    )


def _split_patient_ids(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Recreate the checkpoint's saved split or patient-level CV assignment."""
    split_dataset = build_dataset(
        args, augment=False, split="all" if args.dataset == "organmnist3d" else None
    )
    if args.splits_file is not None:
        return saved_split_patient_ids(args, split_dataset)
    folds = make_dataset_patient_folds(
        split_dataset,
        n_folds=args.n_folds,
        seed=args.data_seed,
        target_fn=_cq500_patient_stratum if args.dataset == "cq500" else None,
    )
    return folds.get_split(
        args.fold, train_ratio=args.train_ratio, train_seed=args.data_seed
    )


def _load_model(args: argparse.Namespace, checkpoint: dict[str, Any], device: torch.device):
    num_classes, _, _ = task_config(
        args.dataset, adni_task=args.adni_task, cq500_task=args.cq500_task
    )
    if args.encoder == "neurovfm":
        modality = args.neurovfm_modality
        if modality == "auto":
            modality = "ct" if args.dataset == "cq500" else "mri"
        encoder = NeuroVFMEncoder(
            repo=args.neurovfm_repo,
            weights=args.weights or DEFAULT_NEUROVFM_WEIGHTS,
            device=device,
            modality=modality,
        )
        model = NeuroVFMVolumeClassifier(
            encoder,
            input_channels=args.neurovfm_input_channels,
            volume_shape=args.neurovfm_volume_shape,
            input_normalization=args.neurovfm_input_normalization,
            hidden_dim=args.hidden_dim or None,
            num_classes=num_classes,
        ).to(device)
        model.load_trainable_state_dict(checkpoint["model"])
        return model
    if args.weights is not None and args.encoder == "dinov3":
        encoder = load_custom_dinov3_encoder(
            checkpoint=args.weights,
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
    model.load_trainable_state_dict(checkpoint["model"])
    return model


def _metrics_for_loader(
    model: MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
    *,
    class_weights: Optional[torch.Tensor],
    bce_pos_weight: Optional[torch.Tensor],
    multi_label: bool = False,
) -> dict[str, object]:
    y_true, y_pred, y_probability = collect_predictions(
        model, loader, device, multi_label=multi_label
    )
    loss, _, _ = evaluate(
        model,
        loader,
        device,
        class_weights=class_weights,
        bce_pos_weight=bce_pos_weight,
        multi_label=multi_label,
    )
    return {
        "n": int(len(y_true)),
        "loss": loss,
        **compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=model.num_classes, multi_label=multi_label
        ),
    }


def evaluate_head(cli_args: argparse.Namespace) -> Path:
    checkpoint = torch.load(cli_args.head, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{cli_args.head} is not a classification MST checkpoint")
    args = _evaluation_args(cli_args, checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dataset in {"organmnist3d", "breastdm"}:
        train_dataset = build_dataset(args, augment=False, split="train")
        val_dataset = build_dataset(args, augment=False, split="val")
        test_dataset = build_dataset(args, augment=False, split="test")
    else:
        train_ids, val_ids, test_ids = _split_patient_ids(args)
        train_dataset = build_dataset(args, augment=False, patient_ids=train_ids)
        val_dataset = build_dataset(args, augment=False, patient_ids=val_ids)
        test_dataset = build_dataset(args, augment=False, patient_ids=test_ids)
        _require_dataset_patient_ids(train_dataset, train_ids, split_name="train")
        _require_dataset_patient_ids(val_dataset, val_ids, split_name="validation")
        _require_dataset_patient_ids(test_dataset, test_ids, split_name="test")

    collate_fn = (
        collate_variable_depth_volumes
        if args.dataset == "cq500" and args.n_slices is None
        else None
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn) if test_dataset is not None else None
    if args.dataset != "organmnist3d":
        _assert_loader_patient_disjoint(train_loader, val_loader, test_loader)

    num_classes, _, _ = task_config(
        args.dataset, adni_task=args.adni_task, cq500_task=args.cq500_task
    )
    class_weights: Optional[torch.Tensor] = None
    bce_pos_weight: Optional[torch.Tensor] = None
    multi_label = is_multilabel_task(args.dataset, cq500_task=args.cq500_task)
    if args.weight_ce_loss and multi_label:
        labels = collect_labels(train_dataset)
        positives = labels.sum(axis=0)
        negatives = len(labels) - positives
        bce_pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    elif args.weight_ce_loss and is_binary_task(num_classes):
        labels = collect_labels(train_dataset)
        negative_count = int((labels == 0).sum())
        positive_count = int((labels == 1).sum())
        if not negative_count or not positive_count:
            raise ValueError("Cannot compute weighted BCE loss without both classes")
        bce_pos_weight = torch.tensor([negative_count / positive_count], device=device)
    elif not is_binary_task(num_classes) and not multi_label:
        labels = collect_labels(train_dataset)
        counts = {index: int((labels == index).sum()) for index in range(num_classes)}
        class_weights = inverse_frequency_weights(counts, num_classes=num_classes).to(device)

    model = _load_model(args, checkpoint, device)
    metrics: dict[str, dict[str, object]] = {}
    if val_loader is not None:
        metrics["val"] = _metrics_for_loader(model, val_loader, device, class_weights=class_weights, bce_pos_weight=bce_pos_weight, multi_label=multi_label)
    if test_loader is not None:
        metrics["test"] = _metrics_for_loader(model, test_loader, device, class_weights=class_weights, bce_pos_weight=bce_pos_weight, multi_label=multi_label)

    output = cli_args.output or cli_args.head.parent / "metrics_summary.json"
    save_run_summary(output, args=args, metrics_by_split=metrics)
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = evaluate_head(parse_args(argv))
    logger.info("Saved evaluation summary to %s", output)


if __name__ == "__main__":
    main()
