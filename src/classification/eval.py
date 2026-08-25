"""Evaluate a saved classification MST head on the dataset's held-out split.

The checkpoint supplies the model and encoder configuration, so the required
input is a ``last_mst.pt`` (or ``best_mst.pt``) file. Duke and ADNI also need
the patient split JSON used for training; OrganMNIST3D uses its official split
arrays directly. Metrics are written in the same format as ``run_summary.json``.
"""

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
from classification.train import (  # noqa: E402
    _adni_patient_stratum,
    _assert_loader_patient_disjoint,
    _dataset_patient_ids,
    _duke_patient_stratum,
    _require_dataset_patient_ids,
    build_dataset,
    collect_labels,
    collect_predictions,
    compute_classification_metrics,
    evaluate,
    is_binary_task,
    save_run_summary,
    task_config,
)
from datasets.adni import DEFAULT_ROOT as ADNI_DEFAULT_ROOT  # noqa: E402
from datasets.organmnist3d import DEFAULT_ROOT as ORGANMNIST3D_DEFAULT_ROOT  # noqa: E402
from utils.load_dinov3 import (  # noqa: E402
    DINOV3_REPO,
    REPO_ROOT,
    inverse_frequency_weights,
    load_encoder,
)
from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from utils.splits import patient_level_stratified_split  # noqa: E402
from utils.vit_lora import add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_DUKE_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head", type=Path, help="Path to last_mst.pt or best_mst.pt")
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help=(
            "Patient split JSON containing train, val, and test assignments. "
            "Required for Duke/ADNI; unused for OrganMNIST3D."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <head directory>/metrics_summary.json).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
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
        "organmnist3d": ORGANMNIST3D_DEFAULT_ROOT,
    }.get(dataset)
    if data_root is None:
        raise ValueError(f"Unsupported dataset in checkpoint: {dataset!r}")
    return argparse.Namespace(
        head=cli_args.head,
        splits_file=cli_args.splits_file,
        output=cli_args.output,
        data_root=data_root,
        csv_path=cli_args.csv_path,
        scan=cli_args.scan,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        dinov3_repo=cli_args.dinov3_repo,
        device=cli_args.device,
        dataset=dataset,
        adni_task=checkpoint.get("adni_task") or "cn_ad",
        n_slices=int(_value(checkpoint, "n_slices")),
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
        weight_ce_loss=bool(checkpoint.get("weight_ce_loss", False)),
        val_frac=float(checkpoint.get("val_frac", 0.1)),
        test_frac=float(checkpoint.get("test_frac", checkpoint.get("val_frac", 0.1))),
        seed=0,
    )


def _split_patient_ids(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Load the saved split with the same ADNI compatibility settings as training."""
    if args.splits_file is None:
        raise ValueError(f"--splits-file is required for dataset={args.dataset!r}")
    split_dataset = build_dataset(args, augment=False)
    stratum_fn = _adni_patient_stratum if args.dataset == "adni" else _duke_patient_stratum
    _, metadata = patient_level_stratified_split(
        split_dataset,
        dataset_name=args.dataset,
        train_fraction=1.0 - args.val_frac - args.test_frac,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        stratum_fn=stratum_fn,
        split_file=args.splits_file,
        use_saved_split_config=False,
        allow_saved_patient_superset=args.dataset == "adni",
        validate_saved_patient_strata=args.dataset != "adni",
    )
    splits = metadata["splits"]
    train_ids = [str(patient_id) for patient_id in splits["train"]]
    val_ids = [str(patient_id) for patient_id in splits["val"]]
    test_ids = [str(patient_id) for patient_id in splits["test"]]
    if args.dataset == "adni":
        valid_ids = _dataset_patient_ids(split_dataset)
        train_ids = [patient_id for patient_id in train_ids if patient_id in valid_ids]
        val_ids = [patient_id for patient_id in val_ids if patient_id in valid_ids]
        test_ids = [patient_id for patient_id in test_ids if patient_id in valid_ids]
    return train_ids, val_ids, test_ids


def _load_model(args: argparse.Namespace, checkpoint: dict[str, Any], device: torch.device) -> MultiSliceDinoModel:
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
    num_classes, _, _ = task_config(args.dataset, adni_task=args.adni_task)
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
) -> dict[str, object]:
    y_true, y_pred, y_probability = collect_predictions(model, loader, device)
    loss, _, _ = evaluate(
        model,
        loader,
        device,
        class_weights=class_weights,
        bce_pos_weight=bce_pos_weight,
    )
    return {
        "n": int(len(y_true)),
        "loss": loss,
        **compute_classification_metrics(
            y_true, y_pred, y_probability, num_classes=model.num_classes
        ),
    }


def evaluate_head(cli_args: argparse.Namespace) -> Path:
    checkpoint = torch.load(cli_args.head, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{cli_args.head} is not a classification MST checkpoint")
    args = _evaluation_args(cli_args, checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dataset == "organmnist3d":
        train_dataset = build_dataset(args, augment=False, split="train")
        val_dataset = build_dataset(args, augment=False, split="val")
        test_dataset = build_dataset(args, augment=False, split="test")
    else:
        train_ids, val_ids, test_ids = _split_patient_ids(args)
        train_dataset = build_dataset(args, augment=False, patient_ids=train_ids)
        val_dataset = (
            build_dataset(args, augment=False, patient_ids=val_ids) if val_ids else None
        )
        test_dataset = (
            build_dataset(args, augment=False, patient_ids=test_ids) if test_ids else None
        )
        _require_dataset_patient_ids(train_dataset, train_ids, split_name="train")
        if val_dataset is not None:
            _require_dataset_patient_ids(
                val_dataset, val_ids, split_name="validation"
            )
        if test_dataset is not None:
            _require_dataset_patient_ids(test_dataset, test_ids, split_name="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if test_dataset is not None else None
    if args.dataset != "organmnist3d":
        _assert_loader_patient_disjoint(train_loader, val_loader, test_loader)

    num_classes, _, _ = task_config(args.dataset, adni_task=args.adni_task)
    class_weights: Optional[torch.Tensor] = None
    bce_pos_weight: Optional[torch.Tensor] = None
    if args.weight_ce_loss and is_binary_task(num_classes):
        labels = collect_labels(train_dataset)
        negative_count = int((labels == 0).sum())
        positive_count = int((labels == 1).sum())
        if not negative_count or not positive_count:
            raise ValueError("Cannot compute weighted BCE loss without both classes")
        bce_pos_weight = torch.tensor([negative_count / positive_count], device=device)
    elif not is_binary_task(num_classes):
        labels = collect_labels(train_dataset)
        counts = {index: int((labels == index).sum()) for index in range(num_classes)}
        class_weights = inverse_frequency_weights(counts, num_classes=num_classes).to(device)

    model = _load_model(args, checkpoint, device)
    metrics: dict[str, dict[str, object]] = {}
    if val_loader is not None:
        metrics["val"] = _metrics_for_loader(model, val_loader, device, class_weights=class_weights, bce_pos_weight=bce_pos_weight)
    if test_loader is not None:
        metrics["test"] = _metrics_for_loader(model, test_loader, device, class_weights=class_weights, bce_pos_weight=bce_pos_weight)

    output = cli_args.output or cli_args.head.parent / "metrics_summary.json"
    save_run_summary(output, args=args, metrics_by_split=metrics)
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = evaluate_head(parse_args(argv))
    logger.info("Saved evaluation summary to %s", output)


if __name__ == "__main__":
    main()
