"""Command-line and YAML configuration entry point for classification."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import yaml

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.load_dinov3 import (  # noqa: E402
    DINOV3_REPO,
    ENCODER_CHOICES,
    FEATURE_CHOICES,
    REPO_ROOT,
)
from datasets.adni import (  # noqa: E402
    ADNI_TASK_CHOICES,
    DEFAULT_ADNI_TASK,
    DEFAULT_ROOT as ADNI_DEFAULT_ROOT,
)
from datasets.cq500 import (  # noqa: E402
    CQ500_TASK_CHOICES,
    DEFAULT_ROOT as CQ500_DEFAULT_ROOT,
)
from datasets.organmnist3d import DEFAULT_ROOT as ORGANMNIST3D_DEFAULT_ROOT  # noqa: E402
from classification.train import train  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"
DATASET_CHOICES = ("duke", "adni", "cq500", "organmnist3d")
AGGREGATOR_CHOICES = ("transformer", "mean")
ENCODER_TRAINING_CHOICES = ("frozen", "lora")
EARLY_STOPPING_METRIC_CHOICES = ("bce_loss", "f1", "auroc")


def _optional_n_slices(value: object) -> Optional[int]:
    """Parse a positive slice count or the explicit ``null``/``none`` value."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return None
    return int(value)

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
        help=(
            "duke: binary breast cancer; adni: diagnosis task selected by "
            "--adni-task; cq500: task selected by --cq500-task; "
            "organmnist3d: official 11-class volume splits"
        ),
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
        "--cq500-task",
        choices=list(CQ500_TASK_CHOICES),
        default="ich",
        help="CQ500 task: ich is binary BCE; subtype is five-logit multi-label BCE",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (defaults to the selected dataset's standard path)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Metadata CSV (ADNI or CQ500 reads.csv; ignored by Duke and OrganMNIST3D)",
    )
    parser.add_argument("--scan", type=str, default="pre")
    parser.add_argument(
        "--n-slices",
        type=_optional_n_slices,
        default=8,
        help=(
            "Number of depth slices after resampling; use null only for CQ500 "
            "to retain native depth (capped at 128)"
        ),
    )
    parser.add_argument(
        "--cq500-max-slices",
        type=int,
        default=128,
        help=(
            "Resampling target for native-depth CQ500 volumes longer than this "
            "limit when --n-slices is null (default: 128)"
        ),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--include-bilateral",
        action="store_true",
        help="Include bilateral Duke cases (ignored by ADNI, CQ500, and OrganMNIST3D)",
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
        "--reduce-lr-on-plateau",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reduce the learning rate when the selected validation metric plateaus "
            "(default; use --no-reduce-lr-on-plateau for a fixed learning rate)"
        ),
    )
    parser.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=0.1,
        help="Factor applied when ReduceLROnPlateau triggers (default: 0.1)",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=3,
        help="Validation epochs without improvement before reducing LR (default: 3)",
    )
    parser.add_argument(
        "--lr-plateau-threshold",
        type=float,
        default=0.005,
        help="Absolute validation-metric improvement required to reset LR patience",
    )
    parser.add_argument(
        "--cosine-lr",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --reduce-lr-on-plateau; "
            "cosine annealing is no longer used"
        ),
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate allowed by ReduceLROnPlateau (default: 0)",
    )
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of deterministic patient-level cross-validation folds",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Zero-based validation-fold index; the next fold is used for test",
    )
    parser.add_argument(
        "--data-seed", type=int, default=0,
        help="Random seed used to assign patients to cross-validation folds",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=1.0,
        help="Class-balanced fraction of the selected training patients to use",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help=(
            "Legacy JSON patient split with train, val, and test lists. When "
            "provided, it takes precedence over --n-folds/--fold."
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
        args.data_root = {
            "duke": DEFAULT_DATA_ROOT,
            "adni": ADNI_DEFAULT_ROOT,
            "cq500": CQ500_DEFAULT_ROOT,
            "organmnist3d": ORGANMNIST3D_DEFAULT_ROOT,
        }[args.dataset]
    if args.n_slices is not None and args.n_slices <= 0:
        parser.error("--n-slices must be positive")
    if args.n_slices is None:
        if args.dataset != "cq500":
            parser.error("--n-slices null is supported only for CQ500")
    if args.cq500_max_slices <= 0:
        parser.error("--cq500-max-slices must be positive")
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
    if args.min_lr > args.lr:
        parser.error("--min-lr cannot exceed --lr")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        parser.error("--lr-plateau-factor must be in (0, 1)")
    if args.lr_plateau_patience < 0:
        parser.error("--lr-plateau-patience must be non-negative")
    if args.lr_plateau_threshold < 0:
        parser.error("--lr-plateau-threshold must be non-negative")
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
    if args.n_folds < 3:
        parser.error("--n-folds must be at least 3")
    if not 0 <= args.fold < args.n_folds:
        parser.error("--fold must be in [0, --n-folds)")
    if not 0.0 < args.train_ratio <= 1.0:
        parser.error("--train-ratio must be in (0, 1]")
    if args.splits_file is not None and not args.splits_file.is_file():
        parser.error(f"--splits-file does not exist or is not a file: {args.splits_file}")
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
