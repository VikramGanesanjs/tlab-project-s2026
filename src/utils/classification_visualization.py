"""Visualize labeled MRI slices in CLS-token UMAP space.

Example::

    python -m utils.classification_visualization \
        --checkpoint /path/to/classification/checkpoint \
        --dinov3-repo /path/to/dinov3 \
        --data-root /path/to/data/ADNI \
        --n-slices 100 \
        --output umap_adni.png

The default ``auto`` encoder detects BrainDINO/DINOv3 ``.pth`` files and sends
distributed-checkpoint directories through the shared checkpoint loader. The checkpoint
is used as a DINOv3-compatible backbone; colors come from the dataset labels.

The ``volume`` subcommand extracts trained multi-slice-transformer (MST) volume
tokens from a saved ``classification.run`` run, for example::

    python -m utils.classification_visualization volume \\
        /path/to/run_summary.json --n-volumes 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPO_ROOT = SRC_DIR.parent
DINOV3_DEFAULT_REPO = Path("/common/ganesanv/tlab/opt/dinov3")
ADNI_DEFAULT_DATA_ROOT = Path("/common/ganesanv/tlab/data/ADNI")
DUKE_DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcia" / "duke_breast_cancer_processed"

from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    DEFAULT_ADNI_TASK,
    build_adni_transform,
)
from datasets.duke import DukeClassificationDataset, build_duke_transform  # noqa: E402
from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from classification import model as classification_model  # noqa: E402
from classification import run as classification_run  # noqa: E402
from classification import train as classification_train  # noqa: E402
from dinov3_baseline import (  # noqa: E402
    load_braindino_encoder,
    load_dinov3_encoder,
    load_encoder,
)
from utils.vit_lora import add_lora_to_vit, freeze_non_lora_parameters  # noqa: E402

LOGGER = logging.getLogger("classification_visualization")
ENCODER_CHOICES = ("auto", "custom", "dinov3", "braindino")


def build_dataset(args):
    if args.dataset == "adni":
        return (
            ADNIClassificationDataset(
                root=args.data_root,
                csv_path=args.csv_path,
                task=args.adni_task,
                z_min=args.z_min,
                z_max=args.z_max,
                transform=build_adni_transform(args.image_size, augment=False),
            ),
            None,
        )
    return (
        DukeClassificationDataset(
            root=args.data_root,
            scan=args.scan,
            z_min=args.z_min,
            z_max=args.z_max,
            include_bilateral=args.include_bilateral,
            transform=build_duke_transform(args.image_size, augment=False),
        ),
        ("non-cancerous", "cancerous"),
    )


def _slice_z(dataset, index: int) -> int:
    """Read the axial z-index from the repository dataset entry."""
    entry = dataset._entries[index]  # Both classification datasets store z in entries.
    if len(entry) == 2:  # ADNI: (record, z)
        return int(entry[1])
    return int(entry[4])  # Duke: (patient, side, path, depth, z, label)


def sample_slices(dataset, n_slices: int, seed: int | None):
    if n_slices < 3:
        raise ValueError("--n-slices must be at least 3 for UMAP")
    if n_slices > len(dataset):
        LOGGER.warning("Requested %d slices, dataset has %d; using all slices", n_slices, len(dataset))
        n_slices = len(dataset)
    if n_slices < 3:
        raise ValueError("The dataset must contain at least 3 slices for UMAP")
    requested_slices = n_slices
    rng = np.random.default_rng(seed)
    by_z = {}
    for index in range(len(dataset)):
        by_z.setdefault(_slice_z(dataset, index), []).append(index)

    z_to_patients = {}
    for z, z_indices in by_z.items():
        patients = {}
        for index in z_indices:
            patients.setdefault(dataset.get_patient_id(index), []).append(index)
        z_to_patients[z] = patients
    eligible_z = [z for z, patients in z_to_patients.items() if len(patients) >= n_slices]
    if eligible_z:
        z = int(rng.choice(eligible_z))
    else:
        z = max(z_to_patients, key=lambda value: len(z_to_patients[value]))
        n_slices = len(z_to_patients[z])
        LOGGER.warning(
            "No z-index has %d unique patients; using z=%d with %d patients",
            requested_slices,
            z,
            n_slices,
        )
    patient_ids = list(z_to_patients[z])
    rng.shuffle(patient_ids)
    indices = [int(rng.choice(z_to_patients[z][patient])) for patient in patient_ids[:n_slices]]
    indices = np.asarray(indices, dtype=np.int64)
    images, labels = [], []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image)
        labels.append(int(label))
    LOGGER.info(
        "Selected %d slices at z=%d from %d unique patients",
        len(indices),
        z,
        len({dataset.get_patient_id(int(index)) for index in indices}),
    )
    return torch.stack(images), np.asarray(labels, dtype=np.int64), indices


def _looks_like_braindino(checkpoint: Path) -> bool:
    """Recognize the teacher/backbone container used by BrainDINO .pth files."""
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("teacher"), dict):
        return False
    teacher_keys = loaded["teacher"].keys()
    return any(str(key).startswith("backbone.") for key in teacher_keys)


def load_encoder(args, device: torch.device) -> torch.nn.Module:
    # Distributed checkpoints are handled by the shared checkpoint loader.
    if args.checkpoint.is_dir():
        return load_custom_dinov3_encoder(
            checkpoint=args.checkpoint,
            repo_dir=args.dinov3_repo,
            device=device,
            encoder_training="frozen",
            lora_rank=8,
        )

    encoder = args.encoder
    if encoder == "auto":
        encoder = "braindino" if _looks_like_braindino(args.checkpoint) else "dinov3"
    if encoder == "dinov3":
        return load_dinov3_encoder(
            repo_dir=args.dinov3_repo,
            weights=args.checkpoint,
            model_name=args.dinov3_model,
            device=device,
        )
    if encoder == "braindino":
        return load_braindino_encoder(
            repo_dir=args.dinov3_repo,
            weights=args.checkpoint,
            device=device,
        )
    return load_custom_dinov3_encoder(
        checkpoint=args.checkpoint,
        repo_dir=args.dinov3_repo,
        device=device,
        encoder_training="frozen",
        lora_rank=8,
    )


@torch.inference_mode()
def extract_cls_tokens(
    model, images: torch.Tensor, device: torch.device, batch_size: int
) -> np.ndarray:
    tokens = []
    for start in range(0, len(images), batch_size):
        output = model.forward_features(images[start : start + batch_size].to(device))
        tokens.append(output["x_norm_clstoken"].float().cpu().numpy())
    return np.concatenate(tokens, axis=0)


def run_umap(features: np.ndarray, seed: int | None, n_neighbors: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("Install UMAP with: pip install umap-learn") from exc
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(n_neighbors, len(features) - 1),
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_umap(
    coordinates: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output: Path,
) -> None:
    colors = plt.get_cmap("tab10")(np.arange(len(class_names)) % 10)
    fig, ax = plt.subplots(figsize=(14, 11))
    for label, name in enumerate(class_names):
        selected = labels == label
        if selected.any():
            ax.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=36,
                color=colors[label],
                alpha=0.8,
                label=name,
            )
    ax.set_title("MRI slices in CLS-token UMAP space, colored by diagnosis")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Diagnosis", loc="best")
    ax.margins(0.08)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="auto")
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_DEFAULT_REPO)
    parser.add_argument("--dinov3-model", default="dinov3_vitb16")
    parser.add_argument("--dataset", choices=("adni", "duke"), default="adni")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--scan", default="pre", help="Duke scan type, e.g. pre or T1")
    parser.add_argument("--include-bilateral", action="store_true")
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--n-slices", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("classification_umap.png"))
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def parse_volume_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample multi-slice volumes and save their trained MST global tokens. "
            "The run summary supplies the dataset, encoder, and MST architecture."
        )
    )
    parser.add_argument(
        "run_summary",
        type=Path,
        help="Path to a classification run_summary.json file",
    )
    parser.add_argument(
        "--n-volumes",
        type=int,
        default=100,
        help="Number of distinct dataset volumes to sample (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Volumes processed per inference batch (default: 8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mst-checkpoint",
        type=Path,
        default=None,
        help="Override the MST checkpoint (defaults to run-summary directory/best_mst.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz file (default: <run-summary directory>/volume_tokens.npz)",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _read_run_parameters(run_summary: Path) -> Mapping[str, Any]:
    if not run_summary.is_file():
        raise FileNotFoundError(f"Run summary not found: {run_summary}")
    with run_summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict) or not isinstance(summary.get("parameters"), dict):
        raise ValueError(f"{run_summary} must contain a mapping under 'parameters'")
    return summary["parameters"]


def _find_mst_checkpoint(run_summary: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"MST checkpoint not found: {override}")
        return override
    # Current classification runs write .pt. Retain .py as a fallback for old runs
    # that may have used that extension.
    for name in ("best_mst.pt", "best_mst.py"):
        candidate = run_summary.parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find MST checkpoint beside the run summary; expected "
        f"{run_summary.parent / 'best_mst.pt'}"
    )


def _run_value(
    parameters: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    """Prefer the run summary, falling back to checkpoint metadata for old runs."""
    value = parameters.get(name)
    if value is not None:
        return value
    return checkpoint.get(name, default)


def _build_volume_dataset(parameters: Mapping[str, Any]):
    dataset_name = parameters.get("dataset")
    if dataset_name not in classification_run.DATASET_CHOICES:
        raise ValueError(
            f"Run summary has dataset={dataset_name!r}; expected one of "
            f"{classification_run.DATASET_CHOICES}"
        )
    values = dict(parameters)
    values["dataset"] = dataset_name
    values["n_slices"] = int(values.get("n_slices", 8))
    values["image_size"] = int(values.get("image_size", 224))
    values["adni_task"] = values.get("adni_task") or DEFAULT_ADNI_TASK
    values["scan"] = values.get("scan") or "pre"
    values["include_bilateral"] = bool(values.get("include_bilateral", False))
    if values.get("data_root") is None:
        values["data_root"] = (
            classification_run.ADNI_DEFAULT_ROOT
            if dataset_name == "adni"
            else classification_run.DEFAULT_DATA_ROOT
        )
    # This mirrors classification training validation/test inference: samples are resampled
    # and normalized exactly as training expects, but no random augmentation is
    # applied while extracting a representation.
    values["augment"] = False
    return classification_train.build_dataset(argparse.Namespace(**values), augment=False)


def _load_volume_model(
    parameters: Mapping[str, Any], checkpoint_path: Path, device: torch.device
) -> classification_model.MultiSliceDinoModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(
            f"{checkpoint_path} is not a classification checkpoint with a 'model' state"
        )
    checkpoint: Mapping[str, Any] = payload
    dataset_name = _run_value(parameters, checkpoint, "dataset")
    adni_task = _run_value(parameters, checkpoint, "adni_task", DEFAULT_ADNI_TASK)
    num_classes, _, _ = classification_train.task_config(dataset_name, adni_task=adni_task)
    num_classes = int(_run_value(parameters, checkpoint, "num_classes", num_classes))
    encoder_name = str(_run_value(parameters, checkpoint, "encoder", "dinov3"))
    encoder_training = str(_run_value(parameters, checkpoint, "encoder_training", "frozen"))
    weights_value = _run_value(parameters, checkpoint, "weights")
    weights = Path(weights_value) if weights_value is not None else None
    repo_dir = Path(_run_value(parameters, checkpoint, "dinov3_repo", DINOV3_DEFAULT_REPO))
    model_name = str(_run_value(parameters, checkpoint, "model_name", "dinov3_vitb16"))
    lora_rank = int(_run_value(parameters, checkpoint, "lora_r", 16))

    # Keep this loader construction identical to classification.train(), including
    # the special distributed-checkpoint path and optional LoRA adapters.
    if weights is not None and encoder_name == "dinov3":
        encoder = load_custom_dinov3_encoder(
            checkpoint=weights,
            repo_dir=repo_dir,
            device=device,
            encoder_training=encoder_training,
            lora_rank=lora_rank,
        )
    else:
        encoder = load_encoder(
            encoder_name,
            device=device,
            weights=weights,
            repo_dir=repo_dir,
            model_name=model_name,
        )
        if encoder_training == "lora":
            add_lora_to_vit(encoder, r=lora_rank)
            freeze_non_lora_parameters(encoder)

    d_model = int(_run_value(parameters, checkpoint, "d_model", 768))
    hidden_dim = int(_run_value(parameters, checkpoint, "hidden_dim", d_model))
    model = classification_model.MultiSliceDinoModel(
        encoder,
        n_slices=int(_run_value(parameters, checkpoint, "n_slices", 8)),
        features=str(_run_value(parameters, checkpoint, "features", "cls")),
        n_cls_tokens=int(_run_value(parameters, checkpoint, "n_cls_tokens", 1)),
        aggregator=str(_run_value(parameters, checkpoint, "slice_aggregator", "transformer")),
        d_model=d_model,
        depth=int(_run_value(parameters, checkpoint, "mst_depth", 2)),
        n_heads=int(_run_value(parameters, checkpoint, "mst_heads", 12)),
        ffn_dim=int(_run_value(parameters, checkpoint, "mst_ffn_dim", 3072)),
        dropout=float(_run_value(parameters, checkpoint, "mst_dropout", 0.1)),
        hidden_dim=hidden_dim or d_model,
        num_classes=num_classes,
        encoder_training=encoder_training,
    ).to(device)
    model.load_trainable_state_dict(payload["model"])
    model.eval()
    return model


@torch.inference_mode()
def extract_volume_tokens(
    model: classification_model.MultiSliceDinoModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the MST global tokens and labels in loader order."""
    tokens, labels = [], []
    model.eval()
    for images, targets in loader:
        volume_tokens = model.extract_volume_token(
            images.to(device, non_blocking=True)
        )
        tokens.append(volume_tokens.float().cpu().numpy())
        labels.append(torch.as_tensor(targets).cpu().numpy())
    if not tokens:
        raise RuntimeError("No volumes were available for token extraction")
    return np.concatenate(tokens, axis=0), np.concatenate(labels, axis=0)


def volume(argv: Sequence[str] | None = None) -> Path:
    """Run the ``volume`` command and return the saved token archive path."""
    args = parse_volume_args(argv)
    if args.n_volumes <= 0:
        raise ValueError("--n-volumes must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    run_summary = args.run_summary.resolve()
    parameters = _read_run_parameters(run_summary)
    checkpoint_path = _find_mst_checkpoint(run_summary, args.mst_checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = _build_volume_dataset(parameters)
    if len(dataset) == 0:
        raise RuntimeError("The configured multi-slice dataset contains no volumes")
    n_volumes = min(args.n_volumes, len(dataset))
    if n_volumes < args.n_volumes:
        LOGGER.warning("Requested %d volumes, dataset has %d; using all volumes", args.n_volumes, len(dataset))
    indices = np.random.default_rng(args.seed).choice(len(dataset), size=n_volumes, replace=False)
    indices = np.asarray(indices, dtype=np.int64)
    patient_ids = np.asarray([str(dataset.get_patient_id(int(index))) for index in indices])
    get_image_id = getattr(dataset, "get_image_id", None)
    image_ids = np.asarray(
        [str(get_image_id(int(index))) if get_image_id is not None else "" for index in indices]
    )
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info("Loading MST checkpoint %s on %s", checkpoint_path, device)
    model = _load_volume_model(parameters, checkpoint_path, device)
    tokens, labels = extract_volume_tokens(model, loader, device)
    output = args.output or run_summary.parent / "volume_tokens.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        global_tokens=tokens,
        labels=labels,
        indices=indices,
        patient_ids=patient_ids,
        image_ids=image_ids,
        run_summary=str(run_summary),
        checkpoint=str(checkpoint_path),
    )
    LOGGER.info("Saved %d volume tokens with shape %s to %s", len(tokens), tokens.shape, output)
    return output


def main(argv: Sequence[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if argv and argv[0] == "volume":
        volume(argv[1:])
        return

    args = parse_args(argv)
    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by 16")
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.data_root is None:
        args.data_root = (
            ADNI_DEFAULT_DATA_ROOT
            if args.dataset == "adni"
            else DUKE_DEFAULT_DATA_ROOT
        )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, default_class_names = build_dataset(args)
    images, labels, indices = sample_slices(dataset, args.n_slices, args.seed)
    class_names = default_class_names or tuple(dataset.class_names)
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise ValueError(f"Dataset labels {sorted(set(labels.tolist()))} do not match class names {class_names}")
    LOGGER.info("Sampled %d slices from %s", len(images), args.dataset.upper())

    model = load_encoder(args, device)
    features = extract_cls_tokens(model, images, device, args.batch_size)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    LOGGER.info("CLS features: %s", features.shape)

    coordinates = run_umap(features, args.seed, args.n_neighbors)
    plot_umap(coordinates, labels, class_names, args.output)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
