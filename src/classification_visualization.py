"""Visualize labeled MRI slices in CLS-token UMAP space.

Example::

    python src/classification_visualization.py \
        --checkpoint /path/to/dino_mst/checkpoint \
        --dinov3-repo /path/to/dinov3 \
        --data-root /path/to/data/ADNI \
        --n-slices 100 \
        --output umap_adni.png

The default ``auto`` encoder detects BrainDINO/DINOv3 ``.pth`` files and sends
distributed-checkpoint directories through the shared checkpoint loader. The checkpoint
is used as a DINOv3-compatible backbone; colors come from the dataset labels.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
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
from merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from dinov3_baseline import load_braindino_encoder, load_dinov3_encoder  # noqa: E402

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


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
