"""Compare ADNI patch-feature PCA maps across four DINOv3 backbones.

The script draws the same patient-diverse ADNI slices for every model, then
renders an original image alongside independently whitened PCA maps of its
patch tokens.  DINOv3 and BrainDINO use the repository's standard weights;
provide the two adapted-backbone checkpoints on the command line.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    DEFAULT_ADNI_TASK,
    build_adni_transform,
)
from dinov3_baseline import (  # noqa: E402
    DEFAULT_BRAINDINO_WEIGHTS,
    DEFAULT_DINOV3_WEIGHTS,
    DINOV3_REPO,
    load_braindino_encoder,
    load_dinov3_encoder,
)
from utils.pca_dino_backbones import (  # noqa: E402
    ADNI_DEFAULT_ROOT,
    MEAN,
    STD,
    _batched_patch_features,
    load_pca_backbone,
    pca_rgb,
)


LOGGER = logging.getLogger("features_comparison")
DEFAULT_IMAGE_SIZE = 224


def load_images(args: argparse.Namespace):
    """Return the shared, reproducibly sampled ADNI image batch and labels."""
    dataset = ADNIClassificationDataset(
        root=args.data_root,
        csv_path=args.csv_path,
        task=args.adni_task,
        z_min=args.z_min,
        z_max=args.z_max,
        transform=build_adni_transform(args.image_size, augment=False),
    )
    indices = _sample_indices(dataset, args.n_images, args.seed)
    images = torch.stack([dataset[index][0] for index in indices])
    image_ids = [dataset.get_image_id(index) for index in indices]
    labels = [str(dataset.get_target(index)) for index in indices]
    return images, image_ids, labels


def _sample_indices(dataset, n_images: int, seed: int | None) -> list[int]:
    """Mirror the PCA helper's patient-diverse sampling and return its indices."""
    if n_images <= 0:
        raise ValueError("--n-images must be positive")
    n_images = min(n_images, len(dataset))
    rng = np.random.default_rng(seed)
    by_patient = {}
    for index in range(len(dataset)):
        by_patient.setdefault(dataset.get_patient_id(index), []).append(index)
    patient_ids = list(by_patient)
    rng.shuffle(patient_ids)
    indices = [int(rng.choice(by_patient[patient])) for patient in patient_ids[:n_images]]
    if len(indices) < n_images:
        chosen = set(indices)
        remaining = np.asarray(
            [index for index in range(len(dataset)) if index not in chosen]
        )
        rng.shuffle(remaining)
        indices.extend(remaining[: n_images - len(indices)].tolist())
    return indices


def save_comparison_plot(
    images: torch.Tensor,
    image_ids: list[str],
    labels: list[str],
    pca_maps: np.ndarray,
    model_names: list[str],
    output: Path,
) -> None:
    """Save an original-plus-PCA grid with consistent, publication-ready styling."""
    originals = (
        (images * STD + MEAN)
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    n_images, n_models = pca_maps.shape[:2]
    figure, axes = plt.subplots(
        n_images,
        n_models + 1,
        figsize=(3.1 * (n_models + 1), 3.15 * n_images + 0.45),
        squeeze=False,
        layout="constrained",
    )
    titles = ["ADNI slice", *model_names]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12, fontweight="semibold", pad=10)

    for row in range(n_images):
        original_axis = axes[row, 0]
        original_axis.imshow(originals[row], cmap="gray")
        original_axis.axis("off")
        original_axis.set_ylabel(
            f"{image_ids[row]}\nclass {labels[row]}",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )
        for column in range(n_models):
            axis = axes[row, column + 1]
            axis.imshow(pca_maps[row, column], interpolation="nearest")
            axis.axis("off")

    figure.suptitle(
        "ADNI patch-feature PCA comparison",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ADNI_DEFAULT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    parser.add_argument(
        "--extended-pretraining-checkpoint",
        "--extended-pretraining-weights",
        dest="extended_pretraining_checkpoint",
        type=Path,
        required=True,
        help="DCP directory or merged teacher .pth from extended pretraining.",
    )
    parser.add_argument(
        "--three-d-aware-finetuning-checkpoint",
        "--3d-aware-finetuning-checkpoint",
        "--three-d-aware-finetuning-weights",
        dest="three_d_aware_finetuning_checkpoint",
        type=Path,
        required=True,
        help="DCP directory or merged teacher .pth from 3-D-aware fine tuning.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("features_comparison.png")
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.n_images <= 0:
        parser.error("--n-images must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.image_size <= 0 or args.image_size % 16:
        parser.error("--image-size must be positive and divisible by 16")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images, image_ids, labels = load_images(args)
    grid = (args.image_size // 16, args.image_size // 16)
    model_loaders = [
        (
            "DINOv3",
            lambda: load_dinov3_encoder(
                repo_dir=args.dinov3_repo,
                weights=DEFAULT_DINOV3_WEIGHTS,
                device=device,
            ),
        ),
        (
            "BrainDINO",
            lambda: load_braindino_encoder(
                repo_dir=args.dinov3_repo,
                weights=DEFAULT_BRAINDINO_WEIGHTS,
                device=device,
            ),
        ),
        (
            "Extended pretraining",
            lambda: load_pca_backbone(
                args.extended_pretraining_checkpoint,
                repo_dir=args.dinov3_repo,
                device=device,
            ),
        ),
        (
            "3-D-aware fine tuning",
            lambda: load_pca_backbone(
                args.three_d_aware_finetuning_checkpoint,
                repo_dir=args.dinov3_repo,
                device=device,
            ),
        ),
    ]
    pca_maps = np.empty(
        (len(images), len(model_loaders), grid[0], grid[1], 3), dtype=np.float32
    )
    for column, (name, load_model) in enumerate(model_loaders):
        LOGGER.info("Loading %s (%d/%d)", name, column + 1, len(model_loaders))
        model = load_model()
        features = _batched_patch_features(
            model, images, grid, device, args.batch_size
        )
        for row in range(len(images)):
            pca_maps[row, column] = pca_rgb(features[row], grid)
        del features, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_comparison_plot(
        images,
        image_ids,
        labels,
        pca_maps,
        [name for name, _ in model_loaders],
        args.output,
    )
    LOGGER.info("Saved feature comparison to %s", args.output)


if __name__ == "__main__":
    main()
