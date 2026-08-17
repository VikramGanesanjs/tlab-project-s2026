"""PCA visualization of DINO patch features.

Use ``evolution`` as a subcommand to compare a series of distributed
checkpoints on the same sampled images.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from dinov3_baseline import load_braindino_encoder, load_dinov3_encoder  # noqa: E402
from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    DEFAULT_ADNI_TASK,
    build_adni_transform,
)
from datasets.duke import DukeClassificationDataset, build_duke_transform  # noqa: E402

LOGGER = logging.getLogger("pca_dino_backbones")
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
ADNI_DEFAULT_ROOT = Path("/common/ganesanv/tlab/data/ADNI")
DUKE_DEFAULT_ROOT = Path("/common/ganesanv/tlab/data/tcia/duke_breast_cancer_processed")
DINOV3_DEFAULT_REPO = Path("/common/ganesanv/tlab/opt/dinov3")
DEFAULT_IMAGE_SIZE = 224


def load_pca_backbone(
    checkpoint: Path,
    *,
    repo_dir: Path,
    device: torch.device,
    lora_rank: int = 8,
):
    """Load a DINOv3 PCA backbone from DCP or either merged ``.pth`` format.

    ``load_custom_dinov3_encoder`` selects the SSL ViT configuration for the
    plain merged export and the released DINOv3 configuration for the
    ``--hub-compatible`` export by inspecting the presence of storage tokens.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir() and not checkpoint.is_file():
        raise FileNotFoundError(f"PCA backbone checkpoint not found: {checkpoint}")
    checkpoint_kind = "merged backbone .pth" if checkpoint.is_file() else "distributed checkpoint"
    LOGGER.info("Loading %s (%s)", checkpoint, checkpoint_kind)
    return load_custom_dinov3_encoder(
        checkpoint=checkpoint,
        repo_dir=repo_dir,
        device=device,
        encoder_training="frozen",
        lora_rank=lora_rank,
    )


def sample_adni_slice(args):
    """Construct the repository ADNI dataset and return one random sample."""
    dataset = ADNIClassificationDataset(
        root=args.data_root,
        csv_path=args.csv_path,
        task=args.adni_task,
        z_min=args.z_min,
        z_max=args.z_max,
        transform=build_adni_transform(args.image_size, augment=False),
    )
    if args.sample_index is None:
        generator = np.random.default_rng(args.seed)
        index = int(generator.integers(len(dataset)))
    else:
        index = args.sample_index
        if not 0 <= index < len(dataset):
            raise IndexError(f"--sample-index must be in [0, {len(dataset)}), got {index}")
    image, label = dataset[index]
    LOGGER.info(
        "Sampled ADNI index=%d image_id=%s patient=%s label=%s",
        index,
        dataset.get_image_id(index),
        dataset.get_patient_id(index),
        label,
    )
    return image.unsqueeze(0), index


@torch.inference_mode()
def patch_features(model, image, grid, device):
    # DINOv3-family backbones compute RoPE from the runtime patch grid. For
    # ViT-B/16, 512x512 therefore arrives here as a 32x32 patch grid.
    output = model.forward_features(image.to(device))
    features = output["x_norm_patchtokens"]
    expected = grid[0] * grid[1]
    if features.shape[1] != expected:
        raise ValueError(f"Backbone returned {features.shape[1]} patches, expected {expected}")
    return features.float().cpu().numpy()


def pca_rgb(features, grid):
    """Fit the whitened PCA used by the DINOv3 reference notebook."""
    projected = PCA(n_components=3, whiten=True).fit_transform(features)
    projected -= projected.min(axis=0)
    projected /= np.maximum(projected.max(axis=0), 1e-12)
    return projected.reshape(*grid, 3)


def save_plot(image, maps, output):
    display = (image[0].cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(display)
    axes[0].set_title("ADNI slice")
    axes[0].axis("off")
    for axis, (title, projection) in zip(axes[1:], maps):
        axis.imshow(projection, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _evolution_dataset(args):
    root = args.data_root or (
        ADNI_DEFAULT_ROOT if args.dataset == "adni" else DUKE_DEFAULT_ROOT
    )
    if args.dataset == "adni":
        return ADNIClassificationDataset(
            root=root,
            csv_path=args.csv_path,
            task=args.adni_task,
            z_min=args.z_min,
            z_max=args.z_max,
            transform=build_adni_transform(args.image_size, augment=False),
        )
    return DukeClassificationDataset(
        root=root,
        scan=args.scan,
        z_min=args.z_min,
        z_max=args.z_max,
        include_bilateral=args.include_bilateral,
        transform=build_duke_transform(args.image_size, augment=False),
    )


def _sample_evolution_images(dataset, n_images: int, seed: int | None):
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
        remaining = np.asarray(
            [index for index in range(len(dataset)) if index not in set(indices)]
        )
        rng.shuffle(remaining)
        indices.extend(remaining[: n_images - len(indices)].tolist())
    images, names = [], []
    for index in indices:
        image, _ = dataset[index]
        images.append(image)
        if hasattr(dataset, "get_image_id"):
            names.append(dataset.get_image_id(index))
        else:
            names.append(f"{dataset.get_patient_id(index)}_{dataset.get_side(index)}")
    return torch.stack(images), names


def _checkpoint_sort_key(path: Path):
    numbers = re.findall(r"\d+", path.name)
    return (0, int(numbers[-1])) if numbers else (1, path.name)


def discover_distributed_checkpoints(parent: Path):
    """Find distributed-checkpoint children under a parent directory."""
    if not parent.is_dir():
        raise FileNotFoundError(f"Checkpoint parent not found: {parent}")

    def is_checkpoint(path: Path) -> bool:
        if (path / ".metadata").is_file():
            return True
        return any(
            child.is_dir() and (child / ".metadata").is_file()
            for child in path.iterdir()
        )

    checkpoints = sorted(
        (child for child in parent.iterdir() if child.is_dir() and is_checkpoint(child)),
        key=_checkpoint_sort_key,
    )
    if not checkpoints:
        raise ValueError(f"No distributed checkpoint subfolders found under {parent}")
    return checkpoints


def select_checkpoint_stride(checkpoints: list[Path], stride: int) -> list[Path]:
    """Keep every ``stride``-th discovered checkpoint.

    The first checkpoint is retained, so ``stride=3`` selects entries 0, 3,
    6, and so on.  The final partial stride is intentionally retained; the
    number of checkpoints need not be divisible by ``stride``.
    """
    if stride <= 0:
        raise ValueError("--checkpoint-stride must be positive")
    if stride == 1:
        return checkpoints

    return checkpoints[::stride]


def _batched_patch_features(model, images, grid, device, batch_size):
    batches = []
    for start in range(0, len(images), batch_size):
        batches.append(
            patch_features(model, images[start : start + batch_size], grid, device)
        )
    return np.concatenate(batches, axis=0)


def save_evolution_plot(images, pca_maps, image_names, checkpoint_names, output):
    """Save original images plus PCA maps, with images as rows."""
    n_images, n_checkpoints = pca_maps.shape[:2]
    originals = (
        (images * STD + MEAN)
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    fig, axes = plt.subplots(
        n_images,
        n_checkpoints + 1,
        figsize=(3.2 * (n_checkpoints + 1), 3.2 * n_images),
        squeeze=False,
    )
    for row in range(n_images):
        axes[row, 0].imshow(originals[row])
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original")
        for column in range(n_checkpoints):
            axis = axes[row, column + 1]
            axis.imshow(pca_maps[row, column], interpolation="nearest")
            axis.axis("off")
            if row == 0:
                axis.set_title(checkpoint_names[column])
        axes[row, 0].set_ylabel(
            str(image_names[row]), rotation=0, labelpad=35, va="center"
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_evolution(args):
    """Compute and plot patch-feature PCA maps for a checkpoint series."""
    dataset = _evolution_dataset(args)
    images, image_names = _sample_evolution_images(
        dataset, args.n_images, args.seed
    )
    checkpoints = select_checkpoint_stride(
        discover_distributed_checkpoints(args.checkpoint_parent),
        args.checkpoint_stride,
    )
    LOGGER.info(
        "Selected %d checkpoint(s) with stride=%d",
        len(checkpoints),
        args.checkpoint_stride,
    )
    grid = (args.image_size // 16, args.image_size // 16)
    LOGGER.info("Evolution image size=%dx%d, patch grid=%s", args.image_size, args.image_size, grid)
    pca_maps = np.empty(
        (len(images), len(checkpoints), grid[0], grid[1], 3), dtype=np.float32
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for column, checkpoint in enumerate(checkpoints):
        LOGGER.info("Loading checkpoint %s (%d/%d)", checkpoint.name, column + 1, len(checkpoints))
        model = load_pca_backbone(
            checkpoint=checkpoint,
            repo_dir=args.dinov3_repo,
            device=device,
            lora_rank=8,
        )
        features = _batched_patch_features(
            model, images, grid, device, args.batch_size
        )
        for row in range(len(images)):
            pca_maps[row, column] = pca_rgb(features[row], grid)
        del features, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_evolution_plot(
        images,
        pca_maps,
        image_names,
        [checkpoint.name for checkpoint in checkpoints],
        args.output,
    )
    LOGGER.info("Saved checkpoint evolution plot: %s", args.output)


def parse_evolution_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pca_dino_backbones.py evolution",
        description="Plot patch-feature PCA evolution across distributed checkpoints.",
    )
    parser.add_argument("--checkpoint-parent", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-stride",
        "--checkpoint-modulo",
        "--every-nth-checkpoint",
        dest="checkpoint_stride",
        type=int,
        default=1,
        help=(
            "Keep every Nth checkpoint in discovery order, starting with the "
            "first (default: 1)."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--dataset", choices=("adni", "duke"), default="adni")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--scan", default="pre")
    parser.add_argument("--include-bilateral", action="store_true")
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=Path("pca_checkpoint_evolution.png"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.image_size <= 0 or args.image_size % 16:
        parser.error("--image-size must be positive and divisible by 16")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.checkpoint_stride <= 0:
        parser.error("--checkpoint-stride must be positive")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ADNI_DEFAULT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sampling")
    parser.add_argument("--dinov3-checkpoint", type=Path, required=True)
    parser.add_argument("--braindino-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--custom-checkpoint",
        type=Path,
        required=True,
        help=(
            "DINOv3 DCP directory, plain merged teacher .pth, or "
            "--hub-compatible merged teacher .pth"
        ),
    )
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_DEFAULT_REPO)
    parser.add_argument("--dinov3-model", default="dinov3_vitb16")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--output", type=Path, default=Path("pca_adni_slice.png"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "evolution":
        args = parse_evolution_args(sys.argv[2:])
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        plot_checkpoint_evolution(args)
        return
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image, index = sample_adni_slice(args)
    grid = (args.image_size // 16, args.image_size // 16)
    LOGGER.info("Image size=%dx%d, patch grid=%s", args.image_size, args.image_size, grid)

    model_loaders = [
        (
            "DINOv3",
            lambda: load_dinov3_encoder(
                repo_dir=args.dinov3_repo,
                weights=args.dinov3_checkpoint,
                model_name=args.dinov3_model,
                device=device,
            ),
        ),
        (
            "BrainDINO",
            lambda: load_braindino_encoder(
                repo_dir=args.dinov3_repo,
                weights=args.braindino_checkpoint,
                device=device,
            ),
        ),
        (
            "Custom teacher backbone",
            lambda: load_pca_backbone(
                checkpoint=args.custom_checkpoint,
                repo_dir=args.dinov3_repo,
                device=device,
                lora_rank=8,
            ),
        ),
    ]
    feature_sets = []
    for name, load_model in model_loaders:
        model = load_model()
        features = patch_features(model, image, grid, device)[0]
        feature_sets.append((name, features))
        LOGGER.info(
            "%s features for ADNI sample %d: %s",
            name,
            index,
            features.shape,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    maps = []
    for name, features in feature_sets:
        maps.append((name, pca_rgb(features, grid)))
    save_plot(image, maps, args.output)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
