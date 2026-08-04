"""PCA visualization of three DINO backbones on one sampled ADNI slice."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dino_mst import _load_custom_dinov3_encoder  # noqa: E402
from dinov3_baseline import load_braindino_encoder, load_dinov3_encoder  # noqa: E402
from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    DEFAULT_ROOT,
    DEFAULT_ADNI_TASK,
    build_adni_transform,
)

LOGGER = logging.getLogger("pca_dino_backbones")
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


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
    output = model.forward_features(image.to(device))
    features = output["x_norm_patchtokens"]
    expected = grid[0] * grid[1]
    if features.shape[1] != expected:
        raise ValueError(f"Backbone returned {features.shape[1]} patches, expected {expected}")
    return features[0].float().cpu().numpy()


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sampling")
    parser.add_argument("--dinov3-checkpoint", type=Path, required=True)
    parser.add_argument("--braindino-checkpoint", type=Path, required=True)
    parser.add_argument("--custom-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--dinov3-model", default="dinov3_vitb16")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("pca_adni_slice.png"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by 16")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image, index = sample_adni_slice(args)
    grid = (args.image_size // 16, args.image_size // 16)

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
            "Custom distributed checkpoint",
            lambda: _load_custom_dinov3_encoder(
                checkpoint=args.custom_checkpoint,
                repo_dir=args.dinov3_repo,
                device=device,
                encoder_training="frozen",
                lora_rank=8,
            ),
        ),
    ]
    feature_sets = []
    for name, load_model in model_loaders:
        model = load_model()
        features = patch_features(model, image, grid, device)
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
