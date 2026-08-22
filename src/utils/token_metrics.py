"""Token-feature metrics and Gram-matrix comparisons across DINOv3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from utils.load_dinov3 import (  # noqa: E402
    DEFAULT_BRAINDINO_WEIGHTS,
    DEFAULT_DINOV3_WEIGHTS,
    DINOV3_REPO,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_braindino_encoder,
    load_dinov3_encoder,
)
from datasets.adni import (  # noqa: E402
    ADNIClassificationDataset,
    DEFAULT_ADNI_TASK,
    DEFAULT_ROOT as ADNI_DEFAULT_ROOT,
    build_adni_transform,
)
from datasets.duke import (  # noqa: E402
    DukeClassificationDataset,
    build_duke_transform,
)

LOGGER = logging.getLogger("token_metrics")
DUKE_DEFAULT_ROOT = SRC_DIR.parent / "data" / "tcia" / "duke_breast_cancer_processed"


def _build_dataset(args):
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


def _sample_images(dataset, n_images: int, seed: int | None):
    if n_images <= 0:
        raise ValueError("n_images must be positive")
    n_images = min(n_images, len(dataset))
    rng = np.random.default_rng(seed)
    by_patient = {}
    for index in range(len(dataset)):
        by_patient.setdefault(dataset.get_patient_id(index), []).append(index)
    patients = list(by_patient)
    rng.shuffle(patients)
    indices = [int(rng.choice(by_patient[patient])) for patient in patients[:n_images]]
    if len(indices) < n_images:
        remaining = [index for index in range(len(dataset)) if index not in indices]
        rng.shuffle(remaining)
        indices.extend(remaining[: n_images - len(indices)])
    images = torch.stack([dataset[index][0] for index in indices])
    LOGGER.info("Sampled %d images from %d patients", len(images), len(set(
        dataset.get_patient_id(index) for index in indices
    )))
    return images


def _discover_checkpoints(parent: Path):
    if not parent.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {parent}")
    checkpoints = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        if (child / ".metadata").is_file():
            checkpoints.append(child)
    if not checkpoints:
        raise ValueError(f"No distributed checkpoints found under {parent}")
    return sorted(
        checkpoints,
        key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name),
    )


def _select_checkpoint_stride(checkpoints: list[Path], stride: int) -> list[Path]:
    """Keep checkpoints whose index in discovery order is divisible by stride."""
    if stride <= 0:
        raise ValueError("--checkpoint-stride must be positive")
    return checkpoints[::stride]


def _load_custom(checkpoint: Path, args, device: torch.device):
    return load_custom_dinov3_encoder(
        checkpoint=checkpoint,
        repo_dir=args.dinov3_repo,
        device=device,
        encoder_training="frozen",
        lora_rank=args.lora_rank,
    )


def _load_reference(encoder: str, args, device: torch.device):
    """Load the fixed DINOv3 or BrainDINO reference encoder."""
    if encoder == "dinov3":
        return load_dinov3_encoder(
            repo_dir=args.dinov3_repo,
            weights=DEFAULT_DINOV3_WEIGHTS,
            model_name="dinov3_vitb16",
            device=device,
        )
    if encoder == "braindino":
        return load_braindino_encoder(
            repo_dir=args.dinov3_repo,
            weights=DEFAULT_BRAINDINO_WEIGHTS,
            device=device,
        )
    raise ValueError(f"Unknown reference encoder: {encoder!r}")


@torch.inference_mode()
def _features(model, images: torch.Tensor, device: torch.device, batch_size: int):
    cls_tokens, patch_tokens = [], []
    for start in range(0, len(images), batch_size):
        output = model.forward_features(images[start : start + batch_size].to(device))
        cls_tokens.append(output["x_norm_clstoken"].float().cpu())
        patch_tokens.append(output["x_norm_patchtokens"].float().cpu())
    return torch.cat(cls_tokens), torch.cat(patch_tokens)


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank after row-wise L2 normalization of feature vectors."""
    features = F.normalize(features, p=2, dim=-1)
    singular_values = torch.linalg.svdvals(features)
    total = singular_values.sum()
    if total <= torch.finfo(singular_values.dtype).eps:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _cls_effective_rank_summary(cls_tokens: torch.Tensor, n_bootstrap: int = 100):
    """Return CLS effective rank and bootstrap standard deviation over images."""
    rank = effective_rank(cls_tokens)
    if len(cls_tokens) < 2:
        return rank, 0.0
    generator = torch.Generator().manual_seed(0)
    bootstrap_ranks = []
    for _ in range(n_bootstrap):
        indices = torch.randint(
            len(cls_tokens), (len(cls_tokens),), generator=generator
        )
        bootstrap_ranks.append(effective_rank(cls_tokens[indices]))
    return rank, float(torch.tensor(bootstrap_ranks).std(unbiased=False))


def _gram_matrices(patches: torch.Tensor) -> torch.Tensor:
    """Compute cosine-similarity Gram matrices over patch tokens."""
    patches = F.normalize(patches, p=2, dim=-1)
    return torch.bmm(patches, patches.transpose(1, 2))


def _foreground_patch_mask(
    images: torch.Tensor,
    *,
    patch_size: int = 16,
    background_threshold: float = 8 / 255,
    min_foreground_fraction: float = 0.10,
) -> torch.Tensor:
    """Return a ``[batch, n_patches]`` mask that keeps non-background patches.

    ADNI slices are windowed to 8-bit intensity before ImageNet normalization,
    and their background is zero.  The red channel is sufficient because the
    grayscale image is replicated into all three channels.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images shaped [B, C, H, W], got {tuple(images.shape)}")
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError(f"Image dimensions must be divisible by patch_size={patch_size}")
    if not 0 <= min_foreground_fraction <= 1:
        raise ValueError("min_foreground_fraction must be in [0, 1]")

    intensities = (
        images[:, :1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    ).clamp(0, 1)
    foreground = intensities > background_threshold
    coverage = F.avg_pool2d(foreground.float(), kernel_size=patch_size, stride=patch_size)
    return coverage.flatten(1) >= min_foreground_fraction


def _validate_patch_mask(grams: torch.Tensor, patch_mask: torch.Tensor | None) -> None:
    if patch_mask is not None and patch_mask.shape != grams.shape[:2]:
        raise ValueError(
            "patch_mask must match the batch and patch dimensions of grams: "
            f"expected {tuple(grams.shape[:2])}, got {tuple(patch_mask.shape)}"
        )


def _patch_grid_pair_geometry(n_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return squared distances and the unique off-diagonal pair mask."""
    grid_size = int(n_patches**0.5)
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Expected a square patch grid, but received {n_patches} patch tokens"
        )
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)
    squared_distances = (coordinates[:, None] - coordinates[None, :]).square().sum(-1)
    upper_triangle = torch.triu(
        torch.ones((n_patches, n_patches), dtype=torch.bool), diagonal=1
    )
    return squared_distances, upper_triangle


def _mean_and_std(values: torch.Tensor) -> tuple[float, float]:
    return float(values.mean()), float(values.std(unbiased=False))


def _patch_effective_rank_summary(
    patches: torch.Tensor, patch_mask: torch.Tensor | None = None
) -> tuple[float, float]:
    """Summarize per-image patch-token effective rank."""
    if patch_mask is not None and patch_mask.shape != patches.shape[:2]:
        raise ValueError(
            "patch_mask must match the batch and patch dimensions of patches: "
            f"expected {tuple(patches.shape[:2])}, got {tuple(patch_mask.shape)}"
        )
    ranks = torch.tensor(
        [
            effective_rank(tokens if patch_mask is None else tokens[patch_mask[index]])
            for index, tokens in enumerate(patches)
        ]
    )
    return _mean_and_std(ranks)


def _mean_off_diagonal_similarity(
    grams: torch.Tensor, patch_mask: torch.Tensor | None = None
) -> tuple[float, float]:
    """Summarize the mean off-diagonal patch similarity per image."""
    _validate_patch_mask(grams, patch_mask)
    n_patches = grams.shape[-1]
    mask = ~torch.eye(n_patches, dtype=torch.bool, device=grams.device)
    values = []
    for index, gram in enumerate(grams):
        valid = mask if patch_mask is None else mask & torch.outer(patch_mask[index], patch_mask[index])
        values.append(gram[valid].mean() if valid.any() else torch.tensor(0.0))
    return _mean_and_std(torch.stack(values))


def _spatial_specificity(
    grams: torch.Tensor, patch_mask: torch.Tensor | None = None
) -> list[tuple[float, float, float]]:
    """Return mean patch similarity by Euclidean patch-grid distance.

    Only one copy of each off-diagonal Gram-matrix pair is used.  For each
    distance, the mean similarity is computed separately for every sampled
    image, and the returned uncertainty is their standard deviation.
    """
    _validate_patch_mask(grams, patch_mask)
    n_patches = grams.shape[-1]
    squared_distances, upper_triangle = _patch_grid_pair_geometry(n_patches)

    summaries = []
    for squared_distance in torch.unique(squared_distances[upper_triangle], sorted=True):
        distance_mask = upper_triangle & (squared_distances == squared_distance)
        image_means = []
        for index, gram in enumerate(grams):
            valid = distance_mask if patch_mask is None else (
                distance_mask & torch.outer(patch_mask[index], patch_mask[index])
            )
            if valid.any():
                image_means.append(gram[valid].mean())
        if image_means:
            mean, std = _mean_and_std(torch.stack(image_means))
            summaries.append((float(squared_distance.float().sqrt()), mean, std))
    return summaries


def _spatial_specificity_correlation(
    grams: torch.Tensor, patch_mask: torch.Tensor | None = None
) -> tuple[float, float]:
    """Summarize Pearson r of patch-pair distance versus similarity per image."""
    _validate_patch_mask(grams, patch_mask)
    squared_distances, upper_triangle = _patch_grid_pair_geometry(grams.shape[-1])
    distances = squared_distances.float().sqrt()
    correlations = []
    for index, gram in enumerate(grams):
        valid = upper_triangle if patch_mask is None else (
            upper_triangle & torch.outer(patch_mask[index], patch_mask[index])
        )
        pair_distances = distances[valid]
        similarities = gram[valid]
        if len(pair_distances) < 2 or pair_distances.std(unbiased=False) == 0:
            continue
        correlation = torch.corrcoef(torch.stack((pair_distances, similarities)))[0, 1]
        if torch.isfinite(correlation):
            correlations.append(correlation)
    if not correlations:
        return 0.0, 0.0
    return _mean_and_std(torch.stack(correlations))


def _masked_gram_distances(
    grams: torch.Tensor,
    reference_grams: torch.Tensor,
    patch_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-image Gram distances after excluding background patch pairs."""
    if grams.shape != reference_grams.shape:
        raise ValueError("grams and reference_grams must have identical shapes")
    _validate_patch_mask(grams, patch_mask)
    distances = []
    for index, (gram, reference_gram) in enumerate(zip(grams, reference_grams)):
        keep = slice(None) if patch_mask is None else patch_mask[index]
        difference = gram[keep][:, keep] - reference_gram[keep][:, keep]
        distances.append(torch.linalg.vector_norm(difference))
    return torch.stack(distances)


def _checkpoint_iteration(name: str, fallback: int) -> int:
    """Use the final number in a checkpoint name as its training iteration."""
    numbers = re.findall(r"\d+", name)
    return int(numbers[-1]) if numbers else fallback


def _selected_checkpoints(checkpoint_parent: Path, stride: int) -> list[tuple[int, Path]]:
    """Return ``(discovery_index, checkpoint)`` pairs after stride selection."""
    discovered = _discover_checkpoints(checkpoint_parent)
    checkpoints = _select_checkpoint_stride(discovered, stride)
    selected = list(zip(range(0, len(discovered), stride), checkpoints))
    LOGGER.info("Selected %d checkpoint(s) with stride=%d", len(selected), stride)
    return selected


def _metric_patch_mask(images: torch.Tensor, mask_background: bool) -> torch.Tensor | None:
    """Build and report the optional mask shared by every patch-based metric."""
    if not mask_background:
        return None
    patch_mask = _foreground_patch_mask(images)
    LOGGER.info(
        "Keeping %.1f%% of patch tokens after background masking",
        100 * float(patch_mask.float().mean()),
    )
    return patch_mask


def metrics_evolution(checkpoint_parent: Path, *, args):
    """Compute token metrics for every selected checkpoint in one pass each."""
    dataset = _build_dataset(args)
    images = _sample_images(dataset, args.n_images, args.seed)
    checkpoints = _selected_checkpoints(checkpoint_parent, args.checkpoint_stride)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    patch_mask = _metric_patch_mask(images, args.mask_background)

    results = []
    for index, checkpoint in checkpoints:
        LOGGER.info("Loading %s (%d/%d)", checkpoint.name, len(results) + 1, len(checkpoints))
        model = _load_custom(checkpoint, args, device)
        cls_tokens, patches = _features(model, images, device, args.batch_size)
        grams = _gram_matrices(patches)
        cls_rank, cls_rank_error = _cls_effective_rank_summary(cls_tokens)
        patch_rank, patch_rank_error = _patch_effective_rank_summary(patches, patch_mask)
        off_diagonal, off_diagonal_error = _mean_off_diagonal_similarity(grams, patch_mask)
        spatial_correlation, spatial_correlation_error = _spatial_specificity_correlation(
            grams, patch_mask
        )
        results.append(
            {
                "checkpoint": checkpoint.name,
                "index": index,
                "iteration": _checkpoint_iteration(checkpoint.name, index),
                "cls_effective_rank": cls_rank,
                "cls_effective_rank_error": cls_rank_error,
                "patch_effective_rank": patch_rank,
                "patch_effective_rank_error": patch_rank_error,
                "mean_off_diagonal_similarity": off_diagonal,
                "mean_off_diagonal_similarity_error": off_diagonal_error,
                "spatial_specificity_correlation": spatial_correlation,
                "spatial_specificity_correlation_error": spatial_correlation_error,
                "spatial_specificity": _spatial_specificity(grams, patch_mask),
            }
        )
        print(
            f"{checkpoint.name}\tcls_effective_rank={cls_rank:.6f}±{cls_rank_error:.6f}"
            f"\tpatch_effective_rank={patch_rank:.6f}±{patch_rank_error:.6f}"
            f"\tmean_off_diagonal_similarity={off_diagonal:.6f}±{off_diagonal_error:.6f}"
            f"\tspatial_specificity_r={spatial_correlation:.6f}±{spatial_correlation_error:.6f}"
        )
        del cls_tokens, patches, grams, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def comparison_evolution(checkpoint_parent: Path, *, args):
    """Measure patch-Gram distance from each checkpoint to a reference encoder."""
    dataset = _build_dataset(args)
    images = _sample_images(dataset, args.n_images, args.seed)
    checkpoints = _selected_checkpoints(checkpoint_parent, args.checkpoint_stride)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    patch_mask = _metric_patch_mask(images, args.mask_background)
    reference = _load_reference(args.reference_encoder, args, device)
    _, reference_patches = _features(reference, images, device, args.batch_size)
    reference_grams = _gram_matrices(reference_patches)
    del reference_patches, reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for position, (index, checkpoint) in enumerate(checkpoints, start=1):
        LOGGER.info("Loading %s (%d/%d)", checkpoint.name, position, len(checkpoints))
        model = _load_custom(checkpoint, args, device)
        _, patches = _features(model, images, device, args.batch_size)
        grams = _gram_matrices(patches)
        distances = _masked_gram_distances(grams, reference_grams, patch_mask)
        mean, std = _mean_and_std(distances)
        results.append(
            (checkpoint.name, index, _checkpoint_iteration(checkpoint.name, index), mean, std)
        )
        print(f"{checkpoint.name}\tmean_gram_distance={mean:.6f}±{std:.6f}")
        del patches, grams, distances, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _write_metrics_csv(path: Path | None, results) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "checkpoint",
                "index",
                "iteration",
                "metric",
                "spatial_distance",
                "value",
                "error",
            )
        )
        for result in results:
            for metric in (
                "cls_effective_rank",
                "patch_effective_rank",
                "mean_off_diagonal_similarity",
                "spatial_specificity_correlation",
            ):
                writer.writerow(
                    (
                        result["checkpoint"],
                        result["index"],
                        result["iteration"],
                        metric,
                        "",
                        result[metric],
                        result[f"{metric}_error"],
                    )
                )
            for distance, value, error in result["spatial_specificity"]:
                writer.writerow(
                    (
                        result["checkpoint"],
                        result["index"],
                        result["iteration"],
                        "spatial_specificity",
                        distance,
                        value,
                        error,
                    )
                )
    LOGGER.info("Saved results to %s", path)


def _write_metrics_plot(path: Path | None, results) -> None:
    if path is None:
        return
    figure, axes = plt.subplots(3, 2, figsize=(14, 14), layout="constrained")
    checkpoints = [result["checkpoint"] for result in results]
    positions = [result["iteration"] for result in results]
    for axis, metric, title, ylabel in (
        (axes[0, 0], "cls_effective_rank", "CLS effective rank", "Effective rank"),
        (axes[0, 1], "patch_effective_rank", "Patch effective rank", "Effective rank"),
        (
            axes[1, 0],
            "mean_off_diagonal_similarity",
            "Mean off-diagonal Gram value",
            "Cosine similarity",
        ),
    ):
        axis.errorbar(
            positions,
            [result[metric] for result in results],
            yerr=[result[f"{metric}_error"] for result in results],
            fmt="o-",
            capsize=3,
        )
        axis.set_title(title)
        axis.set_xlabel("Training iteration")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, checkpoints, rotation=35, ha="right")
        axis.grid(True, alpha=0.25)

    spatial_axis = axes[1, 1]
    for result in results:
        distances, values, errors = zip(*result["spatial_specificity"])
        spatial_axis.errorbar(
            distances,
            values,
            yerr=errors,
            fmt="o-",
            capsize=2,
            label=result["checkpoint"],
        )
    spatial_axis.set_title("Spatial specificity")
    spatial_axis.set_xlabel("Patch-grid distance")
    spatial_axis.set_ylabel("Patch-token similarity")
    spatial_axis.grid(True, alpha=0.25)
    spatial_axis.legend(title="Checkpoint", fontsize="small")

    correlation_axis = axes[2, 0]
    correlation_axis.errorbar(
        positions,
        [result["spatial_specificity_correlation"] for result in results],
        yerr=[result["spatial_specificity_correlation_error"] for result in results],
        fmt="o-",
        capsize=3,
    )
    correlation_axis.set_title("Spatial-specificity correlation evolution")
    correlation_axis.set_xlabel("Training iteration")
    correlation_axis.set_ylabel("Pearson r")
    correlation_axis.set_xticks(positions, checkpoints, rotation=35, ha="right")
    correlation_axis.grid(True, alpha=0.25)
    axes[2, 1].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved plot to %s", path)


def _write_comparison_csv(path: Path | None, results) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("checkpoint", "index", "iteration", "mean_gram_distance", "error"))
        writer.writerows(results)
    LOGGER.info("Saved results to %s", path)


def _write_comparison_plot(path: Path | None, results) -> None:
    if path is None:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = [iteration for _, _, iteration, _, _ in results]
    axis.errorbar(
        positions,
        [mean for _, _, _, mean, _ in results],
        yerr=[std for _, _, _, _, std in results],
        fmt="o-",
        capsize=3,
    )
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Patch-Gram distance to reference")
    axis.set_xticks(positions, [name for name, _, _, _, _ in results], rotation=35, ha="right")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved plot to %s", path)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-parent", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-stride",
        "--checkpoint-modulo",
        "--every-nth-checkpoint",
        dest="checkpoint_stride",
        type=int,
        default=1,
        help="Keep every Nth checkpoint in discovery order, starting with the first (default: 1).",
    )
    parser.add_argument("--dataset", choices=("adni", "duke"), default="adni")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
    parser.add_argument("--scan", default="pre")
    parser.add_argument("--include-bilateral", action="store_true")
    parser.add_argument("--z-min", type=float, default=0.25)
    parser.add_argument("--z-max", type=float, default=0.75)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
    parser.add_argument(
        "--mask-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude patches with less than 10%% non-background pixels from "
            "patch metrics (default: enabled; use --no-mask-background to disable)."
        ),
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    metrics_parser = subparsers.add_parser(
        "metrics", help="Compute CLS, patch, Gram, and spatial-specificity metrics."
    )
    _add_common_arguments(metrics_parser)
    metrics_parser.add_argument("--output", type=Path, default=Path("token_metrics.csv"))
    metrics_parser.add_argument("--plot", type=Path, default=Path("token_metrics.png"))

    comparison_parser = subparsers.add_parser(
        "comparison", help="Measure patch-Gram distance to a reference encoder."
    )
    _add_common_arguments(comparison_parser)
    comparison_parser.add_argument(
        "--reference-encoder",
        choices=("dinov3", "braindino"),
        default="dinov3",
        help="Reference encoder for patch-Gram distance (default: dinov3).",
    )
    comparison_parser.add_argument(
        "--output", type=Path, default=Path("token_metrics_comparison.csv")
    )
    comparison_parser.add_argument(
        "--plot", type=Path, default=Path("token_metrics_comparison.png")
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    if args.command == "metrics" and args.image_size < 32:
        raise ValueError("--image-size must be at least 32 for spatial specificity")
    if args.n_images <= 0 or args.batch_size <= 0:
        raise ValueError("--n-images and --batch-size must be positive")
    if args.checkpoint_stride <= 0:
        raise ValueError("--checkpoint-stride must be positive")
    if args.command == "metrics" and args.n_images < 2:
        raise ValueError("--n-images must be at least 2 for metrics")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "metrics":
        results = metrics_evolution(args.checkpoint_parent, args=args)
        _write_metrics_csv(args.output, results)
        _write_metrics_plot(args.plot, results)
    else:
        results = comparison_evolution(args.checkpoint_parent, args=args)
        _write_comparison_csv(args.output, results)
        _write_comparison_plot(args.plot, results)


if __name__ == "__main__":
    main()
