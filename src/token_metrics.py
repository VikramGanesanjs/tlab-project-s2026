"""Effective-rank and patch-Gram metrics across DINOv3 checkpoints."""

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

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from merge_dcp_lora import load_custom_dinov3_encoder  # noqa: E402
from dinov3_baseline import (  # noqa: E402
    DEFAULT_DINOV3_WEIGHTS,
    DINOV3_REPO,
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


def _load_custom(checkpoint: Path, args, device: torch.device):
    return load_custom_dinov3_encoder(
        checkpoint=checkpoint,
        repo_dir=args.dinov3_repo,
        device=device,
        encoder_training="frozen",
        lora_rank=args.lora_rank,
    )


def _load_reference(checkpoint: Path, args, device: torch.device):
    if checkpoint.is_dir():
        return _load_custom(checkpoint, args, device)
    return load_dinov3_encoder(
        repo_dir=args.dinov3_repo,
        weights=checkpoint,
        model_name=args.dinov3_model,
        device=device,
    )


@torch.inference_mode()
def _features(model, images: torch.Tensor, device: torch.device, batch_size: int):
    cls_tokens, patch_tokens = [], []
    for start in range(0, len(images), batch_size):
        output = model.forward_features(images[start : start + batch_size].to(device))
        cls_tokens.append(output["x_norm_clstoken"].float().cpu())
        patch_tokens.append(output["x_norm_patchtokens"].float().cpu())
    return torch.cat(cls_tokens), torch.cat(patch_tokens)


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank of a matrix whose rows are sampled-image features."""
    singular_values = torch.linalg.svdvals(features)
    total = singular_values.sum()
    if total <= torch.finfo(singular_values.dtype).eps:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def _effective_rank_summary(features: torch.Tensor, n_bootstrap: int = 100):
    """Return effective rank and bootstrap standard deviation."""
    rank = effective_rank(features)
    if len(features) < 2:
        return rank, 0.0
    generator = torch.Generator().manual_seed(0)
    bootstrap_ranks = []
    for _ in range(n_bootstrap):
        indices = torch.randint(
            len(features), (len(features),), generator=generator
        )
        bootstrap_ranks.append(effective_rank(features[indices]))
    return rank, float(torch.tensor(bootstrap_ranks).std(unbiased=False))


def _gram_matrices(patches: torch.Tensor) -> torch.Tensor:
    """Compute cosine-similarity Gram matrices over patch tokens."""
    patches = F.normalize(patches, p=2, dim=-1)
    return torch.bmm(patches, patches.transpose(1, 2))


def token_metrics_evolution(
    checkpoint_parent: Path,
    *,
    args,
):
    """Compute both metrics while making one forward pass per model batch."""
    dataset = _build_dataset(args)
    images = _sample_images(dataset, args.n_images, args.seed)
    checkpoints = _discover_checkpoints(checkpoint_parent)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    reference = _load_reference(args.reference_checkpoint, args, device)
    _, reference_patches = _features(reference, images, device, args.batch_size)
    reference_grams = _gram_matrices(reference_patches)
    del reference_patches, reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for checkpoint in checkpoints:
        LOGGER.info("Loading %s (%d/%d)", checkpoint.name, len(results) + 1, len(checkpoints))
        model = _load_custom(checkpoint, args, device)
        cls_tokens, patches = _features(model, images, device, args.batch_size)
        rank, rank_error = _effective_rank_summary(cls_tokens)
        grams = _gram_matrices(patches)
        distances = torch.linalg.vector_norm(grams - reference_grams, dim=(1, 2))
        distance = float(distances.mean())
        distance_error = float(distances.std(unbiased=False))
        results.append((checkpoint.name, rank, rank_error, distance, distance_error))
        print(
            f"{checkpoint.name}\teffective_rank={rank:.6f}±{rank_error:.6f}"
            f"\tmean_gram_distance={distance:.6f}±{distance_error:.6f}"
        )
        del cls_tokens, patches, grams, distances, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def eff_rank_evolution(
    checkpoint_parent: Path,
    *,
    args,
):
    """Report effective rank, sharing the combined metric implementation."""
    return [(name, rank, rank_error) for name, rank, rank_error, _, _ in token_metrics_evolution(
        checkpoint_parent, args=args
    )]


def cka_evolution(
    checkpoint_parent: Path,
    *,
    args,
):
    """Report Gram distance, sharing the combined metric implementation."""
    return [(name, distance, distance_error) for name, _, _, distance, distance_error in token_metrics_evolution(
        checkpoint_parent, args=args
    )]


def _write_results(path: Path | None, header, results):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(results)
    LOGGER.info("Saved results to %s", path)


def _write_plot(path: Path | None, results):
    if path is None:
        return
    iterations = []
    for index, (name, _, _, _, _) in enumerate(results):
        numbers = re.findall(r"\d+", name)
        iterations.append(int(numbers[-1]) if numbers else index)
    ranks = [rank for _, rank, _, _, _ in results]
    rank_errors = [error for _, _, error, _, _ in results]
    distances = [distance for _, _, _, distance, _ in results]
    distance_errors = [error for _, _, _, _, error in results]

    figure, rank_axis = plt.subplots(figsize=(8, 5))
    distance_axis = rank_axis.twinx()
    rank_line = rank_axis.errorbar(
        iterations,
        ranks,
        yerr=rank_errors,
        fmt="o-",
        capsize=3,
        color="tab:blue",
        label="Effective rank",
    )
    distance_line = distance_axis.errorbar(
        iterations,
        distances,
        yerr=distance_errors,
        fmt="s-",
        capsize=3,
        color="tab:orange",
        label="Mean Gram distance",
    )
    rank_axis.set_xlabel("Iteration")
    rank_axis.set_ylabel("Effective rank", color="tab:blue")
    distance_axis.set_ylabel("Mean normalized patch-Gram distance", color="tab:orange")
    rank_axis.tick_params(axis="y", labelcolor="tab:blue")
    distance_axis.tick_params(axis="y", labelcolor="tab:orange")
    rank_axis.grid(True, alpha=0.25)
    rank_axis.legend(
        [rank_line, distance_line],
        ["Effective rank", "Mean Gram distance"],
        loc="best",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved plot to %s", path)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("evolution", "Measure both metrics in one pass per checkpoint."),
        ("eff_rank_evolution", "Measure CLS-token effective rank."),
        ("cka_evolution", "Measure patch-Gram distance to a reference."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--checkpoint-parent", type=Path, required=True)
        subparser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_DINOV3_WEIGHTS)
        subparser.add_argument("--dataset", choices=("adni", "duke"), default="adni")
        subparser.add_argument("--data-root", type=Path, default=None)
        subparser.add_argument("--csv-path", type=Path, default=None)
        subparser.add_argument("--adni-task", default=DEFAULT_ADNI_TASK)
        subparser.add_argument("--scan", default="pre")
        subparser.add_argument("--include-bilateral", action="store_true")
        subparser.add_argument("--z-min", type=float, default=0.25)
        subparser.add_argument("--z-max", type=float, default=0.75)
        subparser.add_argument("--n-images", type=int, default=5)
        subparser.add_argument("--batch-size", type=int, default=4)
        subparser.add_argument("--image-size", type=int, default=224)
        subparser.add_argument("--seed", type=int, default=None)
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--lora-rank", type=int, default=8)
        subparser.add_argument("--dinov3-repo", type=Path, default=DINOV3_REPO)
        subparser.add_argument("--dinov3-model", default="dinov3_vitb16")
        subparser.add_argument("--output", type=Path, default=Path("token_metrics.csv"))
        subparser.add_argument(
            "--plot",
            type=Path,
            default=Path("token_metrics_evolution.png"),
        )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    if args.n_images <= 0 or args.batch_size <= 0:
        raise ValueError("--n-images and --batch-size must be positive")
    if args.command in ("evolution", "eff_rank_evolution") and args.n_images < 2:
        raise ValueError("--n-images must be at least 2 for effective rank")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    combined_results = token_metrics_evolution(args.checkpoint_parent, args=args)
    _write_plot(args.plot, combined_results)
    if args.command == "evolution":
        results = combined_results
        _write_results(
            args.output,
            (
                "checkpoint",
                "effective_rank",
                "effective_rank_error",
                "mean_gram_distance",
                "mean_gram_distance_error",
            ),
            results,
        )
    elif args.command == "eff_rank_evolution":
        results = [(name, rank, rank_error) for name, rank, rank_error, _, _ in combined_results]
        _write_results(args.output, ("checkpoint", "effective_rank", "effective_rank_error"), results)
    else:
        results = [(name, distance, distance_error) for name, _, _, distance, distance_error in combined_results]
        _write_results(args.output, ("checkpoint", "mean_gram_distance", "mean_gram_distance_error"), results)


if __name__ == "__main__":
    main()
