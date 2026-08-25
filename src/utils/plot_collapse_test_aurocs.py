#!/usr/bin/env python3
"""Plot validation and test AUROC across collapse-test checkpoints.

Each checkpoint directory is expected to contain three subdirectories named
``run0`` through ``run2``. Each run must contain a metrics summary JSON file;
the filename is configurable for either training (``run_summary.json``) or
last-checkpoint evaluation (``last_metrics_summary.json``) results. Error bars are the sample standard deviation
across the runs at a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_aurocs(
    runs_dir: Path, summary_filename: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return iterations and mean/std AUROCs, validating three runs per iteration."""
    records: dict[int, dict[str, list[float]]] = {}
    for summary_path in runs_dir.glob(f"*/run*/{summary_filename}"):
        iteration = int(summary_path.parent.parent.name)
        with summary_path.open() as handle:
            summary = json.load(handle)
        metrics = summary["metrics"]
        record = records.setdefault(iteration, {"val": [], "test": []})
        for split in ("val", "test"):
            record[split].append(float(metrics[split]["auroc"]))

    if not records:
        raise FileNotFoundError(f"No {summary_filename} files found under {runs_dir}")

    iterations = np.array(sorted(records))
    for iteration in iterations:
        counts = {split: len(records[int(iteration)][split]) for split in ("val", "test")}
        if counts != {"val": 3, "test": 3}:
            raise ValueError(f"Expected three runs at iteration {iteration}, found {counts}")

    val = np.array([records[int(iteration)]["val"] for iteration in iterations])
    test = np.array([records[int(iteration)]["test"] for iteration in iterations])
    return iterations, val.mean(axis=1), val.std(axis=1, ddof=1), test.mean(axis=1), test.std(axis=1, ddof=1)


def plot(runs_dir: Path, output_path: Path, summary_filename: str) -> None:
    iterations, val_mean, val_std, test_mean, test_std = load_aurocs(runs_dir, summary_filename)

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.errorbar(
        iterations,
        val_mean,
        yerr=val_std,
        marker="o",
        capsize=4,
        linewidth=2,
        label="Validation AUROC",
    )
    axis.errorbar(
        iterations,
        test_mean,
        yerr=test_std,
        marker="s",
        capsize=4,
        linewidth=2,
        label="Test AUROC",
    )
    axis.set(
        xlabel="Iteration",
        ylabel="AUROC",
        title="Validation and Test AUROC by Checkpoint",
        xticks=iterations,
        ylim=(0.5, 1.0),
    )
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs_dir",
        nargs="?",
        type=Path,
        default=Path("runs/collapse_test"),
        help="Directory containing checkpoint/run*/<summary-file> files.",
    )
    parser.add_argument(
        "--summary-file",
        default="run_summary.json",
        help="Metrics summary filename in each run directory (default: run_summary.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <runs_dir>/val_test_auroc_by_iteration.png).",
    )
    args = parser.parse_args()
    output = args.output or args.runs_dir / "val_test_auroc_by_iteration.png"
    plot(args.runs_dir, output, args.summary_file)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
