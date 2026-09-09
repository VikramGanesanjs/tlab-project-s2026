"""Compare ADNI test AUROC across BrainDINO, DINOv3, and 3-D-aware fine tuning.

The script reads five cross-validation folds for each method, writes the
resulting tidy dataframe, and produces one boxplot for each ADNI task.

Example
-------
python -m utils.plot_adni_classification_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFICATION_ROOT = REPO_ROOT / "classification" / "adni"
DEFAULT_SPATIAL_WINDOW_ROOT = REPO_ROOT / "runs" / "loss_fn_comparison" / "spatial_window"
TASKS = ("cn_ad", "cn_mci")
METHODS = ("DINOv3", "BrainDINO", "3-D Aware Fine Tuning")
PAIRS = (("BrainDINO", "DINOv3"), ("3-D Aware Fine Tuning", "DINOv3"),
         ("3-D Aware Fine Tuning", "BrainDINO"))


def test_auroc(summary_path: Path) -> float:
    """Read the held-out AUROC from a classification run summary."""
    with summary_path.open() as handle:
        summary = json.load(handle)
    try:
        return float(summary["metrics"]["test"]["auroc"])
    except KeyError as exc:
        raise ValueError(f"No test AUROC in {summary_path}") from exc


def collect_results(classification_root: Path, spatial_window_root: Path) -> pd.DataFrame:
    """Create a tidy dataframe with one test AUROC per method, task, and fold."""
    records: list[dict[str, object]] = []
    for task in TASKS:
        for directory_name, method in (("braindino", "BrainDINO"), ("dinov3", "DINOv3")):
            for summary_path in sorted((classification_root / task / directory_name).glob("*/run_summary.json")):
                records.append(
                    {
                        "task": task,
                        "method": method,
                        "fold": int(summary_path.parent.name),
                        "checkpoint": pd.NA,
                        "test_auroc": test_auroc(summary_path),
                        "run_summary": str(summary_path),
                    }
                )

        for summary_path in sorted((spatial_window_root).glob(f"*/{task}/4999/run_summary.json")):
            records.append(
                {
                    "task": task,
                    "method": "3-D Aware Fine Tuning",
                    "fold": int(summary_path.parents[2].name),
                    "checkpoint": 4999,
                    "test_auroc": test_auroc(summary_path),
                    "run_summary": str(summary_path),
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    expected = {(task, method) for task in TASKS for method in METHODS}
    observed = set(dataframe[["task", "method"]].itertuples(index=False, name=None))
    if observed != expected or len(dataframe) != 30:
        raise ValueError(
            "Expected five folds for every task/method combination; "
            f"found {len(dataframe)} records across {sorted(observed)}."
        )
    return dataframe.sort_values(["task", "method", "fold"], kind="stable").reset_index(drop=True)


def add_mann_whitney_annotations(ax: plt.Axes, task_data: pd.DataFrame) -> None:
    """Test paired fold differences against zero and add significance stars."""
    minimum = task_data["test_auroc"].min()
    maximum = task_data["test_auroc"].max()
    spacing = max((maximum - minimum) * 0.08, 0.01)
    ax.set_ylim(top=maximum + spacing * (len(PAIRS) + 1))
    for level, (first_method, second_method) in enumerate(PAIRS):
        fold_scores = task_data.pivot(index="fold", columns="method", values="test_auroc")
        differences = fold_scores[first_method] - fold_scores[second_method]
        _, p_value = mannwhitneyu(
            differences,
            np.zeros(len(differences)),
            alternative="greater",
        )
        first_x, second_x = METHODS.index(first_method), METHODS.index(second_method)
        stars = "ns" if p_value > 0.05 else "*" if p_value > 0.01 else "**" if p_value > 0.001 else "***"
        y = maximum + spacing * (level + 1)
        ax.plot([first_x, first_x, second_x, second_x], [y, y + spacing / 4, y + spacing / 4, y],
                color="black")
        ax.text((first_x + second_x) / 2, y + spacing / 3, stars, ha="center", va="bottom")


def plot_task(dataframe: pd.DataFrame, task: str, output_path: Path) -> None:
    """Save a single task-specific AUROC comparison boxplot."""
    task_data = dataframe.loc[dataframe["task"] == task]
    fig, ax = plt.subplots()
    sns.boxplot(
        data=task_data,
        x="method",
        y="test_auroc",
        order=METHODS,
        hue="method",
        hue_order=METHODS,
        legend=False,
        ax=ax,
    )
    add_mann_whitney_annotations(ax, task_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--spatial-window-root", type=Path, default=DEFAULT_SPATIAL_WINDOW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    args = parser.parse_args()

    dataframe = collect_results(args.classification_root, args.spatial_window_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_dir / "adni_auroc_comparison.csv", index=False)
    for task in TASKS:
        plot_task(dataframe, task, args.output_dir / f"adni_{task}_auroc_comparison.png")


if __name__ == "__main__":
    main()
