"""Compare CQ500 ICH test AUROC across DINOv3 adaptation methods.

The script reads the five cross-validation folds for DINOv3, MedDINOv3, and
SSL fine-tuning; writes a tidy dataframe; and saves a hue-colored boxplot with
paired, directional Mann--Whitney U significance annotations.

Example
-------
python -m utils.plot_cq500_ich_comparison
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
DEFAULT_RESULTS_ROOT = REPO_ROOT / "classification" / "cq500" / "ich"
METHODS = ("DINOv3", "MedDINOv3", "SSL Fine Tune")
METHOD_DIRECTORIES = (("dinov3", "DINOv3"), ("meddinov3", "MedDINOv3"),
                      ("ssl_finetune", "SSL Fine Tune"))
PAIRS = (("MedDINOv3", "DINOv3"), ("SSL Fine Tune", "DINOv3"),
         ("SSL Fine Tune", "MedDINOv3"))


def test_auroc(summary_path: Path) -> float:
    """Read the held-out AUROC from a run summary."""
    with summary_path.open() as handle:
        summary = json.load(handle)
    try:
        return float(summary["metrics"]["test"]["auroc"])
    except KeyError as exc:
        raise ValueError(f"No test AUROC in {summary_path}") from exc


def collect_results(results_root: Path) -> pd.DataFrame:
    """Create a tidy dataframe with one test AUROC per method and fold."""
    records: list[dict[str, object]] = []
    for directory_name, method in METHOD_DIRECTORIES:
        for summary_path in sorted((results_root / directory_name).glob("*/run_summary.json")):
            records.append(
                {
                    "method": method,
                    "fold": int(summary_path.parent.name),
                    "test_auroc": test_auroc(summary_path),
                    "run_summary": str(summary_path),
                }
            )

    dataframe = pd.DataFrame.from_records(records)
    expected = {(method, fold) for method in METHODS for fold in range(5)}
    observed = set(dataframe[["method", "fold"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            "Expected exactly five folds (0--4) for every method; "
            f"found {len(dataframe)} records."
        )
    return dataframe.sort_values(["method", "fold"], kind="stable").reset_index(drop=True)


def significance_stars(p_value: float) -> str:
    """Format a p-value as standard statistical significance stars."""
    if p_value > 0.05:
        return "ns"
    if p_value > 0.01:
        return "*"
    if p_value > 0.001:
        return "**"
    return "***"


def add_mann_whitney_annotations(ax: plt.Axes, dataframe: pd.DataFrame) -> None:
    """Test paired fold differences against zero and add stars inside the axes."""
    fold_scores = dataframe.pivot(index="fold", columns="method", values="test_auroc")
    minimum, maximum = dataframe["test_auroc"].agg(["min", "max"])
    spacing = max((maximum - minimum) * 0.08, 0.01)
    ax.set_ylim(top=maximum + spacing * (len(PAIRS) + 1))
    for level, (first_method, second_method) in enumerate(PAIRS):
        differences = fold_scores[first_method] - fold_scores[second_method]
        _, p_value = mannwhitneyu(
            differences,
            np.zeros(len(differences)),
            alternative="greater",
        )
        first_x, second_x = METHODS.index(first_method), METHODS.index(second_method)
        y = maximum + spacing * (level + 1)
        ax.plot(
            [first_x, first_x, second_x, second_x],
            [y, y + spacing / 4, y + spacing / 4, y],
            color="black",
        )
        ax.text(
            (first_x + second_x) / 2,
            y + spacing / 3,
            significance_stars(p_value),
            ha="center",
            va="bottom",
        )


def plot(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the CQ500 ICH AUROC comparison boxplot."""
    fig, ax = plt.subplots()
    sns.boxplot(
        data=dataframe,
        x="method",
        y="test_auroc",
        order=METHODS,
        hue="method",
        hue_order=METHODS,
        legend=False,
        ax=ax,
    )
    add_mann_whitney_annotations(ax, dataframe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-stem", default="cq500_ich_auroc_comparison")
    args = parser.parse_args()

    dataframe = collect_results(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_dir / f"{args.output_stem}.csv", index=False)
    plot(dataframe, args.output_dir / f"{args.output_stem}.png")


if __name__ == "__main__":
    main()
