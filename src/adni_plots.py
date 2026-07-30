#!/usr/bin/env python3
"""Plot ADNI training run metrics from run_summary.json files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "adni"
OUTPUT_DIR = RUNS_DIR / "plots"

METRICS = ("auroc", "f1")
MARGINAL_FACTOR_CONFIGS = (
    {"name": "slice_aggregator", "title": "Slice Aggregator", "stratify_by_encoder": True},
    {"name": "encoder", "title": "Encoder", "stratify_by_encoder": False},
    {"name": "features", "title": "Features", "stratify_by_encoder": True},
)
FACTOR_CONFIGS = (
    {
        "name": "slice_aggregator",
        "title": "Slice Aggregator",
        "x": "slice_aggregator",
        "hue": "encoder",
        "col": "features",
    },
    {
        "name": "encoder",
        "title": "Encoder",
        "x": "encoder",
        "hue": "slice_aggregator",
        "col": "features",
    },
    {
        "name": "features",
        "title": "Features",
        "x": "features",
        "hue": "encoder",
        "col": "slice_aggregator",
    },
)

SLICE_AGGREGATOR_LABELS = {
    "mean": "mean pooling",
    "transformer": "transformer",
}

FACET_LABELS = {
    "encoder": "Encoder",
    "features": "Features",
    "slice_aggregator": "Slice agg.",
}


def add_shared_legend(fig: plt.Figure, axes: Iterable[plt.Axes], title: str) -> None:
    """Move subplot legends into one figure-level legend outside the plot grid."""
    handles: list = []
    labels: list[str] = []

    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if handles:
        fig.legend(
            handles,
            labels,
            title=title,
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            frameon=False,
        )


def format_facet_title(name: str, value: str) -> str:
    """Format facet labels compactly enough to fit above each subplot."""
    return f"{FACET_LABELS.get(name, name)}: {value}"


def load_run_summaries(runs_dir: Path) -> pd.DataFrame:
    """Load test metrics and hyperparameters from each run_summary.json."""
    rows: list[dict] = []

    for summary_path in sorted(runs_dir.glob("*/run_summary.json")):
        with summary_path.open() as f:
            data = json.load(f)

        params = data["parameters"]
        test_metrics = data["metrics"]["test"]

        rows.append(
            {
                "run_name": params.get("run_name", summary_path.parent.name),
                "encoder": params["encoder"],
                "slice_aggregator": params["slice_aggregator"],
                "features": params["features"],
                "auroc": test_metrics["auroc"],
                "f1": test_metrics["f1"],
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No run_summary.json files found under {runs_dir}/<run_name>/"
        )

    df = pd.DataFrame(rows)
    df["slice_aggregator"] = df["slice_aggregator"].map(
        lambda value: SLICE_AGGREGATOR_LABELS.get(value, value)
    )
    return df


def plot_factor_effect(
    df: pd.DataFrame,
    factor: dict,
    output_dir: Path,
) -> None:
    """Create AUROC and F1 bar plots for one experimental factor."""
    col_values = sorted(df[factor["col"]].unique())
    fig, axes = plt.subplots(
        len(METRICS),
        len(col_values),
        figsize=(5.25 * len(col_values), 4.75 * len(METRICS)),
        squeeze=False,
    )
    fig.suptitle(f"Test Performance by {factor['title']}", y=0.98)

    for row_idx, metric in enumerate(METRICS):
        for col_idx, col_value in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            subset = df[df[factor["col"]] == col_value]

            sns.barplot(
                data=subset,
                x=factor["x"],
                y=metric,
                hue=factor["hue"],
                ax=ax,
                errorbar=None,
                palette="muted",
            )

            ax.set_ylim(0, 1)
            ax.set_xlabel(factor["title"] if row_idx == len(METRICS) - 1 else "")
            ax.set_ylabel(metric.upper())
            ax.set_title(format_facet_title(factor["col"], col_value), pad=12)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            ax.tick_params(axis="x", rotation=15)

    add_shared_legend(fig, axes.ravel(), factor["hue"].replace("_", " ").title())

    output_path = output_dir / f"adni_{factor['name']}_effect.png"
    fig.tight_layout(rect=(0, 0, 0.86, 0.95), pad=1.4, w_pad=2.2, h_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def compute_marginal_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average test metrics, stratifying by encoder except for encoder itself."""
    rows: list[dict] = []

    for factor in MARGINAL_FACTOR_CONFIGS:
        factor_name = factor["name"]
        if factor["stratify_by_encoder"]:
            grouped = (
                df.groupby([factor_name, "encoder"], as_index=False)[list(METRICS)]
                .mean()
                .rename(columns={factor_name: "choice"})
            )
        else:
            grouped = (
                df.groupby(factor_name, as_index=False)[list(METRICS)]
                .mean()
                .rename(columns={factor_name: "choice"})
            )
            grouped["encoder"] = None

        grouped["factor"] = factor["title"]
        grouped["stratify_by_encoder"] = factor["stratify_by_encoder"]
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def plot_marginal_means(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean test performance averaged over the other hyperparameters."""
    marginal_df = compute_marginal_means(df)
    id_vars = ["factor", "choice", "stratify_by_encoder"]
    if marginal_df["encoder"].notna().any():
        id_vars.append("encoder")

    metric_df = marginal_df.melt(
        id_vars=id_vars,
        value_vars=list(METRICS),
        var_name="metric",
        value_name="score",
    )

    fig, axes = plt.subplots(
        len(METRICS),
        len(MARGINAL_FACTOR_CONFIGS),
        figsize=(6 * len(MARGINAL_FACTOR_CONFIGS), 4.75 * len(METRICS)),
        squeeze=False,
    )
    fig.suptitle(
        "Mean Test Performance by Hyperparameter "
        "(slice aggregator and features stratified by encoder)",
        y=0.98,
    )

    for row_idx, metric in enumerate(METRICS):
        for col_idx, factor in enumerate(MARGINAL_FACTOR_CONFIGS):
            ax = axes[row_idx, col_idx]
            factor_subset = metric_df[
                (metric_df["metric"] == metric)
                & (metric_df["factor"] == factor["title"])
            ]

            if factor["stratify_by_encoder"]:
                sns.barplot(
                    data=factor_subset,
                    x="choice",
                    y="score",
                    hue="encoder",
                    ax=ax,
                    errorbar=None,
                    palette="muted",
                )
            else:
                sns.barplot(
                    data=factor_subset,
                    x="choice",
                    y="score",
                    hue="choice",
                    ax=ax,
                    errorbar=None,
                    palette="muted",
                    legend=False,
                )

            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel(metric.upper())
            ax.set_title(factor["title"])
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.tick_params(axis="x", rotation=15)

    add_shared_legend(fig, axes.ravel(), "Encoder")

    output_path = output_dir / "adni_marginal_means.png"
    fig.tight_layout(rect=(0, 0, 0.86, 0.95), pad=1.4, w_pad=2.2, h_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing ADNI run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where plot images are saved",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")

    df = load_run_summaries(args.runs_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} runs from {args.runs_dir}")
    print(df.sort_values(["encoder", "slice_aggregator", "features"]).to_string(index=False))

    for factor in FACTOR_CONFIGS:
        plot_factor_effect(df, factor, args.output_dir)

    plot_marginal_means(df, args.output_dir)


if __name__ == "__main__":
    main()
