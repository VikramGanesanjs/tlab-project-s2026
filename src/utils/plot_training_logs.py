#!/usr/bin/env python3
"""Plot training metrics from DINOv3 logs or JSONL metric files.

The training logger writes records like::

    Training [  120/125000] ... total_loss: 16.2 (17.1) ... time: 0.7

The value before parentheses is the value for the current iteration and the
value in parentheses is the running average.  This script can plot either.

Examples
--------
python -m utils.plot_training_logs runs/continued_pretraining/adni/5353170_0_log.out

python -m utils.plot_training_logs \
    runs/continued_pretraining/adni-fixed/training_metrics.json \
    --output runs/training_curves.png --smooth 5 --no-show
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ITERATION_RE = re.compile(r"\[\s*(\d+)\s*/\s*\d+\]")
METRIC_RE = re.compile(
    rf"(?P<key>[A-Za-z][A-Za-z0-9_.-]*):\s*"
    rf"(?P<value>{NUMBER})(?:\s*\(\s*(?P<average>{NUMBER})\s*\))?"
)


@dataclass
class ParsedLog:
    """Metrics extracted from one log file."""

    path: Path
    iterations: list[int]
    values: list[dict[str, tuple[float, float | None]]]

    @property
    def label(self) -> str:
        """A short label suitable for a plot legend."""

        if self.path.stem == "training_metrics":
            return self.path.parent.name
        return self.path.stem.removesuffix("_log")


def parse_log(path: str | Path) -> ParsedLog:
    """Parse training records from *path*.

    Non-training lines and fields without numeric values are ignored.  Each
    metric is stored as ``(current_value, running_average)``; records without
    a parenthesized average use ``None`` for the second item.
    """

    path = Path(path)
    iterations: list[int] = []
    values: list[dict[str, tuple[float, float | None]]] = []

    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if "Training" not in line:
                continue

            iteration_match = ITERATION_RE.search(line)
            if iteration_match is None:
                continue

            metrics: dict[str, tuple[float, float | None]] = {}
            # Ignore the timestamp/logger prefix (for example
            # ``helpers.py:105``), which is not a training metric.
            training_text = line[line.index("Training") :]
            for match in METRIC_RE.finditer(training_text):
                current = float(match.group("value"))
                average_text = match.group("average")
                average = float(average_text) if average_text else None
                metrics[match.group("key")] = (current, average)

            if metrics:
                iterations.append(int(iteration_match.group(1)))
                values.append(metrics)

    if not iterations:
        raise ValueError(f"No training records found in {path}")

    return ParsedLog(path=path, iterations=iterations, values=values)


def parse_metrics_json(path: str | Path) -> ParsedLog:
    """Parse newline-delimited JSON records written by ``MetricLogger``."""

    path = Path(path)
    iterations: list[int] = []
    values: list[dict[str, tuple[float, float | None]]] = []

    with path.open(encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
            if not isinstance(record, dict) or "iteration" not in record:
                raise ValueError(f"Expected an object with an iteration on line {line_number} of {path}")

            metrics: dict[str, tuple[float, float | None]] = {}
            for key, value in record.items():
                if key == "iteration" or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                metrics[key] = (float(value), None)

            if metrics:
                iterations.append(int(record["iteration"]))
                values.append(metrics)

    if not iterations:
        raise ValueError(f"No metric records found in {path}")

    return ParsedLog(path=path, iterations=iterations, values=values)


def parse_input(path: str | Path) -> ParsedLog:
    """Parse a legacy text log or a newline-delimited JSON metrics file."""

    path = Path(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return parse_metrics_json(path)
    return parse_log(path)


def _metric_names(logs: Iterable[ParsedLog], kind: str) -> list[str]:
    """Return metric names for one of the plot panels."""

    names = {
        key
        for log in logs
        for row in log.values
        for key in row
        if _is_metric(key, kind)
    }

    # Keep common aggregate metrics first, then use a stable alphabetical
    # order for the dynamically discovered loss terms.
    preferred = {
        "loss": 0,
        "total_loss": 1,
        "lr": 0,
        "time": 0,
        "data": 1,
    }
    return sorted(names, key=lambda name: (preferred.get(name, 10), name))


def _is_metric(name: str, kind: str) -> bool:
    if kind == "loss":
        return "loss" in name and not name.endswith(("_weight", "_weights"))
    if kind == "lr":
        return name == "lr" or name.endswith("_lr")
    if kind == "grad":
        return name.endswith("grad_norm") or name.endswith("_grad_norm")
    if kind == "time":
        return name in {"time", "data"} or name.endswith("_time")
    raise ValueError(f"Unknown metric kind: {kind}")


def _series(
    log: ParsedLog, metric: str, value_kind: str, smooth: int, max_iterations: int | None
) -> tuple[list[int], list[float]]:
    points: list[tuple[int, float]] = []
    value_index = 0 if value_kind == "current" else 1

    for iteration, row in zip(log.iterations, log.values):
        if max_iterations is not None and iteration > max_iterations:
            continue
        value = row.get(metric)
        if value is None:
            continue
        selected = value[value_index]
        # ``time`` and ``data`` in these logs are instantaneous values and do
        # not have a parenthesized running average.  Keep them visible when
        # --value average is selected by falling back to the current value.
        if selected is None:
            selected = value[0]
        points.append((iteration, float(selected)))

    if smooth <= 1 or len(points) < 2:
        return [point[0] for point in points], [point[1] for point in points]

    # A centered moving average keeps the output length unchanged and avoids
    # requiring pandas just for smoothing.
    half_window = smooth // 2
    smoothed: list[float] = []
    for index in range(len(points)):
        start = max(0, index - half_window)
        end = min(len(points), index + half_window + 1)
        smoothed.append(sum(value for _, value in points[start:end]) / (end - start))
    return [point[0] for point in points], smoothed


def plot_logs(
    logs: list[ParsedLog],
    output: str | Path,
    value_kind: str = "current",
    smooth: int = 1,
    max_iterations: int | None = None,
    show: bool = True,
) -> None:
    """Create and save the training-metrics plot.

    Losses and gradient norms each get their own axis so large metrics do not
    compress smaller ones. Learning rates and timing metrics remain grouped.
    """

    if smooth < 1:
        raise ValueError("smooth must be at least 1")

    # Import plotting dependencies only when plotting, so parse_log remains
    # useful in lightweight scripts and tests without a display.
    import matplotlib.pyplot as plt

    loss_names = _metric_names(logs, "loss")
    grad_names = _metric_names(logs, "grad")
    plot_specs: list[tuple[str, str, list[str]]] = [
        (f"Loss: {name}", "loss", [name]) for name in loss_names
    ]
    plot_specs.append(("Learning rate", "lr", _metric_names(logs, "lr")))
    plot_specs.extend(
        (f"Gradient norm: {name}", "grad", [name]) for name in grad_names
    )
    plot_specs.append(("Timing", "time", _metric_names(logs, "time")))

    columns = 2
    rows = (len(plot_specs) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, max(6, 3.5 * rows)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    multiple_logs = len(logs) > 1

    for axis, (title, kind, names) in zip(axes_flat, plot_specs):
        for log in logs:
            for metric in names:
                x, y = _series(log, metric, value_kind, smooth, max_iterations)
                if not x:
                    continue
                separate_metric = len(names) == 1
                label = log.label if separate_metric and multiple_logs else metric
                if not separate_metric and multiple_logs:
                    label = f"{log.label}: {metric}"
                axis.plot(x, y, linewidth=1.2, label=label)
        axis.set_title(title)
        axis.set_ylabel(
            "Loss" if kind == "loss" else
            "Gradient norm" if kind == "grad" else
            "Learning rate" if kind == "lr" else
            "Seconds"
        )
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small", ncol=2)

    for axis in axes_flat[: len(plot_specs)]:
        axis.set_xlabel("Iteration")
    for axis in axes_flat[len(plot_specs) :]:
        axis.set_visible(False)
    value_label = "current value" if value_kind == "current" else "running average"
    fig.suptitle(f"Training metrics ({value_label})", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved {output} ({sum(len(log.iterations) for log in logs)} records)")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Training logs or JSONL metric files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("training_curves.png"),
        help="Output image path (default: training_curves.png)",
    )
    parser.add_argument(
        "--value",
        choices=("current", "average"),
        default="current",
        help="Plot per-iteration values or legacy log averages in parentheses (default: current)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        metavar="N",
        help="Centered moving-average window; 1 disables smoothing (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Only plot records through this global iteration",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logs = [parse_input(path) for path in args.logs]
    for log in logs:
        print(f"Parsed {log.path}: {len(log.iterations)} training records")
    plot_logs(
        logs,
        output=args.output,
        value_kind=args.value,
        smooth=args.smooth,
        max_iterations=args.max_iterations,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
