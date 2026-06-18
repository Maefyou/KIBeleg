from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# A grid is a pair of RunRecord attributes used as the heatmap rows (y) and columns (x).
GRID_SPECS = {
    "layers_heads": {
        "row_attr": "layers",
        "col_attr": "heads",
        "ylabel": "Number of layers",
        "xlabel": "Number of heads",
    },
    "points_dmodel": {
        "row_attr": "num_points",
        "col_attr": "d_model",
        "ylabel": "Number of sample points",
        "xlabel": "d_model (embedding dim)",
    },
}


@dataclass(frozen=True)
class RunRecord:
    layers: int
    heads: int
    d_model: int
    num_points: int
    trained_epochs: int
    value: float
    run_dir: str


def _load_metric(metrics_path: Path, metric_name: str) -> float:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    if metric_name == "final_epoch_loss":
        if "final_epoch_loss" not in metrics:
            raise KeyError(f"Missing 'final_epoch_loss' in {metrics_path}")
        return float(metrics["final_epoch_loss"])

    if metric_name == "last_window_loss_avg":
        summaries = metrics.get("training_summaries", [])
        if not summaries:
            raise KeyError(f"Missing 'training_summaries' in {metrics_path}")
        return float(summaries[-1]["loss_avg"])

    if metric_name == "last_window_loss_min":
        summaries = metrics.get("training_summaries", [])
        if not summaries:
            raise KeyError(f"Missing 'training_summaries' in {metrics_path}")
        if "loss_min" not in summaries[-1]:
            raise KeyError(
                f"Missing 'loss_min' in last training summary of {metrics_path}. "
                "This run predates min-loss tracking; retrain to record it."
            )
        return float(summaries[-1]["loss_min"])

    if metric_name not in metrics:
        raise KeyError(f"Missing '{metric_name}' in {metrics_path}")
    return float(metrics[metric_name])


def collect_runs(run_root: Path, metric_name: str) -> list[RunRecord]:
    """Find every run (by its metrics.json) under run_root and read its dims + metric."""
    runs: list[RunRecord] = []
    if not run_root.exists():
        return runs

    for metrics_path in sorted(run_root.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        config = metrics.get("config", {})
        for key in ("num_encoder_layers", "nhead", "d_model", "num_points", "epochs"):
            if key not in config:
                raise KeyError(f"Missing 'config.{key}' in {metrics_path}")

        runs.append(
            RunRecord(
                layers=int(config["num_encoder_layers"]),
                heads=int(config["nhead"]),
                d_model=int(config["d_model"]),
                num_points=int(config["num_points"]),
                trained_epochs=int(config["epochs"]),
                value=_load_metric(metrics_path, metric_name),
                run_dir=str(metrics_path.parent),
            )
        )
    return runs


def build_matrix(runs: list[RunRecord], row_attr: str, col_attr: str) -> tuple[np.ndarray, list[int], list[int]]:
    rows_sorted = sorted({getattr(r, row_attr) for r in runs})
    cols_sorted = sorted({getattr(r, col_attr) for r in runs})
    matrix = np.full((len(rows_sorted), len(cols_sorted)), np.nan, dtype=float)

    row_index = {value: idx for idx, value in enumerate(rows_sorted)}
    col_index = {value: idx for idx, value in enumerate(cols_sorted)}
    for item in runs:
        matrix[row_index[getattr(item, row_attr)], col_index[getattr(item, col_attr)]] = item.value

    return matrix, rows_sorted, cols_sorted


def save_heatmap(
    matrix: np.ndarray,
    rows: list[int],
    cols: list[int],
    output_path: Path,
    metric_name: str,
    trained_epochs: int | None,
    xlabel: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, origin="lower")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels([str(col) for col in cols])
    ax.set_yticklabels([str(row) for row in rows])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if trained_epochs is None:
        title = f"Model performance heatmap ({metric_name})"
    else:
        title = f"Model performance heatmap ({metric_name}, trained epochs: {trained_epochs})"
    ax.set_title(title)

    for row_idx in range(len(rows)):
        for col_idx in range(len(cols)):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "missing"
            else:
                if trained_epochs is None:
                    label = f"{value:.3f}"
                else:
                    label = f"{value:.3f}\n{trained_epochs} ep"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(
    summary_path: Path,
    matrix: np.ndarray,
    rows: list[int],
    cols: list[int],
    runs: list[RunRecord],
    metric_name: str,
    row_name: str,
    col_name: str,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted({run.trained_epochs for run in runs})
    payload = {
        "metric": metric_name,
        row_name: rows,
        col_name: cols,
        "trained_epochs": epochs[0] if len(epochs) == 1 else epochs,
        "matrix": matrix.tolist(),
        "runs": [asdict(run) for run in runs],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_stem(metric_name: str) -> str:
    if metric_name == "final_epoch_loss":
        return "final_epoch_loss_heatmap"
    if metric_name == "last_window_loss_avg":
        return "final_window_loss_avg_heatmap"
    if metric_name == "last_window_loss_min":
        return "final_window_loss_min_heatmap"
    return f"{metric_name}_heatmap"


def _metric_title(metric_name: str) -> str:
    if metric_name == "final_epoch_loss":
        return "Final epoch loss"
    if metric_name == "last_window_loss_avg":
        return "Final average loss (last 100 epochs)"
    if metric_name == "last_window_loss_min":
        return "Final minimum loss (last 100 epochs)"
    return metric_name.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a model-sweep heatmap from run metrics.")
    parser.add_argument("--run-root", type=str, default="runs", help="Root directory containing run subdirectories.")
    parser.add_argument(
        "--grid",
        type=str,
        default="layers_heads",
        choices=sorted(GRID_SPECS.keys()),
        help="Which two parameters form the heatmap axes.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["both", "final_epoch_loss", "last_window_loss_min", "last_window_loss_avg"],
        help="Which heatmap metric(s) to visualize. 'both' = final epoch loss + min loss of last 100 epochs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis",
        help="Output directory for the generated heatmap images.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="analysis",
        help="Output directory for the JSON summary data.",
    )
    args = parser.parse_args()

    spec = GRID_SPECS[args.grid]
    run_root = Path(args.run_root)
    output_dir = Path(args.output)
    summary_dir = Path(args.summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [args.metric] if args.metric != "both" else ["final_epoch_loss", "last_window_loss_min"]

    for metric_name in metric_names:
        runs = collect_runs(run_root, metric_name)
        if not runs:
            raise RuntimeError(f"No run directories with metrics found in {run_root}")
        matrix, rows, cols = build_matrix(runs, spec["row_attr"], spec["col_attr"])

        epochs = sorted({run.trained_epochs for run in runs})
        trained_epochs = epochs[0] if len(epochs) == 1 else None

        output_path = output_dir / f"{_output_stem(metric_name)}.png"
        summary_path = summary_dir / f"{_output_stem(metric_name)}.json"

        save_heatmap(matrix, rows, cols, output_path, _metric_title(metric_name), trained_epochs,
                     xlabel=spec["xlabel"], ylabel=spec["ylabel"])
        save_summary_json(summary_path, matrix, rows, cols, runs, metric_name,
                          row_name=spec["row_attr"], col_name=spec["col_attr"])

        print(f"Saved heatmap to {output_path}")
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
