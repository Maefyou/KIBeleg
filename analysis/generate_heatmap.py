from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR_PATTERN = re.compile(r"layers_(\d+)_heads_(\d+)_dmodel_(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    layers: int
    heads: int
    d_model: int
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

    if metric_name not in metrics:
        raise KeyError(f"Missing '{metric_name}' in {metrics_path}")
    return float(metrics[metric_name])


def _load_trained_epochs(metrics_path: Path) -> int:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    config = metrics.get("config", {})
    if "epochs" not in config:
        raise KeyError(f"Missing 'config.epochs' in {metrics_path}")
    return int(config["epochs"])


def collect_heatmap_data(run_root: Path, metric_name: str) -> tuple[np.ndarray, list[int], list[int], list[RunRecord]]:
    runs: list[RunRecord] = []
    layer_values: set[int] = set()
    head_values: set[int] = set()

    for run_dir in sorted(run_root.iterdir() if run_root.exists() else []):
        if not run_dir.is_dir():
            continue

        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue

        layers = int(match.group(1))
        heads = int(match.group(2))
        d_model = int(match.group(3))
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metric_value = _load_metric(metrics_path, metric_name)
        trained_epochs = _load_trained_epochs(metrics_path)
        runs.append(
            RunRecord(
                layers=layers,
                heads=heads,
                d_model=d_model,
                trained_epochs=trained_epochs,
                value=metric_value,
                run_dir=str(run_dir),
            )
        )
        layer_values.add(layers)
        head_values.add(heads)

    layers_sorted = sorted(layer_values)
    heads_sorted = sorted(head_values)
    matrix = np.full((len(layers_sorted), len(heads_sorted)), np.nan, dtype=float)

    layer_index = {value: idx for idx, value in enumerate(layers_sorted)}
    head_index = {value: idx for idx, value in enumerate(heads_sorted)}

    for item in runs:
        row = layer_index[item.layers]
        col = head_index[item.heads]
        matrix[row, col] = item.value

    return matrix, layers_sorted, heads_sorted, runs


def save_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    heads: list[int],
    output_path: Path,
    metric_name: str,
    trained_epochs: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, origin="lower")

    ax.set_xticks(np.arange(len(heads)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels([str(head) for head in heads])
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xlabel("Number of heads")
    ax.set_ylabel("Number of layers")
    if trained_epochs is None:
        title = f"Model performance heatmap ({metric_name})"
    else:
        title = f"Model performance heatmap ({metric_name}, trained epochs: {trained_epochs})"
    ax.set_title(title)

    for row_idx in range(len(layers)):
        for col_idx in range(len(heads)):
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


def save_summary_json(summary_path: Path, matrix: np.ndarray, layers: list[int], heads: list[int], runs: list[RunRecord], metric_name: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted({run.trained_epochs for run in runs})
    payload = {
        "metric": metric_name,
        "layers": layers,
        "heads": heads,
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
    return f"{metric_name}_heatmap"


def _metric_title(metric_name: str) -> str:
    if metric_name == "final_epoch_loss":
        return "Final epoch loss"
    if metric_name == "last_window_loss_avg":
        return "Final average loss (last 100 epochs)"
    return metric_name.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a layers/heads heatmap from run metrics.")
    parser.add_argument("--run-root", type=str, default="runs", help="Root directory containing run subdirectories.")
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["both", "final_epoch_loss", "last_window_loss_avg"],
        help="Which heatmap metric(s) to visualize.",
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

    run_root = Path(args.run_root)
    output_dir = Path(args.output)
    summary_dir = Path(args.summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    metric_names = [args.metric] if args.metric != "both" else ["final_epoch_loss", "last_window_loss_avg"]

    for metric_name in metric_names:
        matrix, layers, heads, runs = collect_heatmap_data(run_root, metric_name)
        if not layers or not heads:
            raise RuntimeError(f"No run directories with metrics found in {run_root}")

        epochs = sorted({run.trained_epochs for run in runs})
        trained_epochs = epochs[0] if len(epochs) == 1 else None

        output_path = output_dir / f"{_output_stem(metric_name)}.png"
        summary_path = summary_dir / f"{_output_stem(metric_name)}.json"

        save_heatmap(matrix, layers, heads, output_path, _metric_title(metric_name), trained_epochs)
        save_summary_json(summary_path, matrix, layers, heads, runs, metric_name)

        print(f"Saved heatmap to {output_path}")
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
