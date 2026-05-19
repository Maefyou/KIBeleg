from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

from model import AttentionCircleDetector


@dataclass(frozen=True)
class RunConfig:
    input_dim: int = 3
    d_model: int = 120
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 5000
    num_points: int = 300
    epochs: int = 1000
    batch_size: int = 64
    lr: float = 0.00001
    noise: bool = False
    noise_std: float = 0.05
    clutter: bool = False
    clutter_fraction: float = 0.1


def validate_run_config(config: RunConfig) -> None:
    if config.d_model <= 0:
        raise ValueError(f"d_model must be positive, got {config.d_model}")
    if config.nhead <= 0:
        raise ValueError(f"nhead must be positive, got {config.nhead}")
    if config.num_encoder_layers <= 0:
        raise ValueError(f"num_encoder_layers must be positive, got {config.num_encoder_layers}")
    if config.d_model % 2 != 0:
        raise ValueError(
            f"d_model must be even for positional encoding, got {config.d_model}"
        )
    if config.d_model % config.nhead != 0:
        raise ValueError(
            f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})"
        )
    if config.max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {config.max_seq_len}")


def run_name(config: RunConfig) -> str:
    return f"layers_{config.num_encoder_layers}_heads_{config.nhead}_dmodel_{config.d_model}"


def run_dir(root_dir: str | Path, config: RunConfig) -> Path:
    return Path(root_dir) / run_name(config)


def model_path_for_run(run_directory: str | Path) -> Path:
    return Path(run_directory) / "attention_circle_detector.pth"


def config_path_for_run(run_directory: str | Path) -> Path:
    return Path(run_directory) / "config.json"


def metrics_path_for_run(run_directory: str | Path) -> Path:
    return Path(run_directory) / "metrics.json"


def architecture_path_for_run(run_directory: str | Path) -> Path:
    return Path(run_directory) / "architecture.txt"


def visualizations_dir_for_run(run_directory: str | Path) -> Path:
    return Path(run_directory) / "visualizations"


def save_config(config: RunConfig, path: str | Path) -> None:
    validate_run_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw_config: dict[str, Any] = json.load(handle)
    config = RunConfig(**raw_config)
    validate_run_config(config)
    return config


def build_model(config: RunConfig) -> AttentionCircleDetector:
    validate_run_config(config)
    return AttentionCircleDetector(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )
