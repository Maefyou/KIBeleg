"""Sweep num_points x d_model for a fixed layers/heads architecture.

Companion to sweep_models.py (which sweeps layers x heads). Here the architecture
depth/width in terms of layers and heads is held fixed (default 4x4) and we vary
the number of sample points per scene and the embedding dimension d_model.

d_model must be divisible by the number of heads (and even), so requested values
that are not are bumped up to the nearest valid one via resolve_d_model (e.g. with
4 heads: 90 -> 92, 150 -> 152). Each run is written to its own subdirectory named
points_{P}_dmodel_{D} (the default layers/heads/d_model name would collide across
different num_points), via train.py --run-name.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from sweep_models import resolve_d_model


def run_sweep(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    # Resolve each requested d_model to a value valid for the head count, de-duplicating
    # while preserving order.
    resolved = []
    for d in args.d_models:
        rd = resolve_d_model(d, args.heads)
        if rd != d:
            print(f"[ADAPT] d_model {d} not divisible by {args.heads} heads; using {rd}")
        if rd not in resolved:
            resolved.append(rd)

    combinations = [(p, d) for p in args.points for d in resolved]
    print(f"Starting sweep with {len(combinations)} configurations")
    print(f"Fixed: layers={args.layers}, heads={args.heads}")
    print(f"num_points: {args.points}")
    print(f"d_model:    {resolved}")

    failures: list[str] = []

    for num_points, d_model in combinations:
        run_name = f"points_{num_points}_dmodel_{d_model}"
        run_dir = run_root / run_name

        if args.skip_existing and (run_dir / "metrics.json").exists():
            print(f"[SKIP] {run_name} already completed")
            continue

        train_command = [
            sys.executable,
            "train.py",
            "--num-points", str(num_points),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--d-model", str(d_model),
            "--nhead", str(args.heads),
            "--num-encoder-layers", str(args.layers),
            "--run-root", str(run_root),
            "--run-name", run_name,
        ]
        if args.noise:
            train_command.extend(["--noise", "--noise-std", str(args.noise_std)])
        if args.clutter:
            train_command.extend(["--clutter", "--clutter-fraction", str(args.clutter_fraction)])

        print(f"\n[RUN] {run_name}")
        result = subprocess.run(train_command, cwd=project_root)
        if result.returncode != 0:
            print(f"[FAIL] {run_name} exited with code {result.returncode}; continuing")
            failures.append(run_name)

    if failures:
        print(f"\nSweep finished with {len(failures)} failed run(s):")
        for name in failures:
            print(f"  - {name}")
        raise RuntimeError(f"{len(failures)} run(s) failed during the sweep")

    print("\nSweep completed.")


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep num_points x d_model for fixed layers/heads.")
    parser.add_argument("--points", type=_int_list, default=[100, 150, 200, 250, 300],
                        help="Comma-separated num_points values.")
    parser.add_argument("--d-models", type=_int_list, default=[60, 90, 120, 150, 180],
                        help="Comma-separated requested d_model values (bumped to be head-divisible).")
    parser.add_argument("--layers", type=int, default=4, help="Fixed number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Fixed number of attention heads.")
    parser.add_argument("--epochs", type=int, default=15000, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--run-root", type=str, default="runs_points_dmodel", help="Root directory for runs.")
    parser.add_argument("--noise", action="store_true", default=True, help="Enable noise (on by default).")
    parser.add_argument("--no-noise", dest="noise", action="store_false", help="Disable noise.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Noise standard deviation.")
    parser.add_argument("--clutter", action="store_true", default=True, help="Enable clutter (on by default).")
    parser.add_argument("--no-clutter", dest="clutter", action="store_false", help="Disable clutter.")
    parser.add_argument("--clutter-fraction", type=float, default=0.1, help="Clutter fraction.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have metrics.json.")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
