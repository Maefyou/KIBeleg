import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def run_sweep(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(
            range(args.layers_min, args.layers_max + 1),
            range(args.heads_min, args.heads_max + 1),
        )
    )

    print(f"Starting sweep with {len(combinations)} model configurations")
    print(
        f"Layers: {args.layers_min}-{args.layers_max}, "
        f"Heads: {args.heads_min}-{args.heads_max}, d_model={args.d_model}"
    )

    for num_layers, num_heads in combinations:
        run_name = f"layers_{num_layers}_heads_{num_heads}_dmodel_{args.d_model}"
        run_dir = run_root / run_name

        if args.skip_existing and run_dir.exists():
            print(f"[SKIP] {run_name} already exists")
            continue

        train_command = [
            sys.executable,
            "train.py",
            "--num-points",
            str(args.num_points),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--d-model",
            str(args.d_model),
            "--nhead",
            str(num_heads),
            "--num-encoder-layers",
            str(num_layers),
            "--run-root",
            str(run_root),
        ]

        if args.noise:
            train_command.append("--noise")
            train_command.extend(["--noise-std", str(args.noise_std)])

        if args.clutter:
            train_command.append("--clutter")
            train_command.extend(["--clutter-fraction", str(args.clutter_fraction)])

        print(f"\n[RUN] {run_name}")
        result = subprocess.run(train_command, cwd=project_root)
        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed for layers={num_layers}, heads={num_heads} with exit code {result.returncode}"
            )

    print("\nSweep completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a layer/head model sweep.")
    parser.add_argument("--layers-min", type=int, default=1, help="Minimum number of encoder layers.")
    parser.add_argument("--layers-max", type=int, default=4, help="Maximum number of encoder layers.")
    parser.add_argument("--heads-min", type=int, default=1, help="Minimum number of attention heads.")
    parser.add_argument("--heads-max", type=int, default=4, help="Maximum number of attention heads.")
    parser.add_argument("--d-model", type=int, default=120, help="Transformer model dimension.")
    parser.add_argument("--num-points", type=int, default=200, help="Points per training sample.")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--run-root", type=str, default="runs", help="Root directory for sweep runs.")
    parser.add_argument("--noise", action="store_true", help="Enable noise during training.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Noise standard deviation.")
    parser.add_argument("--clutter", action="store_true", help="Enable clutter during training.")
    parser.add_argument("--clutter-fraction", type=float, default=0.1, help="Clutter fraction.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already existing run directories.")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
