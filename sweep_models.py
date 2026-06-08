import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def resolve_d_model(base_d_model: int, num_heads: int) -> int:
    """Return a valid d_model for the given head count.

    Multi-head attention splits d_model into num_heads equal slices, so d_model
    must be divisible by num_heads; positional encoding additionally needs it to
    be even. base_d_model (120) satisfies this for heads 5, 6, 8 but not 7, so for
    heads=7 we bump up to the nearest even multiple (126 = 7 * 18). The adapted
    value is reflected in the run name (..._dmodel_126) and logged by the caller.
    """
    d_model = base_d_model
    while d_model % num_heads != 0 or d_model % 2 != 0:
        d_model += 1
    return d_model


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

    failures: list[str] = []

    for num_layers, num_heads in combinations:
        model_d_model = resolve_d_model(args.d_model, num_heads)
        run_name = f"layers_{num_layers}_heads_{num_heads}_dmodel_{model_d_model}"
        run_dir = run_root / run_name

        if model_d_model != args.d_model:
            print(
                f"[ADAPT] heads={num_heads}: base d_model {args.d_model} not divisible "
                f"by {num_heads}; using d_model={model_d_model} for this run"
            )

        # A run only counts as "done" once metrics.json was written at the end of
        # training. The run directory itself is created at the start of train.py, so
        # checking mere directory existence would wrongly skip an interrupted model.
        if args.skip_existing and (run_dir / "metrics.json").exists():
            print(f"[SKIP] {run_name} already completed")
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
            str(model_d_model),
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
            # Don't abort the whole sweep on one bad model; record and keep going so
            # the remaining configurations still get trained.
            print(
                f"[FAIL] {run_name} exited with code {result.returncode}; continuing with the rest"
            )
            failures.append(run_name)

    if failures:
        print(f"\nSweep finished with {len(failures)} failed run(s):")
        for name in failures:
            print(f"  - {name}")
        raise RuntimeError(f"{len(failures)} run(s) failed during the sweep")

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
