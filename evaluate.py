import torch
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from data_generator import generate_circle_data
from pathlib import Path

from run_config import (
    RunConfig,
    build_model,
    config_path_for_run,
    load_config,
    model_path_for_run,
    visualizations_dir_for_run,
)

def evaluate_model(
    model_path=None,
    num_points=None,
    visualize=False,
    num_examples=3,
    add_noise=False,
    noise_std=0.05,
    add_clutter=False,
    clutter_fraction=0.1,
    config_path=None,
    run_dir=None,
    viz_dir=None,
):
    """
    Loads a trained model and evaluates its performance on a fixed test set.

    Args:
        model_path (str): Path to the saved model state dictionary.
        num_points (int): The number of points to use in the test sample.
        visualize (bool): If True, save annotated prediction plots for direct visual inspection.
        num_examples (int): Number of example visualizations to save when visualize is True.
        add_noise (bool): If True, add Gaussian noise to grayscale input values.
        noise_std (float): Standard deviation of Gaussian grayscale noise.
        add_clutter (bool): If True, add clutter to grayscale inputs.
        clutter_fraction (float): Fraction of points affected by clutter.
        viz_dir (str): Directory to write visualizations to (defaults to the run's
            visualizations dir, or "evaluation_viz" when no run dir is given).
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set a fixed seed for reproducibility of the test set
    np.random.seed(42)
    torch.manual_seed(42)

    config = None
    if config_path is not None or run_dir is not None:
        resolved_config_path = Path(config_path) if config_path is not None else config_path_for_run(run_dir)
        config = load_config(resolved_config_path)
        if run_dir is None:
            run_dir = resolved_config_path.parent
        if model_path is None:
            model_path = model_path_for_run(run_dir)
        if num_points is None:
            num_points = config.num_points

    if config is None:
        if num_points is None:
            num_points = 500
        config = RunConfig(num_points=num_points, max_seq_len=num_points * 2)

    if model_path is None:
        model_path = "attention_circle_detector.pth"

    if num_points > config.max_seq_len:
        raise ValueError(
            f"num_points ({num_points}) exceeds the trained model capacity ({config.max_seq_len})."
        )

    # --- Load Model ---
    # Instantiate the model architecture
    model = build_model(config).to(device)

    # Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please run train.py first to generate the model file.")
        return

    # Set the model to evaluation mode
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- Generate Fixed Test Data ---
    # We generate one large "batch" for testing
    points, true_vectors, true_labels, circle_info = generate_circle_data(
        num_points,
        add_noise=add_noise,
        noise_std=noise_std,
        add_clutter=add_clutter,
        clutter_fraction=clutter_fraction,
    )

    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device) # Add batch dimension
    vectors_tensor = torch.from_numpy(true_vectors).float().unsqueeze(0).to(device)
    labels_tensor = torch.from_numpy(true_labels).float().unsqueeze(0).to(device)

    print(f"Generated a fixed test set with {num_points} points.")

    # --- Evaluate ---
    with torch.no_grad(): # Disable gradient calculation for inference
        # Forward pass
        pred_vectors, pred_probs, _ = model(points_tensor)

        # Calculate loss
        loss = model.loss_function(pred_vectors, pred_probs, vectors_tensor, labels_tensor)

    print("-" * 30)
    print(f"Final Loss on Test Set: {loss.item():.6f}")
    print("-" * 30)

    if run_dir is not None:
        metrics_path = Path(run_dir) / "evaluation_metrics.json"
        evaluation_metrics = {
            "test_loss": loss.item(),
            "num_points": num_points,
            "noise": add_noise,
            "noise_std": noise_std,
            "clutter": add_clutter,
            "clutter_fraction": clutter_fraction,
        }
        metrics_path.write_text(json.dumps(evaluation_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved evaluation metrics to {metrics_path}")

    # --- Optional Visualization ---
    if visualize:
        if viz_dir is not None:
            output_dir = Path(viz_dir)
        elif run_dir is not None:
            output_dir = visualizations_dir_for_run(run_dir)
        else:
            output_dir = Path("evaluation_viz")
        save_prediction_visualizations(
            model,
            device,
            num_points=num_points,
            num_examples=num_examples,
            add_noise=add_noise,
            noise_std=noise_std,
            add_clutter=add_clutter,
            clutter_fraction=clutter_fraction,
            output_dir=output_dir,
        )


def save_prediction_visualizations(
    model,
    device,
    num_points,
    num_examples,
    add_noise=False,
    noise_std=0.05,
    add_clutter=False,
    clutter_fraction=0.1,
    output_dir="evaluation_viz",
):
    """
    Generate fresh examples, run inference, report per-example metrics and save annotated
    plots comparing ground truth vs. the model's predicted circle for direct visual checks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    print(f"\n--- Saving {num_examples} prediction visualizations to {output_dir}/ ---")

    for example_idx in range(num_examples):
        points, true_vectors, true_labels, circle_info = generate_circle_data(
            num_points,
            add_noise=add_noise,
            noise_std=noise_std,
            add_clutter=add_clutter,
            clutter_fraction=clutter_fraction,
        )
        true_center, true_radius = circle_info

        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_vectors, pred_probs, _ = model(points_tensor)

        pred_vectors_np = pred_vectors.squeeze(0).cpu().numpy()
        pred_probs_np = pred_probs.squeeze(0).cpu().numpy().squeeze(-1)

        xy_points = points[:, :2]
        estimated_center = (xy_points + pred_vectors_np).mean(axis=0)
        pred_inside = (pred_probs_np > 0.5).astype(int)

        # Per-example metrics for quick numeric verification
        vector_error = np.mean(np.linalg.norm(pred_vectors_np - true_vectors, axis=1))
        classification_accuracy = np.mean(pred_inside == true_labels)
        center_error = np.linalg.norm(estimated_center - true_center)

        print(
            f"Example {example_idx + 1}: center error {center_error:.4f} | "
            f"vector MSE {vector_error:.4f} | accuracy {classification_accuracy:.4f}"
        )

        _save_prediction_plot(
            xy_points,
            true_labels,
            pred_inside,
            true_center,
            true_radius,
            estimated_center,
            example_idx + 1,
            add_noise,
            add_clutter,
            output_dir / f"prediction_viz_{example_idx + 1:02d}.png",
        )

    print(f"Saved {num_examples} visualization images to {output_dir}/")


def _save_prediction_plot(
    points,
    true_labels,
    pred_inside,
    true_center,
    true_radius,
    estimated_center,
    example_idx,
    add_noise,
    add_clutter,
    save_path,
):
    """Render and save a single ground-truth vs. prediction plot."""
    inside_points = points[true_labels == 1]
    outside_points = points[true_labels == 0]
    pred_inside_points = points[pred_inside == 1]
    pred_outside_points = points[pred_inside == 0]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', alpha=0.6, s=50,
               label='Inside (Ground Truth)', marker='o')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], color='red', alpha=0.6, s=50,
               label='Outside (Ground Truth)', marker='o')

    ax.scatter(pred_inside_points[:, 0], pred_inside_points[:, 1], s=100, marker='s',
               facecolors='none', edgecolors='blue', linewidths=1.5, label='Pred Inside')
    ax.scatter(pred_outside_points[:, 0], pred_outside_points[:, 1], s=100, marker='s',
               facecolors='none', edgecolors='red', linewidths=1.5, label='Pred Outside')

    circle_true = plt.Circle(true_center, true_radius, color='green', fill=False,
                             linestyle='--', linewidth=2.5, label='Ground Truth Circle')
    ax.add_artist(circle_true)
    ax.plot(true_center[0], true_center[1], 'go', markersize=12, label='True Center')

    ax.plot(estimated_center[0], estimated_center[1], 'r*', markersize=20, label='Estimated Center')
    circle_est = plt.Circle(estimated_center, true_radius, color='orange', fill=False,
                            linestyle=':', linewidth=2, label='Estimated Circle (true radius)')
    ax.add_artist(circle_est)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-coordinate', fontsize=12)
    ax.set_ylabel('Y-coordinate', fontsize=12)

    title = f"Circle Detection Prediction (Example {example_idx})"
    if add_noise:
        title += " [with Noise]"
    if add_clutter:
        title += " [with Clutter]"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Attention Circle Detector model.")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the saved model file.')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to a saved run config file.')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Path to a run directory created by train.py.')
    parser.add_argument('--num-points', type=int, default=None,
                        help='Number of points for the test set.')
    parser.add_argument('--visualize', action='store_true',
                        help='Save annotated prediction plots for direct visual inspection.')
    parser.add_argument('--num-examples', type=int, default=3,
                        help='Number of example visualizations to save when --visualize is set.')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Directory to write visualizations to (defaults to the run dir).')
    parser.add_argument('--noise', action='store_true',
                        help='Enable Gaussian noise in grayscale input values.')
    parser.add_argument('--noise-std', type=float, default=0.05,
                        help='Standard deviation for grayscale Gaussian noise.')
    parser.add_argument('--clutter', action='store_true',
                        help='Enable clutter by randomly perturbing grayscale values of some points.')
    parser.add_argument('--clutter-fraction', type=float, default=0.1,
                        help='Fraction of points affected by clutter when --clutter is set.')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        num_points=args.num_points,
        visualize=args.visualize,
        num_examples=args.num_examples,
        add_noise=args.noise,
        noise_std=args.noise_std,
        add_clutter=args.clutter,
        clutter_fraction=args.clutter_fraction,
        config_path=args.config_path,
        run_dir=args.run_dir,
        viz_dir=args.viz_dir,
    )
