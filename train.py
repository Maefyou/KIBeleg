from dataclasses import asdict
import torch
import torch.optim as optim
from data_generator import generate_circle_data
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from collections import deque

from run_config import (
    RunConfig,
    build_model,
    config_path_for_run,
    architecture_path_for_run,
    metrics_path_for_run,
    model_path_for_run,
    run_dir,
    save_config,
    visualizations_dir_for_run,
)


def format_eta(seconds):
    """Format a duration in seconds as H:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def generate_visualizations(
    model,
    num_visualizations=10,
    num_points=200,
    add_noise=False,
    add_clutter=False,
    output_dir=None,
):
    """
    Generate validation visualizations after training.
    
    Args:
        model: The trained model.
        num_visualizations (int): Number of visualization examples to generate.
        num_points (int): Number of sample points per visualization.
        add_noise (bool): Whether to add noise to test data.
        add_clutter (bool): Whether to add clutter to test data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    output_dir = Path(output_dir or "validation_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating {num_visualizations} validation visualizations ---")
    
    with torch.no_grad():
        for viz_idx in range(num_visualizations):
            # Generate data
            points, _, true_labels, circle_info = generate_circle_data(
                num_points,
                add_noise=add_noise,
                add_clutter=add_clutter,
            )
            
            true_center, true_radius = circle_info
            
            # Run inference
            points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
            pred_vectors, pred_probs, _ = model(points_tensor)
            
            pred_vectors_np = pred_vectors.squeeze(0).cpu().numpy()
            pred_probs_np = pred_probs.squeeze(0).cpu().numpy().squeeze(-1)
            
            # Extract XY coordinates
            xy_points = points[:, :2]
            
            # Estimate circle center from predictions
            predicted_centers = xy_points + pred_vectors_np
            estimated_center = predicted_centers.mean(axis=0)
            
            # Classify predictions as inside/outside based on probability
            pred_inside = (pred_probs_np > 0.5).astype(int)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot ground truth points
            inside_mask = true_labels == 1
            outside_mask = true_labels == 0
            
            inside_points = xy_points[inside_mask]
            outside_points = xy_points[outside_mask]
            
            ax.scatter(
                inside_points[:, 0], inside_points[:, 1],
                color='blue', alpha=0.6, s=50, label='Inside (True)',
                marker='o'
            )
            ax.scatter(
                outside_points[:, 0], outside_points[:, 1],
                color='red', alpha=0.6, s=50, label='Outside (True)',
                marker='o'
            )
            
            # Plot model predictions as edge markers
            pred_inside_mask = pred_inside == 1
            pred_outside_mask = pred_inside == 0
            
            pred_inside_points = xy_points[pred_inside_mask]
            pred_outside_points = xy_points[pred_outside_mask]
            
            ax.scatter(
                pred_inside_points[:, 0], pred_inside_points[:, 1],
                color='blue', alpha=0.3, s=100, marker='s',
                facecolors='none', edgecolors='blue', linewidths=1.5,
                label='Pred Inside'
            )
            ax.scatter(
                pred_outside_points[:, 0], pred_outside_points[:, 1],
                color='red', alpha=0.3, s=100, marker='s',
                facecolors='none', edgecolors='red', linewidths=1.5,
                label='Pred Outside'
            )
            
            # Plot ground truth circle
            circle_true = plt.Circle(
                true_center, true_radius,
                color='green', fill=False, linestyle='--', linewidth=2.5,
                label='True Circle'
            )
            ax.add_artist(circle_true)
            ax.plot(true_center[0], true_center[1], 'go', markersize=12, label='True Center')
            
            # Plot estimated center
            ax.plot(
                estimated_center[0], estimated_center[1],
                'r*', markersize=20, label='Estimated Center'
            )
            
            # Estimated circle
            circle_est = plt.Circle(
                estimated_center, true_radius,
                color='orange', fill=False, linestyle=':', linewidth=2,
                label='Estimated Circle'
            )
            ax.add_artist(circle_est)
            
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X-coordinate', fontsize=12)
            ax.set_ylabel('Y-coordinate', fontsize=12)
            
            title = f"Validation Example {viz_idx + 1}"
            if add_noise or add_clutter:
                conditions = []
                if add_noise:
                    conditions.append("Noise")
                if add_clutter:
                    conditions.append("Clutter")
                title += f" [{', '.join(conditions)}]"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Save figure
            save_path = output_dir / f"validation_viz_{viz_idx + 1:02d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved {num_visualizations} visualization images to {output_dir}/")


def main():
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Train an attention model for circle detection.")
    parser.add_argument('--num-points', type=int, default=300,
                        help='Number of sample points to generate per batch.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--d-model', type=int, default=120,
                        help='Transformer model dimension.')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num-encoder-layers', type=int, default=6,
                        help='Number of transformer encoder layers.')
    parser.add_argument('--run-root', type=str, default='runs',
                        help='Directory where run subdirectories are stored.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Override the run subdirectory name (default is derived from '
                             'layers/heads/d_model). Use this when sweeping a parameter not '
                             'captured by the default name, e.g. num_points, to avoid collisions.')
    parser.add_argument('--noise', action='store_true',
                        help='Enable Gaussian noise in grayscale input values.')
    parser.add_argument('--noise-std', type=float, default=0.05,
                        help='Standard deviation for grayscale Gaussian noise.')
    parser.add_argument('--clutter', action='store_true',
                        help='Enable clutter by randomly perturbing grayscale values of some points.')
    parser.add_argument('--clutter-fraction', type=float, default=0.1,
                        help='Fraction of points affected by clutter when --clutter is set.')
    args = parser.parse_args()

    # --- Training Parameters ---
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_POINTS_PER_SAMPLE = args.num_points
    LEARNING_RATE = args.lr

    config = RunConfig(
        input_dim=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=NUM_POINTS_PER_SAMPLE * 2,
        num_points=NUM_POINTS_PER_SAMPLE,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        noise=args.noise,
        noise_std=args.noise_std,
        clutter=args.clutter,
        clutter_fraction=args.clutter_fraction,
    )

    if args.run_name:
        run_directory = Path(args.run_root) / args.run_name
    else:
        run_directory = run_dir(args.run_root, config)
    run_directory.mkdir(parents=True, exist_ok=True)
    save_config(config, config_path_for_run(run_directory))

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = build_model(config).to(device)
    architecture_path_for_run(run_directory).write_text(f"{model}\n", encoding="utf-8")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Points per Sample: {NUM_POINTS_PER_SAMPLE}, LR: {LEARNING_RATE}")
    print(f"Noise: {args.noise} (std={args.noise_std}), Clutter: {args.clutter} (fraction={args.clutter_fraction})")

    # --- Training Loop ---
    recent_losses = deque(maxlen=100)
    training_start = time.perf_counter()
    training_summaries = []

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()

        # Generate a fresh batch of data for each epoch
        # This acts as our data loader
        points, vectors, labels, _ = generate_circle_data(
            num_points=NUM_POINTS_PER_SAMPLE * BATCH_SIZE,
            add_noise=args.noise,
            noise_std=args.noise_std,
            add_clutter=args.clutter,
            clutter_fraction=args.clutter_fraction,
        )
        
        # Reshape data into batches
        points_batch = torch.from_numpy(points.reshape((BATCH_SIZE, NUM_POINTS_PER_SAMPLE, -1))).float().to(device)
        vectors_batch = torch.from_numpy(vectors.reshape((BATCH_SIZE, NUM_POINTS_PER_SAMPLE, -1))).float().to(device)
        labels_batch = torch.from_numpy(labels.reshape((BATCH_SIZE, NUM_POINTS_PER_SAMPLE))).float().to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        pred_vectors, pred_probs, _ = model(points_batch)

        # Calculate loss
        loss = model.loss_function(pred_vectors, pred_probs, vectors_batch, labels_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        recent_losses.append(epoch_loss)

        epoch_duration = time.perf_counter() - epoch_start

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == NUM_EPOCHS - 1:
            window_size = len(recent_losses)
            avg_loss = sum(recent_losses) / window_size
            min_loss = min(recent_losses)
            elapsed = time.perf_counter() - training_start
            avg_epoch_time = elapsed / (epoch + 1)
            remaining_epochs = NUM_EPOCHS - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            window_start = epoch + 2 - window_size
            window_end = epoch + 1
            training_summaries.append(
                {
                    "window_start_epoch": window_start,
                    "window_end_epoch": window_end,
                    "loss_avg": avg_loss,
                    "loss_min": min_loss,
                    "last_epoch_loss": epoch_loss,
                    "epoch_time_seconds": epoch_duration,
                    "eta_seconds": eta_seconds,
                }
            )
            print(
                f"Epochs {window_start}-{window_end}/{NUM_EPOCHS} | "
                f"Loss avg: {avg_loss:.6f} | Loss min: {min_loss:.6f} | "
                f"Last epoch: {epoch_loss:.6f} | Epoch time: {epoch_duration:.2f}s | "
                f"ETA: {format_eta(eta_seconds)}"
            )

    print("--- Training Finished ---")

    # --- Save the trained model ---
    model_save_path = model_path_for_run(run_directory)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    metrics = {
        "config": asdict(config),
        "training_duration_seconds": time.perf_counter() - training_start,
        "final_epoch_loss": epoch_loss,
        "training_summaries": training_summaries,
    }
    with metrics_path_for_run(run_directory).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved metrics to {metrics_path_for_run(run_directory)}")
    print(f"Saved config to {config_path_for_run(run_directory)}")
    print(f"Saved architecture to {architecture_path_for_run(run_directory)}")

    # --- Generate validation visualizations ---
    generate_visualizations(
        model,
        num_visualizations=10,
        num_points=NUM_POINTS_PER_SAMPLE,
        add_noise=args.noise,
        add_clutter=args.clutter,
        output_dir=visualizations_dir_for_run(run_directory),
    )

    print(f"Run artifacts stored in {run_directory}")

if __name__ == '__main__':
    main()
