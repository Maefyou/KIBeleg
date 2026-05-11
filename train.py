import torch
import torch.optim as optim
from data_generator import generate_circle_data
from model import AttentionCircleDetector
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from collections import deque


def format_eta(seconds):
    """Format a duration in seconds as H:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def generate_visualizations(model, num_visualizations=10, num_points=200, add_noise=False, add_clutter=False):
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
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"validation_viz_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
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
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
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
            save_path = f"{output_dir}/validation_viz_{viz_idx + 1:02d}.png"
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

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = AttentionCircleDetector(input_dim=3, max_seq_len=NUM_POINTS_PER_SAMPLE * 2).to(device) # Ensure max_len is sufficient
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Points per Sample: {NUM_POINTS_PER_SAMPLE}, LR: {LEARNING_RATE}")
    print(f"Noise: {args.noise} (std={args.noise_std}), Clutter: {args.clutter} (fraction={args.clutter_fraction})")

    # --- Training Loop ---
    recent_losses = deque(maxlen=100)
    training_start = time.perf_counter()

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
            elapsed = time.perf_counter() - training_start
            avg_epoch_time = elapsed / (epoch + 1)
            remaining_epochs = NUM_EPOCHS - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            window_start = epoch + 2 - window_size
            window_end = epoch + 1
            print(
                f"Epochs {window_start}-{window_end}/{NUM_EPOCHS} | "
                f"Loss avg: {avg_loss:.6f} | "
                f"Last epoch: {epoch_loss:.6f} | Epoch time: {epoch_duration:.2f}s | "
                f"ETA: {format_eta(eta_seconds)}"
            )

    print("--- Training Finished ---")

    # --- Save the trained model ---
    model_save_path = "attention_circle_detector.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Generate validation visualizations ---
    generate_visualizations(
        model,
        num_visualizations=10,
        num_points=200,
        add_noise=args.noise,
        add_clutter=args.clutter,
    )

if __name__ == '__main__':
    main()
