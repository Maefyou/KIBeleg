import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from data_generator import generate_circle_data
from model import AttentionCircleDetector


def visualize_model_predictions(
    model_path,
    num_points=200,
    num_examples=1,
    add_noise=False,
    add_clutter=False,
    save_prefix="prediction_viz",
):
    """
    Generates synthetic circle data, runs model inference, and visualizes predictions.
    
    Args:
        model_path (str): Path to the saved model.
        num_points (int): Number of sample points per example.
        num_examples (int): Number of examples to visualize.
        add_noise (bool): Whether to add noise to grayscale values.
        add_clutter (bool): Whether to add clutter.
        save_prefix (str): Prefix for saved visualization files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = AttentionCircleDetector(input_dim=3, max_seq_len=num_points * 2).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'pos_encoder.pe' in checkpoint:
            checkpoint.pop('pos_encoder.pe')
        model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return

    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Generate and visualize examples
    for example_idx in range(num_examples):
        print(f"\n--- Example {example_idx + 1} ---")

        # Generate data
        points, true_vectors, true_labels, circle_info = generate_circle_data(
            num_points,
            add_noise=add_noise,
            add_clutter=add_clutter,
        )

        true_center, true_radius = circle_info

        # Run inference
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device)
        with torch.no_grad():
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

        # Compute metrics
        vector_error = np.mean(np.linalg.norm(pred_vectors_np - true_vectors, axis=1))
        classification_accuracy = np.mean(pred_inside == true_labels)
        center_error = np.linalg.norm(estimated_center - true_center)

        print(f"True center: {true_center}, Radius: {true_radius:.4f}")
        print(f"Estimated center: {estimated_center}")
        print(f"Center localization error: {center_error:.4f}")
        print(f"Vector MSE: {vector_error:.4f}")
        print(f"Classification accuracy: {classification_accuracy:.4f}")

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot ground truth points
        inside_mask = true_labels == 1
        outside_mask = true_labels == 0

        inside_points = xy_points[inside_mask]
        outside_points = xy_points[outside_mask]

        ax.scatter(
            inside_points[:, 0], inside_points[:, 1],
            color='blue', alpha=0.6, s=50, label='Inside (Ground Truth)',
            marker='o'
        )
        ax.scatter(
            outside_points[:, 0], outside_points[:, 1],
            color='red', alpha=0.6, s=50, label='Outside (Ground Truth)',
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
            label='Ground Truth Circle'
        )
        ax.add_artist(circle_true)
        ax.plot(true_center[0], true_center[1], 'go', markersize=12, label='True Center')

        # Plot estimated center
        ax.plot(
            estimated_center[0], estimated_center[1],
            'r*', markersize=20, label='Estimated Center'
        )

        # Estimated circle (from mean center)
        circle_est = plt.Circle(
            estimated_center, true_radius,
            color='orange', fill=False, linestyle=':', linewidth=2,
            label='Estimated Circle (true radius)'
        )
        ax.add_artist(circle_est)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)

        title = f"Circle Detection Prediction (Example {example_idx + 1})"
        if add_noise:
            title += " [with Noise]"
        if add_clutter:
            title += " [with Clutter]"
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = f"{save_prefix}_{example_idx + 1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on circle detection task."
    )
    parser.add_argument(
        '--model-path', type=str, default='attention_circle_detector.pth',
        help='Path to the saved model.'
    )
    parser.add_argument(
        '--num-points', type=int, default=200,
        help='Number of sample points per example.'
    )
    parser.add_argument(
        '--num-examples', type=int, default=3,
        help='Number of examples to visualize.'
    )
    parser.add_argument(
        '--noise', action='store_true',
        help='Add Gaussian noise to grayscale values.'
    )
    parser.add_argument(
        '--clutter', action='store_true',
        help='Add clutter to grayscale values.'
    )
    parser.add_argument(
        '--save-prefix', type=str, default='prediction_viz',
        help='Prefix for saved visualization files.'
    )

    args = parser.parse_args()

    visualize_model_predictions(
        model_path=args.model_path,
        num_points=args.num_points,
        num_examples=args.num_examples,
        add_noise=args.noise,
        add_clutter=args.clutter,
        save_prefix=args.save_prefix,
    )
