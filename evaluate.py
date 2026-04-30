import torch
import numpy as np
import argparse
from data_generator import generate_circle_data, visualize_data
from model import AttentionCircleDetector

def evaluate_model(model_path, num_points=500, visualize=False):
    """
    Loads a trained model and evaluates its performance on a fixed test set.

    Args:
        model_path (str): Path to the saved model state dictionary.
        num_points (int): The number of points to use in the test sample.
        visualize (bool): If True, visualizes one of the test predictions.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set a fixed seed for reproducibility of the test set
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Load Model ---
    # Instantiate the model architecture
    # The parameters (d_model, nhead, etc.) must match the saved model
    model = AttentionCircleDetector(max_seq_len=num_points * 2).to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please run train.py first to generate the model file.")
        return
        
    # Set the model to evaluation mode
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- Generate Fixed Test Data ---
    # We generate one large "batch" for testing
    points, true_vectors, true_labels, circle_info = generate_circle_data(num_points)
    
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

    # --- Optional Visualization ---
    if visualize:
        print("Visualizing model predictions on the test set...")
        # We need to get the data back to numpy and remove the batch dimension
        points_np = points_tensor.squeeze(0).cpu().numpy()
        pred_vectors_np = pred_vectors.squeeze(0).cpu().numpy()
        
        # Calculate the predicted circle center for each point
        predicted_centers = points_np + pred_vectors_np
        
        # Estimate the circle center by averaging the predictions
        estimated_center = predicted_centers.mean(axis=0)
        
        visualize_predictions(points_np, true_labels, circle_info, estimated_center)

def visualize_predictions(points, labels, true_circle_params, estimated_center):
    """
    Visualizes the ground truth vs. the model's estimated circle center.
    """
    true_center, true_radius = true_circle_params
    inside_points = points[labels == 1]
    outside_points = points[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points
    ax.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', alpha=0.6, label='Inside Points')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], color='red', alpha=0.6, label='Outside Points')

    # Plot Ground Truth Circle
    circle_shape = plt.Circle(true_center, true_radius, color='green', fill=False, linestyle='--', linewidth=2, label='Ground Truth Circle')
    ax.add_artist(circle_shape)
    ax.plot(true_center[0], true_center[1], 'go', markersize=10, label='Ground Truth Center')

    # Plot Estimated Center
    ax.plot(estimated_center[0], estimated_center[1], 'rx', markersize=12, mew=2, label='Estimated Center')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Model Evaluation: Ground Truth vs. Prediction")
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Attention Circle Detector model.")
    parser.add_argument('--model-path', type=str, default='attention_circle_detector.pth',
                        help='Path to the saved model file.')
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of points for the test set.')
    parser.add_argument('--visualize', action='store_true',
                        help='Include this flag to visualize the model prediction on the test set.')
    
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, num_points=args.num_points, visualize=args.visualize)
