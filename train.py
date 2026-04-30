import torch
import torch.optim as optim
from data_generator import generate_circle_data
from model import AttentionCircleDetector
import numpy as np
import argparse

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
    model = AttentionCircleDetector(max_seq_len=NUM_POINTS_PER_SAMPLE * 2).to(device) # Ensure max_len is sufficient
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- Starting Training ---")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, Points per Sample: {NUM_POINTS_PER_SAMPLE}, LR: {LEARNING_RATE}")

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        # Generate a fresh batch of data for each epoch
        # This acts as our data loader
        points, vectors, labels, _ = generate_circle_data(
            num_points=NUM_POINTS_PER_SAMPLE * BATCH_SIZE
        )
        
        # Reshape data into batches
        points_batch = torch.from_numpy(points.reshape(BATCH_SIZE, NUM_POINTS_PER_SAMPLE, -1)).float().to(device)
        vectors_batch = torch.from_numpy(vectors.reshape(BATCH_SIZE, NUM_POINTS_PER_SAMPLE, -1)).float().to(device)
        labels_batch = torch.from_numpy(labels.reshape(BATCH_SIZE, NUM_POINTS_PER_SAMPLE)).float().to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        pred_vectors, pred_probs, _ = model(points_batch)

        # Calculate loss
        loss = model.loss_function(pred_vectors, pred_probs, vectors_batch, labels_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

    print("--- Training Finished ---")

    # --- Save the trained model ---
    model_save_path = "attention_circle_detector.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
