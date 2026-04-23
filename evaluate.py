import torch
from torch.utils.data import DataLoader
from dataset import CircleDataset
from model import TransformerModel
import numpy as np
import time
from tqdm import tqdm

def evaluate_model(model, test_loader):
    """Evaluates the model on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_loss = 0
    total_vector_error = 0
    total_prob_accuracy = 0
    
    start_time = time.time()

    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc="[Testing]")
        for i, batch in enumerate(test_iterator):
            points = batch['points'].to(device)
            true_vectors = batch['vectors'].to(device)
            true_is_inside = batch['is_inside'].to(device)
            
            pred_vectors, pred_probs = model(points)
            
            # Loss
            mse_loss = torch.nn.MSELoss()(pred_vectors, true_vectors)
            bce_loss = torch.nn.BCELoss()(pred_probs, true_is_inside)
            loss = mse_loss + bce_loss
            total_loss += loss.item()
            
            # Vector error
            vector_error = torch.mean(torch.sqrt(torch.sum((pred_vectors - true_vectors)**2, dim=2))).item()
            total_vector_error += vector_error

            # Probability accuracy
            predicted_labels = (pred_probs > 0.5).float()
            prob_accuracy = (predicted_labels == true_is_inside).float().mean().item()
            total_prob_accuracy += prob_accuracy

            test_iterator.set_postfix(vector_error=f"{vector_error:.4f}", prob_accuracy=f"{prob_accuracy:.4f}")

    end_time = time.time()
    evaluation_duration = end_time - start_time
    
    avg_loss = total_loss / len(test_loader)
    avg_vector_error = total_vector_error / len(test_loader)
    avg_prob_accuracy = total_prob_accuracy / len(test_loader)
    
    print("\nEvaluation Summary:")
    print(f"  - Average Loss: {avg_loss:.4f}")
    print(f"  - Average Vector Error (L2 norm): {avg_vector_error:.4f}")
    print(f"  - Average Probability Accuracy: {avg_prob_accuracy:.4f}")
    print(f"  - Evaluation Duration: {evaluation_duration:.2f} seconds")
    print(f"  - Points processed per second: {len(test_loader.dataset) * test_loader.dataset[0]['points'].shape[0] / evaluation_duration:.2f}")


def main():
    # Hyperparameters from training
    batch_size = 1
    d_model = 128
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512

    # Data loader
    test_dataset = CircleDataset(data_path='data/test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = TransformerModel(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward
    )
    
    # Load the trained model
    model_path = 'circle_detector_model.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
