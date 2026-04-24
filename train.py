import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import CircleDataset
from model import TransformerModel
import time
from tqdm import tqdm
import argparse

def loss_function(pred_vectors, pred_probs, true_vectors, true_is_inside):
    """Custom loss function."""
    mse_loss = nn.MSELoss()(pred_vectors, true_vectors)
    bce_loss = nn.BCELoss()(pred_probs, true_is_inside)
    return mse_loss + bce_loss

def train_model(model, train_loader, val_loader, epochs=10, lr=0.0001):
    """Training loop for the transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for i, batch in enumerate(train_iterator):
            points = batch['points'].to(device)
            true_vectors = batch['vectors'].to(device)
            true_is_inside = batch['is_inside'].to(device)
            
            optimizer.zero_grad()
            
            pred_vectors, pred_probs = model(points)
            
            loss = loss_function(pred_vectors, pred_probs, true_vectors, true_is_inside)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            train_iterator.set_postfix(loss=f"{loss.item():.4f}")

        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  - Average Training Loss: {avg_loss:.4f}")
        print(f"  - Epoch Duration: {epoch_duration:.2f} seconds")

        # Validation
        model.eval()
        val_loss = 0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
        with torch.no_grad():
            for batch in val_iterator:
                points = batch['points'].to(device)
                true_vectors = batch['vectors'].to(device)
                true_is_inside = batch['is_inside'].to(device)
                
                pred_vectors, pred_probs = model(points)
                
                loss = loss_function(pred_vectors, pred_probs, true_vectors, true_is_inside)
                val_loss += loss.item()
                val_iterator.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"  - Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")
    return model

def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train a circle detection model.')
    parser.add_argument('--model', type=str, default='transformer', help='Model to train (transformer or cnn).')
    parser.add_argument('--target_width', type=int, default=128, help='Target width for resizing images.')
    parser.add_argument('--target_height', type=int, default=128, help='Target height for resizing images.')
    args = parser.parse_args()

    # Data loaders
    train_dataset = CircleDataset(data_path='data/train', target_width=args.target_width, target_height=args.target_height)
    val_dataset = CircleDataset(data_path='data/validation', target_width=args.target_width, target_height=args.target_height)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Model
    if args.model == 'transformer':
        model = TransformerModel(width=args.target_width, height=args.target_height)
    elif args.model == 'cnn':
        # Placeholder for CNN model
        raise NotImplementedError("CNN model not yet implemented.")
    else:
        raise ValueError("Unsupported model type. Choose 'transformer' or 'cnn'.")
    
    # Training
    train_model(model, train_loader, val_loader, epochs=5, lr=0.0001)
    
    # Save the trained model
    torch.save(model.state_dict(), 'circle_detector_model.pth')
    print("Model saved to circle_detector_model.pth")

if __name__ == '__main__':
    main()
