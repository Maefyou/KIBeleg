import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import CircleDataset
from model import TransformerModel
import time
from tqdm import tqdm

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
    # Hyperparameters
    batch_size = 1
    epochs = 5
    lr = 0.0001
    d_model = 128
    nhead = 4
    num_encoder_layers = 3
    dim_feedforward = 512

    # Data loaders
    train_dataset = CircleDataset(data_path='data/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = CircleDataset(data_path='data/validation')
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = TransformerModel(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward
    )

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # Save the model
    torch.save(trained_model.state_dict(), 'circle_detector_model.pth')
    print("Model saved to circle_detector_model.pth")

if __name__ == '__main__':
    main()
