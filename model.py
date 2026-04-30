import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class AttentionCircleDetector(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1, max_seq_len=5000):
        super(AttentionCircleDetector, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output head for the vector v_s
        self.vector_head = nn.Linear(d_model, 2)
        
        # Output head for the probability p_s
        self.probability_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        vectors = self.vector_head(output)
        probabilities = self.probability_head(output)
        
        return vectors, probabilities, output

    def loss_function(self, pred_vectors, pred_probs, true_vectors, true_labels):
        """
        Calculates the loss as defined in the project description.
        L((v', p'),(v, y)) = ||v' - v||^2_2 - y log p' - (1 - y) log(1 - p').
        
        Args:
            pred_vectors (torch.Tensor): Predicted vectors (v'). Shape: (batch_size, seq_len, 2)
            pred_probs (torch.Tensor): Predicted probabilities (p'). Shape: (batch_size, seq_len, 1)
            true_vectors (torch.Tensor): Ground truth vectors (v). Shape: (batch_size, seq_len, 2)
            true_labels (torch.Tensor): Ground truth labels (y). Shape: (batch_size, seq_len)
        """
        # Ensure true_labels has the same shape as pred_probs for broadcasting
        if true_labels.dim() == pred_probs.dim() - 1:
            true_labels = true_labels.unsqueeze(-1)

        # Vector loss: ||v' - v||^2_2
        vector_loss = torch.mean(torch.sum((pred_vectors - true_vectors) ** 2, dim=-1))
        
        # Classification loss: - y log p' - (1 - y) log(1 - p')
        # Adding a small epsilon to avoid log(0)
        epsilon = 1e-9
        classification_loss = -torch.mean(
            true_labels * torch.log(pred_probs + epsilon) + 
            (1 - true_labels) * torch.log(1 - pred_probs + epsilon)
        )
        
        total_loss = vector_loss + classification_loss
        return total_loss

if __name__ == '__main__':
    # --- Dummy Data ---
    BATCH_SIZE = 32
    SEQ_LENGTH = 100 # Number of points per sample
    INPUT_DIM = 2 # (x, y) coordinates

    # Create a model instance
    model = AttentionCircleDetector(input_dim=INPUT_DIM)

    # Generate some dummy input data
    # In a real scenario, this would come from our data_generator
    dummy_points = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    
    # Generate dummy ground truth
    dummy_true_vectors = torch.randn(BATCH_SIZE, SEQ_LENGTH, 2)
    dummy_true_labels = torch.randint(0, 2, (BATCH_SIZE, SEQ_LENGTH)).float()

    # --- Forward Pass ---
    pred_vectors, pred_probs, _ = model(dummy_points)

    # --- Calculate Loss ---
    loss = model.loss_function(pred_vectors, pred_probs, dummy_true_vectors, dummy_true_labels)

    print("--- Model Architecture ---")
    print(model)
    print("\n--- Input and Output Shapes ---")
    print(f"Input points shape:      {dummy_points.shape}")
    print(f"Predicted vectors shape:   {pred_vectors.shape}")
    print(f"Predicted probabilities shape: {pred_probs.shape}")
    print(f"Calculated loss:         {loss.item():.4f}")

    # --- Check a single prediction ---
    print("\n--- Example Prediction ---")
    print(f"Example predicted vector for first point: {pred_vectors[0, 0].detach().numpy()}")
    print(f"Example predicted probability for first point: {pred_probs[0, 0].item():.4f}")
