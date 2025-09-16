import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalized Linear model for time series feature extraction.
    This model normalizes the input data and applies linear layers for prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.1):
        super(NLinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # Normalize along the feature dimension
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - means) / stds
        
        # Extract the last time step features
        x_last = x_norm[:, -1, :]
        
        # Apply input projection
        h = self.input_projection(x_last)
        
        # Apply hidden layers
        for layer in self.layers:
            h = layer(h) + h  # Residual connection
        
        # Apply output projection
        output = self.output_projection(h)
        
        return output

class NLinearWithAttention(nn.Module):
    """
    Enhanced NLinear model with attention mechanism for time series feature extraction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.1):
        super(NLinearWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention layer
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # Normalize along the feature dimension
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - means) / stds
        
        # Apply input projection to all time steps
        h = self.input_projection(x_norm)  # [batch_size, seq_len, hidden_dim]
        
        # Self-attention
        q = self.query(h)  # [batch_size, seq_len, hidden_dim]
        k = self.key(h)    # [batch_size, seq_len, hidden_dim]
        v = self.value(h)  # [batch_size, seq_len, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Residual connection and normalization
        h = self.norm1(h + context)
        
        # Extract the last time step features after attention
        h_last = h[:, -1, :]
        
        # Apply hidden layers
        h_out = h_last
        for layer in self.layers:
            h_out = layer(h_out) + h_out  # Residual connection
        
        # Normalization
        h_out = self.norm2(h_out)
        
        # Apply output projection
        output = self.output_projection(h_out)
        
        return output