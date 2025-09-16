import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class PositionalEmbedding(nn.Module):
    """Positional embedding for TimesNet"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Create positional encoding
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """Token embedding for TimesNet"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    """Fixed embedding for TimesNet"""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEmbedding(nn.Module):
    """Time feature embedding for TimesNet"""
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(2, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    """Data embedding for TimesNet"""
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Inception_Block(nn.Module):
    """Inception block for TimesNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Inception_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
        self.activation = nn.ReLU()

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x))
        x3 = self.activation(self.conv3(x))
        x4 = self.activation(self.conv4(x))
        return x1 + x2 + x3 + x4

class TimesBlock(nn.Module):
    """Times block for TimesNet"""
    def __init__(self, d_model, seq_len, factor=5):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.factor = factor
        self.d_model = d_model
        self.inception = nn.Sequential(
            Inception_Block(d_model, d_model, kernel_size=3, stride=1, padding=1),
            Inception_Block(d_model, d_model, kernel_size=5, stride=1, padding=2),
            Inception_Block(d_model, d_model, kernel_size=7, stride=1, padding=3),
            Inception_Block(d_model, d_model, kernel_size=11, stride=1, padding=5),
            Inception_Block(d_model, d_model, kernel_size=21, stride=1, padding=10),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        x_backup = x
        x_periods = []
        for i in range(1, self.factor + 1):
            if seq_len % i == 0:
                period = i
                x_period = x.reshape(batch_size, period, seq_len // period, d_model)
                x_period = x_period.permute(0, 3, 1, 2).contiguous()
                x_period = self.inception(x_period)
                x_period = x_period.permute(0, 2, 3, 1).reshape(batch_size, seq_len, d_model)
                x_periods.append(x_period)
        if x_periods:
            x = x + torch.sum(torch.stack(x_periods), dim=0)
        x = self.norm(x)
        return x

class TimesNet(nn.Module):
    """TimesNet model for time series feature extraction"""
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.1):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Data embedding
        self.embedding = DataEmbedding(input_dim, hidden_dim, dropout)
        
        # TimesBlock layers
        self.model = nn.ModuleList([
            TimesBlock(hidden_dim, seq_len) for _ in range(num_layers)
        ])
        
        # Output layer
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.embedding(x)
        
        # Apply TimesBlock layers
        for layer in self.model:
            x = layer(x)
        
        # Output projection
        output = self.projection(x[:, -1, :])  # Take the last time step
        
        return output