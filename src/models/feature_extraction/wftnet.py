import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveletTransform(nn.Module):
    """Wavelet Transform module for WFTNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(WaveletTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Low-pass and high-pass filters
        self.low_pass = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.high_pass = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Initialize filters with wavelet-like patterns
        self._init_filters()
    
    def _init_filters(self):
        """Initialize filters with wavelet-like patterns"""
        # Haar wavelet-inspired initialization
        with torch.no_grad():
            # Low-pass filter (averaging)
            nn.init.constant_(self.low_pass.weight, 1.0 / self.kernel_size)
            nn.init.zeros_(self.low_pass.bias)
            
            # High-pass filter (difference)
            high_pass_init = torch.zeros_like(self.high_pass.weight)
            high_pass_init[:, :, :self.kernel_size//2] = -1.0 / (self.kernel_size // 2)
            high_pass_init[:, :, self.kernel_size//2:] = 1.0 / (self.kernel_size - self.kernel_size//2)
            self.high_pass.weight.data.copy_(high_pass_init)
            nn.init.zeros_(self.high_pass.bias)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, in_channels]
        # Transpose for 1D convolution: [batch_size, in_channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply filters
        low = self.low_pass(x)  # Approximation coefficients
        high = self.high_pass(x)  # Detail coefficients
        
        # Downsample by factor of 2
        low = low[:, :, ::2]
        high = high[:, :, ::2]
        
        # Transpose back: [batch_size, seq_len//2, out_channels]
        low = low.transpose(1, 2)
        high = high.transpose(1, 2)
        
        return low, high

class InverseWaveletTransform(nn.Module):
    """Inverse Wavelet Transform module for WFTNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(InverseWaveletTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Reconstruction filters
        self.low_recon = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.high_recon = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Initialize filters
        self._init_filters()
    
    def _init_filters(self):
        """Initialize reconstruction filters"""
        with torch.no_grad():
            nn.init.constant_(self.low_recon.weight, 1.0 / self.kernel_size)
            nn.init.zeros_(self.low_recon.bias)
            
            nn.init.constant_(self.high_recon.weight, 1.0 / self.kernel_size)
            nn.init.zeros_(self.high_recon.bias)
    
    def forward(self, low, high):
        # low, high shape: [batch_size, seq_len, in_channels]
        batch_size, seq_len, _ = low.shape
        
        # Transpose for 1D convolution
        low = low.transpose(1, 2)  # [batch_size, in_channels, seq_len]
        high = high.transpose(1, 2)  # [batch_size, in_channels, seq_len]
        
        # Upsample by factor of 2
        low_up = torch.zeros(batch_size, self.in_channels, seq_len*2, device=low.device)
        high_up = torch.zeros(batch_size, self.in_channels, seq_len*2, device=high.device)
        
        low_up[:, :, ::2] = low
        high_up[:, :, ::2] = high
        
        # Apply reconstruction filters
        low_recon = self.low_recon(low_up)
        high_recon = self.high_recon(high_up)
        
        # Combine
        x = low_recon + high_recon
        
        # Transpose back
        x = x.transpose(1, 2)  # [batch_size, seq_len*2, out_channels]
        
        return x

class WaveletAttention(nn.Module):
    """Wavelet-based attention mechanism"""
    def __init__(self, d_model, n_heads=8):
        super(WaveletAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out(context)
        
        return output

class WaveletBlock(nn.Module):
    """Wavelet Block for WFTNet"""
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super(WaveletBlock, self).__init__()
        self.wavelet = WaveletTransform(in_channels, hidden_dim, kernel_size)
        self.inverse_wavelet = InverseWaveletTransform(hidden_dim, in_channels, kernel_size)
        self.attn_low = WaveletAttention(hidden_dim)
        self.attn_high = WaveletAttention(hidden_dim)
        self.norm1_low = nn.LayerNorm(hidden_dim)
        self.norm1_high = nn.LayerNorm(hidden_dim)
        self.norm2_low = nn.LayerNorm(hidden_dim)
        self.norm2_high = nn.LayerNorm(hidden_dim)
        self.ffn_low = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        # Wavelet decomposition
        low, high = self.wavelet(x)
        
        # Process low frequency components
        low_attn = self.attn_low(low)
        low = low + low_attn
        low = self.norm1_low(low)
        low_ffn = self.ffn_low(low)
        low = low + low_ffn
        low = self.norm2_low(low)
        
        # Process high frequency components
        high_attn = self.attn_high(high)
        high = high + high_attn
        high = self.norm1_high(high)
        high_ffn = self.ffn_high(high)
        high = high + high_ffn
        high = self.norm2_high(high)
        
        # Wavelet reconstruction
        output = self.inverse_wavelet(low, high)
        
        return output

class WFTNet(nn.Module):
    """Wavelet Fourier Transform Network for time series feature extraction"""
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.1):
        super(WFTNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = self._positional_encoding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Wavelet blocks
        self.wavelet_blocks = nn.ModuleList([
            WaveletBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def _positional_encoding(self, seq_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # Input embedding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x)
        
        # Apply wavelet blocks
        for block in self.wavelet_blocks:
            x = x + block(x)  # Residual connection
        
        # Output projection (use the last time step)
        output = self.projection(x[:, -1, :])
        
        return output