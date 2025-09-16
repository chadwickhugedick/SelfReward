import torch
import torch.nn as nn
from .timesnet import TimesNet
from .wftnet import WFTNet
from .nlinear import NLinear, NLinearWithAttention

class FeatureExtractionModelFactory:
    """
    Factory class for creating feature extraction models.
    Supports TimesNet, WFTNet, NLinear, and NLinearWithAttention models.
    """
    
    @staticmethod
    def create_model(model_type, input_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.1):
        """
        Create a feature extraction model based on the specified type.
        
        Args:
            model_type (str): Type of model to create ('TimesNet', 'WFTNet', 'NLinear', 'NLinearAttention')
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            seq_len (int): Sequence length
            num_layers (int): Number of layers
            dropout (float): Dropout rate
            
        Returns:
            nn.Module: Feature extraction model
        """
        if model_type.lower() == 'timesnet':
            return TimesNet(input_dim, hidden_dim, output_dim, seq_len, num_layers, dropout)
        elif model_type.lower() == 'wftnet':
            return WFTNet(input_dim, hidden_dim, output_dim, seq_len, num_layers, dropout)
        elif model_type.lower() == 'nlinear':
            return NLinear(input_dim, hidden_dim, output_dim, seq_len, num_layers, dropout)
        elif model_type.lower() == 'nlinearattention':
            return NLinearWithAttention(input_dim, hidden_dim, output_dim, seq_len, num_layers, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: TimesNet, WFTNet, NLinear, NLinearAttention")