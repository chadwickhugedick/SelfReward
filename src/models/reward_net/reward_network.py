import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..feature_extraction.model_factory import FeatureExtractionModelFactory

class RewardNetwork(nn.Module):
    """
    Reward Network for the SRDDQN model.
    Uses a feature extraction model to predict rewards from state-action pairs.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, seq_len=10, model_type='TimesNet', num_layers=2, dropout=0.1):
        """
        Initialize the reward network.
        
        Args:
            state_dim (int): Dimension of the state
            action_dim (int): Dimension of the action (typically 1 for discrete actions)
            hidden_dim (int): Hidden dimension
            seq_len (int): Sequence length for time series models
            model_type (str): Type of feature extraction model
            num_layers (int): Number of layers in the feature extraction model
            dropout (float): Dropout rate
        """
        super(RewardNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Input dimension is state_dim + action_dim (one-hot encoded action)
        input_dim = state_dim + action_dim
        
        # Create feature extraction model
        self.feature_extractor = FeatureExtractionModelFactory.create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Output layer for reward prediction
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single reward value
        )
    
    def forward(self, state_sequence, action):
        """
        Forward pass through the reward network.
        
        Args:
            state_sequence (torch.Tensor): Sequence of states [batch_size, seq_len, state_dim]
            action (torch.Tensor): Action taken [batch_size]
            
        Returns:
            torch.Tensor: Predicted reward [batch_size, 1]
        """
        batch_size = state_sequence.shape[0]
        
        # One-hot encode the action
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        # Expand action to match sequence length
        action_sequence = action_one_hot.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Concatenate state and action
        state_action_sequence = torch.cat([state_sequence, action_sequence], dim=2)
        
        # Extract features
        features = self.feature_extractor(state_action_sequence)
        
        # Predict reward
        reward = self.reward_head(features)
        
        return reward
    
    def get_reward(self, state_sequence, action):
        """
        Get the predicted reward for a state-action pair.
        
        Args:
            state_sequence (torch.Tensor): Sequence of states [batch_size, seq_len, state_dim]
            action (torch.Tensor): Action taken [batch_size]
            
        Returns:
            torch.Tensor: Predicted reward [batch_size]
        """
        with torch.no_grad():
            reward = self.forward(state_sequence, action)
        return reward.squeeze()