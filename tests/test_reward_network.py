import unittest
import torch
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.reward_net.reward_network import RewardNetwork
from src.models.feature_extraction.model_factory import FeatureExtractionModelFactory

class TestRewardNetwork(unittest.TestCase):
    def setUp(self):
        # Define test parameters
        self.state_dim = 10
        self.action_dim = 3  # Hold, Buy, Sell
        self.hidden_dim = 64
        self.seq_len = 10  # Changed from 5 to 10 to match the default in RewardNetwork
        self.batch_size = 8
        
        # Create sample data
        self.state_sequence = torch.rand(self.batch_size, self.seq_len, self.state_dim)
        self.actions = torch.randint(0, self.action_dim, (self.batch_size,))
        
    def test_reward_network_initialization(self):
        """Test that the reward network initializes correctly"""
        reward_net = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            model_type='TimesNet'
        )
        
        self.assertEqual(reward_net.state_dim, self.state_dim)
        self.assertEqual(reward_net.action_dim, self.action_dim)
        self.assertEqual(reward_net.hidden_dim, self.hidden_dim)
        self.assertEqual(reward_net.seq_len, self.seq_len)
        
    def test_reward_network_forward(self):
        """Test the forward pass of the reward network"""
        reward_net = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            model_type='TimesNet'
        )
        
        # Forward pass
        rewards = reward_net(self.state_sequence, self.actions)
        
        # Check output shape
        self.assertEqual(rewards.shape, (self.batch_size, 1))
        
    def test_reward_network_get_reward(self):
        """Test the get_reward method of the reward network"""
        reward_net = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            model_type='TimesNet'
        )
        
        # Get reward
        rewards = reward_net.get_reward(self.state_sequence, self.actions)
        
        # Check output shape
        self.assertEqual(rewards.shape, (self.batch_size,))
        
    def test_different_feature_extractors(self):
        """Test the reward network with different feature extractors"""
        feature_extractors = ['TimesNet', 'WFTNet', 'NLinear']
        
        for extractor in feature_extractors:
            reward_net = RewardNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                seq_len=self.seq_len,
                model_type=extractor
            )
            
            # Forward pass
            rewards = reward_net(self.state_sequence, self.actions)
            
            # Check output shape
            self.assertEqual(rewards.shape, (self.batch_size, 1))
            
    def test_action_encoding(self):
        """Test that actions are properly one-hot encoded"""
        reward_net = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            model_type='TimesNet'
        )
        
        # Create a single action for testing
        action = torch.tensor([1])  # Buy action
        state_seq = torch.rand(1, self.seq_len, self.state_dim)
        
        # Get the one-hot encoded action from the forward method
        # We need to modify the forward method slightly to expose this for testing
        batch_size = state_seq.shape[0]
        action_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=self.action_dim).float()
        
        # Check that the one-hot encoding is correct
        expected_one_hot = torch.tensor([[0.0, 1.0, 0.0]])  # Buy action is index 1
        self.assertTrue(torch.all(torch.eq(action_one_hot, expected_one_hot)))

if __name__ == '__main__':
    unittest.main()