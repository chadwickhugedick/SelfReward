import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agents.srddqn import SRDDQNAgent
from src.models.reward_net.reward_network import RewardNetwork

class TestSelfRewarding(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Define dimensions
        self.state_dim = 10
        self.action_dim = 3
        self.seq_len = 10
        self.hidden_dim = 64
        
        # Create a sample state sequence
        self.state_sequence = np.random.rand(self.seq_len, self.state_dim).astype(np.float32)
        
        # Create a sample action
        self.action = 1  # Buy action
        
        # Create expert reward dictionary
        self.expert_reward_dict = {
            'Min-Max': 0.5,
            'Sharpe': 0.3,
            'Returns': 0.4
        }
        
        # Create SRDDQN agent
        self.agent = SRDDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            reward_model_type='TimesNet',
            reward_labels=['Min-Max'],
            device='cpu'
        )
        
        # Create reward network directly for testing
        self.reward_network = RewardNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            seq_len=self.seq_len,
            model_type='TimesNet'
        )
    
    def test_reward_network_initialization(self):
        """Test that the reward network initializes correctly"""
        self.assertEqual(self.reward_network.state_dim, self.state_dim)
        self.assertEqual(self.reward_network.action_dim, self.action_dim)
        self.assertEqual(self.reward_network.hidden_dim, self.hidden_dim)
        self.assertEqual(self.reward_network.seq_len, self.seq_len)
    
    def test_reward_network_forward(self):
        """Test the forward pass of the reward network"""
        # Convert state sequence and action to tensors
        state_sequence_tensor = torch.FloatTensor(self.state_sequence).unsqueeze(0)  # Add batch dimension
        action_tensor = torch.LongTensor([self.action])
        
        # Forward pass
        reward = self.reward_network(state_sequence_tensor, action_tensor)
        
        # Check output shape
        self.assertEqual(reward.shape, (1, 1))
        
        # Check output is a float tensor
        self.assertTrue(torch.is_tensor(reward))
        self.assertEqual(reward.dtype, torch.float32)
    
    def test_compute_reward_mechanism(self):
        """Test the self-rewarding mechanism"""
        # Get the self-reward prediction
        state_sequence_tensor = torch.FloatTensor(self.state_sequence).unsqueeze(0)
        action_tensor = torch.LongTensor([self.action])
        self_reward = self.reward_network(state_sequence_tensor, action_tensor).item()
        
        # Test with self-reward < expert-reward
        # Mock the reward network to return a lower value
        original_forward = self.agent.reward_network.forward
        self.agent.reward_network.forward = lambda x, y: torch.tensor([[0.2]])
        
        reward = self.agent.compute_reward([self.state_sequence], [self.action], [self.expert_reward_dict])[0]
        
        # The reward should be the expert reward (0.5) since it's higher
        self.assertEqual(reward, 0.5)
        
        # Test with self-reward > expert-reward
        # Mock the reward network to return a higher value
        self.agent.reward_network.forward = lambda x, y: torch.tensor([[0.7]])
        
        reward = self.agent.compute_reward([self.state_sequence], [self.action], [self.expert_reward_dict])[0]
        
        # The reward should be the self reward (0.7) since it's higher
        # Use assertAlmostEqual for floating point comparison
        self.assertAlmostEqual(reward, 0.7, places=5)
        
        # Restore original forward method
        self.agent.reward_network.forward = original_forward
    
    def test_train_reward_network(self):
        """Test training the reward network"""
        # Create batch data
        batch_size = 4
        state_sequences = np.random.rand(batch_size, self.seq_len, self.state_dim).astype(np.float32)
        actions = [0, 1, 2, 1]  # Hold, Buy, Sell, Buy
        expert_rewards = [0.1, 0.5, -0.2, 0.3]
        
        # Get initial parameter values before training
        initial_params = []
        for param in self.agent.reward_network.parameters():
            if param.requires_grad:
                initial_params.append(param.clone().detach())
        
        # Train the reward network
        loss = self.agent.train_reward_network(state_sequences, actions, expert_rewards)
        
        # Check that loss is a float
        self.assertIsInstance(loss, float)
        
        # Check that parameters have been updated
        updated = False
        i = 0
        for param in self.agent.reward_network.parameters():
            if param.requires_grad:
                if not torch.allclose(param, initial_params[i]):
                    updated = True
                    break
                i += 1
        
        self.assertTrue(updated, "Reward network parameters should be updated after training")
    
    def test_reward_selection_in_training(self):
        """Test that the agent selects the higher reward during training"""
        # Test the compute_reward method directly
        state = np.random.rand(self.seq_len, self.state_dim).astype(np.float32)
        action = 1  # Buy action
        
        # Create expert reward dictionary with different values
        expert_reward_dict = {
            'Min-Max': 0.3,  # Lower than self-reward
            'Sharpe': 0.2,
            'Returns': 0.1
        }
        
        # Mock the reward network to return a higher value
        original_forward = self.agent.reward_network.forward
        self.agent.reward_network.forward = lambda x, y: torch.tensor([[0.8]])
        
        # Compute reward using the self-rewarding mechanism
        reward = self.agent.compute_reward([state], [action], [expert_reward_dict])[0]
        
        # The reward should be the self-reward (0.8) since it's higher than expert reward (0.3)
        self.assertAlmostEqual(reward, 0.8, places=5)
        
        # Now test with expert reward higher than self-reward
        expert_reward_dict = {
            'Min-Max': 0.9,  # Higher than self-reward
            'Sharpe': 0.2,
            'Returns': 0.1
        }
        
        # Compute reward again
        reward = self.agent.compute_reward([state], [action], [expert_reward_dict])[0]
        
        # The reward should be the expert reward (0.9) since it's higher than self-reward (0.8)
        self.assertEqual(reward, 0.9)
        
        # Restore original forward method
        self.agent.reward_network.forward = original_forward

    def test_different_reward_labels(self):
        """Test that the agent can use different reward labels"""
        # Create a new agent with a different reward label
        agent = SRDDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            reward_model_type='TimesNet',
            reward_labels=['Sharpe'],  # Use Sharpe instead of Min-Max
            device='cpu'
        )
        
        # Create expert reward dictionary
        expert_reward_dict = {
            'Min-Max': 0.5,
            'Sharpe': 0.7,  # Higher than Min-Max
            'Returns': 0.4
        }
        
        # Mock the reward network to return a specific value
        original_forward = agent.reward_network.forward
        agent.reward_network.forward = lambda x, y: torch.tensor([[0.6]])
        
        # Compute reward using the self-rewarding mechanism
        reward = agent.compute_reward([self.state_sequence], [self.action], [expert_reward_dict])[0]
        
        # The reward should be the expert reward for Sharpe (0.7) since it's higher than self-reward (0.6)
        self.assertEqual(reward, 0.7)
        
        # Restore original forward method
        agent.reward_network.forward = original_forward
    
    def test_integration_with_training(self):
        """Test the integration of self-rewarding with the training process"""
        # Add some experiences to the replay buffer with dictionary rewards
        for _ in range(20):
            state = np.random.rand(self.state_dim).astype(np.float32)
            action = np.random.randint(0, self.action_dim)
            # Use a dictionary reward
            reward = {
                'Min-Max': np.random.uniform(-0.5, 1.0),
                'Sharpe': np.random.uniform(-0.3, 0.8),
                'Returns': np.random.uniform(-0.4, 0.9)
            }
            next_state = np.random.rand(self.state_dim).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            
            self.agent.dqn_agent.replay_buffer.add(state, action, reward, next_state, done)
        
        # Train the agent
        dqn_loss, reward_loss = self.agent.train()
        
        # Check that losses are returned
        self.assertIsInstance(dqn_loss, float)
        self.assertIsInstance(reward_loss, float)
    
    def test_state_sequence_buffer(self):
        """Test the state sequence buffer functionality"""
        # Initially the buffer should be empty
        self.assertEqual(len(self.agent.state_sequence_buffer), 0)
        
        # Add states to the buffer
        for i in range(self.seq_len + 5):  # Add more than seq_len states
            state = np.random.rand(self.state_dim).astype(np.float32)
            self.agent.select_action(state, training=True)
        
        # Buffer should maintain seq_len size
        self.assertEqual(len(self.agent.state_sequence_buffer), self.seq_len)
        
        # Get state sequence
        state_sequence = self.agent.get_state_sequence()
        
        # Check shape
        self.assertEqual(state_sequence.shape, (self.seq_len, self.state_dim))

if __name__ == '__main__':
    unittest.main()