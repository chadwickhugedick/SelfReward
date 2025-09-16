import unittest
import numpy as np
import sys
import os
import torch

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.agents.double_dqn import ReplayBuffer

class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 1000
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Create sample data with different dimensions to test flexibility
        self.state_dim = 10
        self.state = np.random.rand(self.state_dim)
        self.action = 1
        self.reward = 0.5
        self.next_state = np.random.rand(self.state_dim)
        self.done = False
    
    def test_initialization(self):
        """Test if the replay buffer initializes correctly."""
        self.assertEqual(len(self.replay_buffer), 0)
        self.assertEqual(self.replay_buffer.buffer.maxlen, self.buffer_size)
    
    def test_add_experience(self):
        """Test adding an experience to the buffer."""
        initial_length = len(self.replay_buffer)
        self.replay_buffer.add(self.state, self.action, self.reward, self.next_state, self.done)
        
        # Check if length increased
        self.assertEqual(len(self.replay_buffer), initial_length + 1)
        
        # Check if the experience was added correctly
        experience = self.replay_buffer.buffer[-1]
        np.testing.assert_array_equal(experience[0], self.state)
        self.assertEqual(experience[1], self.action)
        self.assertEqual(experience[2], self.reward)
        np.testing.assert_array_equal(experience[3], self.next_state)
        self.assertEqual(experience[4], self.done)
    
    def test_sample_batch(self):
        """Test sampling a batch from the buffer."""
        # Add multiple experiences
        for _ in range(10):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = bool(np.random.randint(0, 2))
            
            self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Sample a batch
        batch_size = 5
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Check shapes
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
        
        # Check types
        for state in states:
            self.assertEqual(len(state), self.state_dim)
    
    def test_buffer_capacity(self):
        """Test that the buffer respects its capacity."""
        # Fill the buffer beyond capacity
        for i in range(self.buffer_size + 10):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = bool(np.random.randint(0, 2))
            
            self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Check that the buffer size is capped at capacity
        self.assertEqual(len(self.replay_buffer), self.buffer_size)
    
    def test_sample_with_insufficient_data(self):
        """Test sampling when buffer has less data than batch size."""
        # Add only a few experiences
        for _ in range(3):
            self.replay_buffer.add(self.state, self.action, self.reward, self.next_state, self.done)
        
        # Try to sample more than available
        batch_size = 5
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Should only return as many as available
        self.assertEqual(len(states), 3)
        self.assertEqual(len(actions), 3)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(len(next_states), 3)
        self.assertEqual(len(dones), 3)
    
    def test_different_reward_types(self):
        """Test the buffer with different reward types (float, dict)."""
        # Test with float reward
        self.replay_buffer.add(self.state, self.action, 0.5, self.next_state, self.done)
        
        # Test with dictionary reward
        reward_dict = {'min_max': 0.7, 'sharpe': 0.3}
        self.replay_buffer.add(self.state, self.action, reward_dict, self.next_state, self.done)
        
        # Sample and check
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(2)
        self.assertEqual(len(rewards), 2)
        
        # One should be float, one should be dict
        reward_types = set(type(r) for r in rewards)
        self.assertTrue(float in reward_types or int in reward_types or np.float64 in reward_types)
        self.assertTrue(dict in reward_types)

if __name__ == '__main__':
    unittest.main()