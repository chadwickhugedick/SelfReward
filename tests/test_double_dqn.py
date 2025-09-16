import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.agents.double_dqn import QNetwork, ReplayBuffer, DoubleDQNAgent

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.state_dim = 10
        self.action_dim = 3
        self.hidden_dim = 64
        self.batch_size = 5
        self.device = 'cpu'
        
        # Create a Q-Network
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create sample data
        self.sample_state = torch.randn(self.batch_size, self.state_dim)
    
    def test_initialization(self):
        """Test if the Q-Network initializes correctly."""
        self.assertEqual(self.q_network.state_dim, self.state_dim)
        self.assertEqual(self.q_network.action_dim, self.action_dim)
        self.assertEqual(self.q_network.hidden_dim, self.hidden_dim)
    
    def test_forward_pass(self):
        """Test if the forward pass produces the expected output shape."""
        output = self.q_network(self.sample_state)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.action_dim))
        
        # Check output is not all zeros or NaNs
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.all(output == 0))

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Create sample data
        self.state = np.random.rand(10)
        self.action = 1
        self.reward = 0.5
        self.next_state = np.random.rand(10)
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
    
    def test_sample_batch(self):
        """Test sampling a batch from the buffer."""
        # Add multiple experiences
        for _ in range(10):
            state = np.random.rand(10)
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            next_state = np.random.rand(10)
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

class TestDoubleDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state_dim = 10
        self.action_dim = 3
        self.hidden_dim = 64
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.buffer_size = 1000
        self.batch_size = 32
        self.device = 'cpu'
        
        # Create agent
        self.agent = DoubleDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
            target_update=self.target_update,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device
        )
        
        # Create sample data
        self.sample_state = np.random.rand(self.state_dim)
    
    def test_initialization(self):
        """Test if the agent initializes correctly."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.hidden_dim, self.hidden_dim)
        self.assertEqual(self.agent.learning_rate, self.learning_rate)
        self.assertEqual(self.agent.gamma, self.gamma)
        self.assertEqual(self.agent.epsilon, self.epsilon_start)
        self.assertEqual(self.agent.epsilon_end, self.epsilon_end)
        self.assertEqual(self.agent.epsilon_decay, self.epsilon_decay)
        self.assertEqual(self.agent.target_update, self.target_update)
        self.assertEqual(self.agent.batch_size, self.batch_size)
        self.assertEqual(self.agent.device, self.device)
        
        # Check if networks are created
        self.assertIsInstance(self.agent.policy_net, QNetwork)
        self.assertIsInstance(self.agent.target_net, QNetwork)
        
        # Check if replay buffer is created
        self.assertIsInstance(self.agent.replay_buffer, ReplayBuffer)
    
    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        # Force exploration by setting epsilon to 1.0
        self.agent.epsilon = 1.0
        
        # Select action multiple times
        actions = [self.agent.select_action(self.sample_state) for _ in range(100)]
        
        # Check if actions are within valid range
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_dim)
        
        # Check if there's variety in actions (exploration)
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)  # Should have more than one unique action
    
    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        # Force exploitation by setting epsilon to 0.0
        self.agent.epsilon = 0.0
        
        # Select action
        action = self.agent.select_action(self.sample_state)
        
        # Check if action is within valid range
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_update_epsilon(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        self.agent.update_epsilon()
        
        # Check if epsilon decreased
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
        # Check if epsilon doesn't go below epsilon_end
        self.agent.epsilon = self.agent.epsilon_end / 2
        self.agent.update_epsilon()
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_update_target_network(self):
        """Test target network update."""
        # Modify policy network weights
        for param in self.agent.policy_net.parameters():
            param.data = param.data + 0.1
        
        # Check that networks are different before update
        policy_params = list(self.agent.policy_net.parameters())
        target_params = list(self.agent.target_net.parameters())
        
        # At least one parameter should be different
        params_equal = all(torch.all(p1 == p2) for p1, p2 in zip(policy_params, target_params))
        self.assertFalse(params_equal)
        
        # Update target network
        self.agent.update_target_network()
        
        # Check that networks are identical after update
        policy_params = list(self.agent.policy_net.parameters())
        target_params = list(self.agent.target_net.parameters())
        
        # All parameters should be equal
        params_equal = all(torch.all(p1 == p2) for p1, p2 in zip(policy_params, target_params))
        self.assertTrue(params_equal)

if __name__ == '__main__':
    unittest.main()