import unittest
import torch
import numpy as np
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.agents.srddqn import SRDDQNAgent

class TestSRDDQNTraining(unittest.TestCase):
    """Test the SRDDQN training algorithm"""
    
    def setUp(self):
        """Set up the test environment"""
        # Define parameters
        self.state_dim = 10
        self.action_dim = 3  # Hold, Buy, Sell
        self.seq_len = 5
        self.hidden_dim = 64
        self.batch_size = 8
        
        # Create SRDDQN agent
        self.agent = SRDDQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            dqn_lr=0.001,
            reward_net_lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            target_update=5,
            buffer_size=100,
            batch_size=self.batch_size,
            reward_model_type='TimesNet',
            reward_labels=['Min-Max'],
            sync_steps=1,
            update_steps=1,
            device='cpu'
        )
        
        # Add some experiences to the replay buffer
        for _ in range(20):
            state = np.random.rand(self.state_dim).astype(np.float32)
            action = np.random.randint(0, self.action_dim)
            # Use a float reward instead of a dictionary for the DQN agent's replay buffer
            reward = np.random.uniform(-0.5, 1.0)
            next_state = np.random.rand(self.state_dim).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            
            self.agent.dqn_agent.replay_buffer.add(state, action, reward, next_state, done)
    
    def test_train_method(self):
        """Test the train method of the SRDDQN agent"""
        # Train the agent
        dqn_loss, reward_loss = self.agent.train()
        
        # Check that losses are returned
        self.assertIsInstance(dqn_loss, float)
        self.assertIsInstance(reward_loss, float)
    
    def test_training_updates_networks(self):
        """Test that training updates both networks"""
        # Get initial parameters
        initial_dqn_params = [param.clone().detach() for param in self.agent.dqn_agent.policy_net.parameters() if param.requires_grad]
        initial_reward_params = [param.clone().detach() for param in self.agent.reward_network.parameters() if param.requires_grad]
        
        # Train multiple times to ensure updates occur
        for _ in range(5):
            self.agent.train()
        
        # Check if DQN parameters have been updated
        dqn_updated = False
        for i, param in enumerate(self.agent.dqn_agent.policy_net.parameters()):
            if param.requires_grad and not torch.allclose(param, initial_dqn_params[i]):
                dqn_updated = True
                break
        
        # Check if reward network parameters have been updated
        reward_updated = False
        i = 0
        for param in self.agent.reward_network.parameters():
            if param.requires_grad:
                if not torch.allclose(param, initial_reward_params[i]):
                    reward_updated = True
                    break
                i += 1
        
        self.assertTrue(dqn_updated, "DQN parameters should be updated after training")
        self.assertTrue(reward_updated, "Reward network parameters should be updated after training")
    
    def test_epsilon_decay(self):
        """Test that epsilon decays during training"""
        # Get initial epsilon
        initial_epsilon = self.agent.dqn_agent.epsilon
        
        # Train multiple times
        for _ in range(10):
            self.agent.train()
            # Select an action to trigger epsilon decay
            state = np.random.rand(self.state_dim).astype(np.float32)
            self.agent.dqn_agent.select_action(state, training=True)
        
        # Check that epsilon has decreased
        self.assertLess(self.agent.dqn_agent.epsilon, initial_epsilon)
    
    def test_target_network_update(self):
        """Test that the target network is updated after target_update steps"""
        # Get initial target network parameters
        initial_target_params = [param.clone().detach() for param in self.agent.dqn_agent.target_net.parameters()]
        
        # Train for exactly target_update steps
        for _ in range(self.agent.dqn_agent.target_update):
            self.agent.train()
            # Select an action to increment steps_done
            state = np.random.rand(self.state_dim).astype(np.float32)
            self.agent.dqn_agent.select_action(state, training=True)
        
        # Check if target network parameters have been updated
        target_updated = False
        for i, param in enumerate(self.agent.dqn_agent.target_net.parameters()):
            if not torch.allclose(param, initial_target_params[i]):
                target_updated = True
                break
        
        self.assertTrue(target_updated, "Target network should be updated after target_update steps")
    
    def test_reward_network_sync(self):
        """Test that the reward network is trained according to sync_steps"""
        # Set a specific sync_steps value
        self.agent.sync_steps = 2
        self.agent.update_steps = 3  # Allow multiple updates
        
        # Reset reward_train_steps and steps_done to ensure we start from a known state
        self.agent.reward_train_steps = 0
        
        # First step - should not train reward network
        self.agent.steps_done = 1  # Not divisible by sync_steps
        dqn_loss1, reward_loss1 = self.agent.train()
        self.assertEqual(self.agent.reward_train_steps, 0)  # No training occurred
        
        # Second step - should train reward network
        self.agent.steps_done = 2  # Divisible by sync_steps
        dqn_loss2, reward_loss2 = self.agent.train()
        self.assertEqual(self.agent.reward_train_steps, 1)  # Training occurred
        
        # Third step - should not train reward network
        self.agent.steps_done = 3  # Not divisible by sync_steps
        dqn_loss3, reward_loss3 = self.agent.train()
        self.assertEqual(self.agent.reward_train_steps, 1)  # No change
        
        # Fourth step - should train reward network
        self.agent.steps_done = 4  # Divisible by sync_steps
        dqn_loss4, reward_loss4 = self.agent.train()
        self.assertEqual(self.agent.reward_train_steps, 2)  # Training occurred

if __name__ == '__main__':
    unittest.main()