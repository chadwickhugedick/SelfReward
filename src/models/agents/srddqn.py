import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from .double_dqn import DoubleDQNAgent, ReplayBuffer
from ..reward_net.reward_network import RewardNetwork

class SRDDQNAgent:
    """
    Self-Rewarding Double DQN (SRDDQN) agent.
    Combines Double DQN with a self-rewarding mechanism.
    """
    
    def __init__(self, state_dim, action_dim, seq_len=10, hidden_dim=128,
                 dqn_lr=0.0001, reward_net_lr=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update=200, tau=0.005, buffer_size=10000, batch_size=64,
                 reward_model_type='TimesNet', reward_labels=['Min-Max'],
                 sync_steps=1, update_steps=1,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 feature_dim=None):
        """
        Initialize the SRDDQN agent.
        
        Args:
            state_dim (int): Dimension of the state
            action_dim (int): Dimension of the action space
            seq_len (int): Sequence length for the reward network
            hidden_dim (int): Hidden dimension
            dqn_lr (float): Learning rate for the DQN
            reward_net_lr (float): Learning rate for the reward network
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Decay rate for epsilon
            target_update (int): Update target network every N steps
            tau (float): Soft update coefficient for target network
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            reward_model_type (str): Type of model for the reward network
            reward_labels (list): List of reward labels to use
            sync_steps (int): Synchronization steps for reward network
            update_steps (int): Update steps for reward network
            device (str): Device to use for training
        """
        self.state_dim = state_dim  # Flattened state dimension
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.reward_labels = reward_labels
        self.sync_steps = sync_steps
        self.update_steps = update_steps
        self.device = device

        # Feature dimension per time step
        self.feature_dim = feature_dim if feature_dim is not None else state_dim // seq_len
        
        # Create Double DQN agent
        self.dqn_agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=dqn_lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device
        )
        
        # Create reward network
        self.reward_network = RewardNetwork(
            state_dim=self.feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            model_type=reward_model_type,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        # Create optimizer for reward network
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(), lr=reward_net_lr)
        
        # Create replay buffer for state sequences
        self.state_sequence_buffer = deque(maxlen=seq_len)
        
        # Initialize step counters
        self.steps_done = 0
        self.reward_train_steps = 0
    
    def select_action(self, state, training=True):
        """
        Select an action using the DQN agent.
        
        Args:
            state (np.ndarray): Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        # Add state to sequence buffer
        if len(self.state_sequence_buffer) == self.seq_len:
            self.state_sequence_buffer.popleft()
        self.state_sequence_buffer.append(state)
        
        # If we don't have enough states yet, take a random action
        if len(self.state_sequence_buffer) < self.seq_len and training:
            return random.randrange(self.action_dim)
        
        # Otherwise, use the DQN agent to select an action
        return self.dqn_agent.select_action(state, training)
    
    def get_state_sequence(self):
        """
        Get the current state sequence.
        
        Returns:
            np.ndarray: State sequence
        """
        # If we don't have enough states yet, pad with zeros
        if len(self.state_sequence_buffer) < self.seq_len:
            padding = [np.zeros_like(self.state_sequence_buffer[0]) for _ in range(self.seq_len - len(self.state_sequence_buffer))]
            return np.array(list(padding) + list(self.state_sequence_buffer))
        else:
            return np.array(list(self.state_sequence_buffer))
    
    def compute_reward(self, state_sequences, actions, expert_reward_dicts):
        """
        Compute the reward using the self-rewarding mechanism for a batch.
        
        Args:
            state_sequences (list of np.ndarray): List of state sequences
            actions (list of int): List of actions taken
            expert_reward_dicts (list of dict): List of dictionaries of expert-defined rewards
            
        Returns:
            list of float: List of computed rewards
        """
        # Convert to tensors
        state_sequences_tensor = torch.FloatTensor(np.array(state_sequences)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Get self-rewarding network predictions
        self_rewards_tensor = self.reward_network(state_sequences_tensor, actions_tensor).squeeze()
        if self_rewards_tensor.dim() == 0:
            self_rewards = [self_rewards_tensor.item()]
        else:
            self_rewards = self_rewards_tensor.tolist()
        
        # Get expert rewards for the selected label
        expert_rewards = [d[self.reward_labels[0]] for d in expert_reward_dicts]
        
        # Return the maximum of self-reward and expert-reward for each
        return [max(sr, er) for sr, er in zip(self_rewards, expert_rewards)]
    
    def train_reward_network(self, state_sequences, actions, expert_rewards):
        """
        Train the reward network using expert-defined rewards.
        
        Args:
            state_sequences (np.ndarray): Batch of state sequences
            actions (list): Batch of actions
            expert_rewards (list): Batch of expert-defined rewards
            
        Returns:
            float: Loss value
        """
        # Convert to tensors
        state_sequences_tensor = torch.FloatTensor(np.array(state_sequences)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        expert_rewards_tensor = torch.FloatTensor(expert_rewards).unsqueeze(1).to(self.device)
        
        # Forward pass
        predicted_rewards = self.reward_network(state_sequences_tensor, actions_tensor)
        
        # Compute loss
        loss = F.mse_loss(predicted_rewards, expert_rewards_tensor)
        
        # Optimize the model
        self.reward_optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_value_(self.reward_network.parameters(), -1, 1)
        self.reward_optimizer.step()
        
        return loss.item()
    
    def train(self):
        """
        Train the SRDDQN agent.
        
        Returns:
            tuple: (dqn_loss, reward_loss)
        """
        if len(self.dqn_agent.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.dqn_agent.replay_buffer.sample(self.batch_size)
        
        # Train the DQN agent
        dqn_loss = self.dqn_agent.train()
        
        # Train the reward network every sync_steps
        reward_loss = 0.0
        if self.steps_done % self.sync_steps == 0 and self.reward_train_steps < self.update_steps:
            # Extract expert rewards for the selected label
            expert_rewards = [r[self.reward_labels[0]] if isinstance(r, dict) else r for r in rewards]
            
            # Create state sequences for each experience
            state_sequences = np.array(states).reshape(len(states), self.seq_len, self.feature_dim)
            
            # Train the reward network
            reward_loss = self.train_reward_network(state_sequences, actions, expert_rewards)
            self.reward_train_steps += 1
        
        # Reset reward train steps counter if needed
        if self.reward_train_steps >= self.update_steps:
            self.reward_train_steps = 0
        
        # Increment step counter
        self.steps_done += 1
        
        return dqn_loss, reward_loss
    
    def save(self, path):
        """
        Save the agent's models.
        
        Args:
            path (str): Path to save the models
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save DQN agent
        self.dqn_agent.save(f"{path}_dqn.pth")
        
        # Save reward network
        torch.save({
            'reward_network': self.reward_network.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'reward_train_steps': self.reward_train_steps
        }, f"{path}_reward.pth")
    
    def load(self, path):
        """
        Load the agent's models.
        
        Args:
            path (str): Path to load the models from
        """
        # Load DQN agent
        self.dqn_agent.load(f"{path}_dqn.pth")
        
        # Load reward network
        checkpoint = torch.load(f"{path}_reward.pth", map_location=self.device)
        self.reward_network.load_state_dict(checkpoint['reward_network'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.reward_train_steps = checkpoint['reward_train_steps']