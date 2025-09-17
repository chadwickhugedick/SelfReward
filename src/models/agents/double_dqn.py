import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Q-Network for the Double DQN agent.
    Maps state to action values.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim (int): Dimension of the state
            action_dim (int): Dimension of the action space
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of hidden layers
            dropout (float): Dropout rate
        """
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input layer
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through the Q-Network.

        Args:
            state (torch.Tensor): State tensor [batch_size, state_dim] or [batch_size, seq_len, features]

        Returns:
            torch.Tensor: Q-values for each action [batch_size, action_dim]
        """
        # Flatten the state if it's 2D
        if state.dim() > 2:
            state = state.reshape(state.size(0), -1)
        return self.model(state)

class ReplayBuffer:
    """
    Optimized replay buffer for storing and sampling experiences efficiently.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        # Pre-allocate arrays for better memory efficiency
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._dones = None
        self._initialized = False
        # Extended storage (original, unflattened states & expert reward dicts)
        self._orig_states = [None] * capacity
        self._expert_reward_dicts = [None] * capacity
    
    def _initialize_arrays(self, state_shape):
        """Initialize pre-allocated arrays based on the first state shape."""
        # For flattened states, calculate the total size
        if len(state_shape) > 1:
            state_size = np.prod(state_shape)
        else:
            state_size = state_shape[0] if len(state_shape) > 0 else 1
            
        self._states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_states = np.zeros((self.capacity, state_size), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=bool)
        self._initialized = True
    
    def add(self, state, action, reward, next_state, done, orig_state=None, expert_reward_dict=None):
        """
        Add an experience to the buffer with optimized memory usage.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received (can be dict or scalar)
            next_state: Next state
            done: Whether the episode is done
        """
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        # Initialize arrays on first use
        if not self._initialized:
            self._initialize_arrays(state.shape)
        
        # Handle different reward formats
        if isinstance(reward, dict):
            reward_value = reward.get('Min-Max', 0.0)
        else:
            reward_value = float(reward)
        
        # Flatten states for storage
        state_flat = state.flatten()
        next_state_flat = next_state.flatten()
        
        # Store in circular buffer
        idx = self.position % self.capacity
        self._states[idx] = state_flat
        self._actions[idx] = action
        self._rewards[idx] = reward_value
        self._next_states[idx] = next_state_flat
        self._dones[idx] = done
        if orig_state is not None:
            self._orig_states[idx] = orig_state
        if expert_reward_dict is not None:
            self._expert_reward_dicts[idx] = expert_reward_dict
        
        # Keep original format for compatibility
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.position += 1
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer efficiently.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        current_size = min(len(self.buffer), self.capacity)
        if current_size == 0:
            return [], [], [], [], []
        
        # Sample random indices
        indices = np.random.choice(current_size, min(batch_size, current_size), replace=False)
        
        # Return sampled arrays
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        
        return states, actions, rewards, next_states, dones

    def sample_extended(self, batch_size):
        """Sample batch including original states & expert reward dicts for SRDDQN reward net training."""
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        # Determine indices used by last sample by re-sampling deterministic subset via matching values.
        # Simpler approach: resample indices directly again.
        current_size = min(len(self.buffer), self.capacity)
        if current_size == 0:
            return [], [], [], [], [], [], []
        indices = np.random.choice(current_size, min(batch_size, current_size), replace=False)
        orig_states = [self._orig_states[i] for i in indices]
        expert_dicts = [self._expert_reward_dicts[i] for i in indices]
        return (self._states[indices], self._actions[indices], self._rewards[indices],
                self._next_states[indices], self._dones[indices], orig_states, expert_dicts)
    
    def sample_sequential(self, batch_size):
        """
        Sample a sequential batch for improved cache efficiency.
        
        Args:
            batch_size (int): Size of the batch to sample
            
        Returns:
            tuple: Sequential batch of experiences
        """
        current_size = min(len(self.buffer), self.capacity)
        if current_size < batch_size:
            return self.sample(batch_size)
        
        # Start from a random position and take sequential samples
        start_idx = np.random.randint(0, current_size - batch_size + 1)
        indices = np.arange(start_idx, start_idx + batch_size)
        
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            int: Current size of the buffer
        """
        return min(len(self.buffer), self.capacity)

class DoubleDQNAgent:
    """
    Double DQN agent for the SRDDQN model.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=200,
                 tau=0.005, buffer_size=10000, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Double DQN agent.
        
        Args:
            state_dim (int): Dimension of the state
            action_dim (int): Dimension of the action space
            hidden_dim (int): Hidden dimension
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Decay rate for epsilon
            target_update (int): Update target network every N steps
            tau (float): Soft update coefficient for target network
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            device (str): Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Create Q-Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_dim)
        else:
            # Exploitation: best action according to policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        """
        Update the exploration rate (epsilon).
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """
        Update the target network using soft updates.
        """
        # Soft update: target_net = tau * policy_net + (1 - tau) * target_net
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def train(self, reward_network=None):
        """
        Train the agent using a batch of experiences.
        
        Args:
            reward_network: Optional reward network for self-rewarding
            
        Returns:
            float: Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values using Double DQN
        # 1. Get actions from policy network
        next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        # 2. Get Q-values from target network for those actions
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
        # 3. Compute expected Q-values
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.detach())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        # Update exploration rate
        self.update_epsilon()
        
        return loss.item()
    
    def save(self, path):
        """
        Save the agent's models.
        
        Args:
            path (str): Path to save the models
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'tau': self.tau
        }, path)
    
    def load(self, path):
        """
        Load the agent's models.
        
        Args:
            path (str): Path to load the models from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.tau = checkpoint.get('tau', 0.005)  # Default to 0.005 if not in checkpoint