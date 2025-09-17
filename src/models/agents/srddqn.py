import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from .double_dqn import DoubleDQNAgent
from ..reward_net.reward_network import RewardNetwork


"""SRDDQNReplayBuffer removed; unified buffer in DoubleDQN now stores extended info."""

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
                 feature_dim=None, pretrained_reward_path=None,
                 reward_net_num_layers=2, reward_net_dropout=0.1,
                 reward_update_interval=50,
                 reward_batch_multiplier=4,
                 reward_warmup_steps=500,
                 max_reward_train_per_episode=5,
                 num_envs=1):
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
            pretrained_reward_path (str): Path to pre-trained reward network
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
        self.pretrained_reward_path = pretrained_reward_path

        # Feature dimension per time step (per-timestep features inside each observation window)
        if feature_dim is not None:
            self.feature_dim = feature_dim
        else:
            # state_dim is flattened per-window (window_size * per_timestep_feature_dim)
            # We fall back to deriving per-timestep features assuming seq_len was used previously
            self.feature_dim = state_dim // seq_len
            print(f"Warning: Calculated feature_dim as {self.feature_dim} from state_dim={state_dim}//seq_len={seq_len}. " +
                  f"This should match environment observation_space.shape[1]")

        # Infer window size (number of timesteps inside a single observation window)
        # state_dim is the flattened single-observation size (window_size * per_timestep_feature_dim)
        # so window_size = state_dim // per_timestep_feature_dim
        self.window_size = int(state_dim // self.feature_dim) if self.feature_dim > 0 else 1
        # Reward network expects each sequence element to be a flattened window of length window_size * feature_dim
        self.reward_input_dim = int(self.window_size * self.feature_dim)
        
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
        
    # Unified buffer usage (extended fields stored inside dqn_agent.replay_buffer)
        
        # Create reward network
        self.reward_network = RewardNetwork(
            state_dim=self.reward_input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            model_type=reward_model_type,
            num_layers=reward_net_num_layers,
            dropout=reward_net_dropout
        ).to(device)
        
        # Create optimizer for reward network
        self.reward_optimizer = torch.optim.Adam(self.reward_network.parameters(), lr=reward_net_lr)
        
        # Load pre-trained reward network if path is provided
        if self.pretrained_reward_path and os.path.exists(self.pretrained_reward_path):
            self.load_pretrained_reward_network(self.pretrained_reward_path)
            print(f"Loaded pre-trained reward network from {self.pretrained_reward_path}")
        
        # Create replay buffer for state sequences
        # Per-environment sequence buffers to avoid cross-env contamination
        self.num_envs = num_envs
        self.state_sequence_buffers = [deque(maxlen=seq_len) for _ in range(self.num_envs)]
        # Simple cache for self-reward computation (for speed optimization)
        self.reward_cache = {}
        self.cache_max_size = 1000
        
        # Initialize step counters
        self.steps_done = 0
        self.reward_train_steps = 0
        # Reward batching scheduling parameters
        self.reward_update_interval = reward_update_interval
        self.reward_batch_multiplier = reward_batch_multiplier
        self.reward_warmup_steps = reward_warmup_steps
        self.max_reward_train_per_episode = max_reward_train_per_episode
        self.episode_reward_train_count = 0
    
    def load_pretrained_reward_network(self, path):
        """
        Load pre-trained reward network weights.
        
        Args:
            path (str): Path to the pre-trained model
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.reward_network.load_state_dict(checkpoint['model_state_dict'])
            self.reward_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded pre-trained reward network from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
        except Exception as e:
            print(f"Warning: Could not load pretrained reward network from {path}: {e}")
            print("Continuing with randomly initialized reward network...")
    
    def compute_self_reward(self, state_sequence, action):
        """
        Compute self-reward using the reward network with caching for speed.

        Args:
            state_sequence (np.ndarray): State sequence [seq_len, reward_input_dim]
            action (int): Action taken

        Returns:
            float: Self-predicted reward
        """
        # Create cache key (simple hash of state sequence and action)
        cache_key = (hash(state_sequence.tobytes()), action)
        
        # Check cache first
        if cache_key in self.reward_cache:
            return self.reward_cache[cache_key]
        
        # Convert to tensors and add batch dimension
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        
        # Get self-reward prediction
        with torch.no_grad():
            self_reward = self.reward_network(state_tensor, action_tensor).item()
        
        # Cache the result (with size limit)
        if len(self.reward_cache) < self.cache_max_size:
            self.reward_cache[cache_key] = self_reward
        
        return self_reward
    
    def compute_final_reward(self, state_sequence, action, expert_reward_dict):
        """
        Compute final reward using self-rewarding mechanism.
        This implements the core logic: reward = max(self_reward, expert_reward)
        
        Args:
            state_sequence (np.ndarray): State sequence [seq_len, feature_dim]
            action (int): Action taken
            expert_reward_dict (dict): Dictionary of expert rewards
            
        Returns:
            float: Final reward for RL training
        """
        # Get self-reward
        self_reward = self.compute_self_reward(state_sequence, action)

        # Get expert reward for selected label (safe lookup)
        expert_reward = expert_reward_dict.get(self.reward_labels[0], 0.0) if isinstance(expert_reward_dict, dict) else 0.0

        # Return maximum of self-reward and expert reward
        final_reward = max(self_reward, expert_reward)

        return final_reward
    
    def store_experience(self, state, action, final_reward, next_state, done, expert_reward_dict):
        """
        Store experience in both DQN replay buffer and SRDDQN buffer.
        
        Args:
            state: Current state
            action: Action taken
            final_reward: Final reward (max of self and expert)
            next_state: Next state
            done: Episode done flag
            expert_reward_dict: Dictionary of expert rewards
        """
        # NOTE: store the full state-sequence (seq_len x reward_input_dim) in the replay buffer
        # so reward-network training can reconstruct the temporal sequence at training time.
        # Flatten states for DQN if they are 2D (single-window observations)
        if len(state.shape) > 1:
            flat_state = state.flatten()
        else:
            flat_state = state

        if len(next_state.shape) > 1:
            flat_next_state = next_state.flatten()
        else:
            flat_next_state = next_state

        # Build orig_state as the sequence available in the per-env buffer at time of storing
        # Default to a sequence built from the current per-env sequence buffer (env 0 if unspecified)
        # Try to find the buffer which contains this state by checking recent buffers; fall back to single state
        orig_sequence = None
        for buf in self.state_sequence_buffers:
            # If the last stored element equals the provided state (by shape/value), assume this is the right buffer
            if len(buf) > 0 and np.array_equal(buf[-1], state):
                orig_sequence = self.get_state_sequence(env_idx=self.state_sequence_buffers.index(buf))
                break

        if orig_sequence is None:
            # Fallback: create a sequence by padding with copies of the provided state
            if len(state.shape) > 1:
                single_flat = state.flatten()
            else:
                single_flat = flat_state
            orig_sequence = np.array([single_flat for _ in range(self.seq_len)])

        # Store in unified replay buffer (include original state sequence & expert reward dict)
        self.dqn_agent.replay_buffer.add(flat_state, action, final_reward, flat_next_state, done,
                                         orig_state=orig_sequence, expert_reward_dict=expert_reward_dict)
    
    def select_action(self, state, training=True, env_idx=0):
        """
        Select an action using the DQN agent.
        
        Args:
            state (np.ndarray): Current state
            training (bool): Whether the agent is in training mode
            
        Returns:
            int: Selected action
        """
        # Add state to the correct per-env sequence buffer (store as flattened for DQN compatibility)
        if len(state.shape) > 1:
            # State is [window_size, feature_dim], flatten for DQN
            flattened_state = state.flatten()
        else:
            flattened_state = state
        # Push to per-env buffer
        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0
        buf = self.state_sequence_buffers[env_idx]
        if len(buf) == self.seq_len:
            buf.popleft()
        buf.append(state.copy())  # Store original shape for reward network
        
        # Use the DQN agent to select an action (it expects flattened state)
        return self.dqn_agent.select_action(flattened_state, training)
    
    def get_state_sequence(self, env_idx=0):
        """
        Get the current state sequence formatted for the reward network.
        Returns:
            np.ndarray: State sequence [seq_len, reward_input_dim]
        """
        # Select the buffer for this env
        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0
        buf = self.state_sequence_buffers[env_idx]

        # If we don't have enough states yet, pad with zeros
        # Build sequence where each element is the full flattened observation window
        if len(buf) < self.seq_len:
            padding_needed = self.seq_len - len(buf)
            # Create padding (zeros of length reward_input_dim)
            padding = [np.zeros(self.reward_input_dim) for _ in range(padding_needed)]

            state_vectors = []
            for state in list(buf):
                if len(state.shape) == 2:
                    # Flatten the entire observation window (window_size x feature_dim)
                    state_vectors.append(state.flatten())
                else:
                    # Already flattened
                    state_vectors.append(state)

            sequence_list = padding + state_vectors
        else:
            # We have enough states
            state_vectors = []
            for state in list(buf):
                if len(state.shape) == 2:
                    state_vectors.append(state.flatten())
                else:
                    state_vectors.append(state)
            sequence_list = state_vectors
        
        # Convert to numpy array
        sequence = np.array(sequence_list)

        return sequence
    
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
        expert_rewards = [d.get(self.reward_labels[0], 0.0) if isinstance(d, dict) else 0.0 for d in expert_reward_dicts]
        
        # Return the maximum of self-reward and expert-reward for each
        return [max(sr, er) for sr, er in zip(self_rewards, expert_rewards)]
    
    def train_reward_network_batch(self, experiences):
        """
        Train the reward network using a batch of experiences efficiently.
        During Phase 2, this trains the reward network to better predict expert rewards.
        
        Args:
            experiences: List of (state, action, reward, next_state, done, expert_reward_dict) tuples
            
        Returns:
            float: Average loss value
        """
        if len(experiences) == 0:
            return 0.0
        
        # Extract data from experiences
        state_sequences = []
        actions = []
        expert_rewards = []
        
        for exp in experiences:
            # Each experience now contains (orig_state, action, reward, next_state, done, expert_reward_dict)
            state, action, stored_reward, next_state, done, expert_reward_dict = exp

            # Reconstruct state sequence from state
            if isinstance(state, np.ndarray) and state.ndim == 1:
                # If the stored 1D array represents a full sequence of flattened windows,
                # reshape according to reward_input_dim per element if it matches expected length.
                per_elem = self.reward_input_dim
                if len(state) == self.seq_len * per_elem:
                    state_sequence = state.reshape(self.seq_len, per_elem)
                else:
                    # Fallback: try evenly splitting into seq_len parts
                    feature_dim = len(state) // self.seq_len
                    state_sequence = state.reshape(self.seq_len, feature_dim)
            else:
                state_sequence = state

            state_sequences.append(state_sequence)
            actions.append(action)
            # Prefer training target = stored final_reward (r_l) if provided (non-zero), else fallback to expert reward
            if stored_reward is not None and stored_reward != 0.0:
                target_val = stored_reward
            else:
                target_val = expert_reward_dict.get(self.reward_labels[0], 0.0) if isinstance(expert_reward_dict, dict) else 0.0
            expert_rewards.append(target_val)
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(np.array(state_sequences)).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        expert_reward_tensor = torch.FloatTensor(expert_rewards).to(self.device)
        
        # Forward pass
        predicted_rewards = self.reward_network(state_tensor, action_tensor).squeeze()
        
        # Handle single sample case
        if predicted_rewards.dim() == 0:
            predicted_rewards = predicted_rewards.unsqueeze(0)
        if expert_reward_tensor.dim() == 0:
            expert_reward_tensor = expert_reward_tensor.unsqueeze(0)
            
        # Compute loss (MSE between predicted and expert rewards)
        loss = F.mse_loss(predicted_rewards, expert_reward_tensor)
        
        # Optimize the model
        self.reward_optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
        self.reward_optimizer.step()
        
        return loss.item()
    
    def train(self):
        """
        Train the SRDDQN agent with proper two-phase training.
        
        Returns:
            tuple: (dqn_loss, reward_loss)
        """
        if len(self.dqn_agent.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Train the DQN agent (this uses the final rewards from DQN replay buffer)
        dqn_loss = self.dqn_agent.train()
        
        # Train the reward network with batch processing every sync_steps
        reward_loss = 0.0
        # Decide if we should train reward net this step based on scheduling parameters
        should_train_reward = (
            self.steps_done >= self.reward_warmup_steps and
            (self.steps_done % self.reward_update_interval == 0) and
            self.episode_reward_train_count < self.max_reward_train_per_episode
        )
        if should_train_reward and len(self.dqn_agent.replay_buffer) >= self.batch_size:
            extended_batch_size = min(len(self.dqn_agent.replay_buffer), self.batch_size * self.reward_batch_multiplier)
            states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr, orig_states, expert_dicts = \
                self.dqn_agent.replay_buffer.sample_extended(extended_batch_size)
            experiences = []
            # Use the sampled actions so the reward network learns action-conditional rewards
            for os, exd, a in zip(orig_states, expert_dicts, actions_arr):
                if os is None or exd is None:
                    continue
                experiences.append((os, int(a), 0.0, os, False, exd))
            if experiences:
                reward_loss = self.train_reward_network_batch(experiences)
                self.episode_reward_train_count += 1
        
        # Increment step counter
        self.steps_done += 1
        
        return dqn_loss, reward_loss

    def on_episode_start(self):
        """Reset per-episode counters at the beginning of an episode."""
        self.episode_reward_train_count = 0
    
    def save(self, path):
        """
        Save the agent's models.
        
        Args:
            path (str): Path to save the models (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Remove .pth extension if present to avoid double extensions
        base_path = path.replace('.pth', '')
        
        # Save DQN agent
        self.dqn_agent.save(f"{base_path}_dqn.pth")
        
        # Save reward network
        torch.save({
            'reward_network': self.reward_network.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'reward_train_steps': self.reward_train_steps
        }, f"{base_path}_reward.pth")
    
    def load(self, path):
        """
        Load the agent's models.
        
        Args:
            path (str): Path to load the models from (without extension)
        """
        # Remove .pth extension if present to avoid double extensions
        base_path = path.replace('.pth', '')
        
        # Load DQN agent
        self.dqn_agent.load(f"{base_path}_dqn.pth")
        
        # Load reward network
        checkpoint = torch.load(f"{base_path}_reward.pth", map_location=self.device)
        self.reward_network.load_state_dict(checkpoint['reward_network'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.reward_train_steps = checkpoint['reward_train_steps']

    @property
    def srddqn_replay_buffer(self):
        """Compatibility property exposing the underlying replay buffer.

        Some tests and older code expect `agent.srddqn_replay_buffer`. Expose the
        DoubleDQN agent's replay buffer here for compatibility.
        """
        return self.dqn_agent.replay_buffer