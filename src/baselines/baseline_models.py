import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random

class RandomAgent:
    """Random agent that selects actions randomly."""
    
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def select_action(self, state, training=False):
        return random.randrange(self.action_dim)
    
    def train(self):
        return 0.0
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass

class BuyAndHoldAgent:
    """Buy and Hold agent that buys at the beginning and holds."""
    
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.has_bought = False
    
    def select_action(self, state, training=False):
        if not self.has_bought:
            self.has_bought = True
            return 2  # Buy action
        else:
            return 1  # Hold action
    
    def train(self):
        return 0.0
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def reset(self):
        self.has_bought = False

class MovingAverageAgent:
    """Moving Average agent that buys when short MA crosses above long MA and sells when it crosses below."""
    
    def __init__(self, action_dim, short_window=10, long_window=50):
        self.action_dim = action_dim
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = []
    
    def select_action(self, state, training=False):
        # Extract price from state (assuming price is the first feature)
        price = state[0]
        self.price_history.append(price)
        
        # If we don't have enough history, hold
        if len(self.price_history) < self.long_window:
            return 1  # Hold action
        
        # Calculate moving averages
        short_ma = np.mean(self.price_history[-self.short_window:])
        long_ma = np.mean(self.price_history[-self.long_window:])
        
        # Previous moving averages
        prev_short_ma = np.mean(self.price_history[-self.short_window-1:-1])
        prev_long_ma = np.mean(self.price_history[-self.long_window-1:-1])
        
        # Check for crossover
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            return 2  # Buy action
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            return 0  # Sell action
        else:
            return 1  # Hold action
    
    def train(self):
        return 0.0
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def reset(self):
        self.price_history = []

class DQNAgent:
    """Standard DQN agent without double Q-learning."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Q-Network
        self.q_network = self._build_q_network().to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Step counter
        self.steps_done = 0
    
    def _build_q_network(self):
        """Build a simple Q-Network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
    
    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self):
        """Train the DQN agent."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Get next Q values
        next_q_values = self.q_network(next_states_tensor).max(1)[0].detach()
        
        # Compute target Q values
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Increment step counter
        self.steps_done += 1
        
        return loss.item()
    
    def save(self, path):
        """Save the agent's model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """Load the agent's model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        transitions = random.sample(self.buffer, batch_size)
        batch = self.Transition(*zip(*transitions))
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        dones = np.array(batch.done)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class A2CAgent:
    """Advantage Actor-Critic (A2C) agent."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, actor_lr=0.0001, critic_lr=0.0001, gamma=0.99,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device
        
        # Actor network (policy)
        self.actor = self._build_actor().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network (value function)
        self.critic = self._build_critic().to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Memory for storing episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Step counter
        self.steps_done = 0
    
    def _build_actor(self):
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self):
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def select_action(self, state, training=True):
        """Select an action using the actor network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            
            if training:
                # Sample action from the probability distribution
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Take the most probable action
                action = action_probs.argmax().item()
            
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def train(self):
        """Train the A2C agent."""
        if len(self.states) == 0:
            return 0.0, 0.0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(self.states).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        rewards_tensor = torch.FloatTensor(self.rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(self.next_states).to(self.device)
        dones_tensor = torch.FloatTensor(self.dones).to(self.device)
        
        # Get values for current and next states
        values = self.critic(states_tensor).squeeze()
        next_values = self.critic(next_states_tensor).squeeze().detach()
        
        # Compute returns and advantages
        returns = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
        advantages = returns - values
        
        # Get action probabilities
        action_probs = self.actor(states_tensor)
        selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Compute actor loss
        actor_loss = -torch.mean(torch.log(selected_action_probs) * advantages.detach())
        
        # Compute critic loss
        critic_loss = F.mse_loss(values, returns.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to avoid exploding gradients
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Increment step counter
        self.steps_done += 1
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path):
        """Save the agent's models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """Load the agent's models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.steps_done = checkpoint['steps_done']

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, actor_lr=0.0001, critic_lr=0.0001, gamma=0.99,
                 clip_ratio=0.2, epochs=10, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # Actor network (policy)
        self.actor = self._build_actor().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network (value function)
        self.critic = self._build_critic().to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Memory for storing episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Step counter
        self.steps_done = 0
    
    def _build_actor(self):
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self):
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def select_action(self, state, training=True):
        """Select an action using the actor network."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            
            if training:
                # Sample action from the probability distribution
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, action])
                self.log_probs.append(log_prob.item())
            else:
                # Take the most probable action
                action = action_probs.argmax().item()
            
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def train(self):
        """Train the PPO agent."""
        if len(self.states) == 0:
            return 0.0, 0.0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(self.states).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        rewards_tensor = torch.FloatTensor(self.rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(self.next_states).to(self.device)
        dones_tensor = torch.FloatTensor(self.dones).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            values = self.critic(states_tensor).squeeze()
            next_values = self.critic(next_states_tensor).squeeze()
            returns = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
            advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            # Get action probabilities
            action_probs = self.actor(states_tensor)
            selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            new_log_probs = torch.log(selected_action_probs)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # Compute surrogate losses
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Compute actor loss
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Compute critic loss
            values = self.critic(states_tensor).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Clip gradients to avoid exploding gradients
            for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Clip gradients to avoid exploding gradients
            for param in self.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Increment step counter
        self.steps_done += 1
        
        return total_actor_loss / self.epochs, total_critic_loss / self.epochs
    
    def save(self, path):
        """Save the agent's models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """Load the agent's models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.steps_done = checkpoint['steps_done']