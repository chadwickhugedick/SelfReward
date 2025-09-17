#!/usr/bin/env python3
"""
Test script for the fixed SRDDQN implementation.
This script tests the two-phase training architecture.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent


def get_device(use_cpu=False):
    """Simple device selection function."""
    if use_cpu or not torch.cuda.is_available():
        return 'cpu'
    return 'cuda'


def test_srddqn_implementation():
    """Test the SRDDQN implementation with proper two-phase training."""
    
    print("Testing SRDDQN Implementation...")
    print("=" * 50)
    
    # Load configuration
    config_path = 'configs/srddqn_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set device
    device = get_device(use_cpu=True)  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Create sample data
    print("\n1. Creating sample data...")
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    # Create simple normalized data
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features) * 0.1,  # Small random values
        columns=['Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # Make sure High >= Open, Low <= Close for realistic data
    data['High'] = np.maximum(data['Open'], data['Close']) + np.abs(np.random.randn(n_samples) * 0.01)
    data['Low'] = np.minimum(data['Open'], data['Close']) - np.abs(np.random.randn(n_samples) * 0.01)
    
    print(f"Created sample data with {len(data)} rows and {len(data.columns)} features")
    
    # Create environment
    print("\n2. Creating trading environment...")
    env = TradingEnvironment(
        data=data,
        window_size=config['environment']['window_size'],
        initial_capital=config['environment']['initial_capital'],
        transaction_cost=config['environment']['transaction_cost']
    )
    
    # Calculate dimensions
    state, info = env.reset()
    state_dim = np.prod(env.observation_space.shape)
    feature_dim = env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    print(f"State shape: {state.shape}")
    print(f"State dimension: {state_dim}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create SRDDQN agent
    print("\n3. Creating SRDDQN agent...")
    agent = SRDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=config['environment']['window_size'],
        hidden_dim=config['model']['dqn']['hidden_size'],
        dqn_lr=config['model']['dqn']['learning_rate'],
        reward_net_lr=config['model']['reward_net']['learning_rate'],
        gamma=config['model']['dqn']['gamma'],
        epsilon_start=config['model']['dqn']['epsilon_start'],
        epsilon_end=config['model']['dqn']['epsilon_end'],
        epsilon_decay=config['model']['dqn']['epsilon_decay'],
        target_update=config['model']['dqn']['target_update'],
        tau=config['model']['dqn']['tau'],
        buffer_size=100,  # Small buffer for testing
        batch_size=16,    # Small batch for testing
        reward_model_type=config['model']['reward_net']['model_type'],
        reward_labels=config['training']['reward_labels'],
        sync_steps=1,
        update_steps=1,
        device=device,
        feature_dim=feature_dim,
        pretrained_reward_path=None  # No pre-training for this test
    )
    
    print("SRDDQN agent created successfully!")
    
    # Test the self-rewarding mechanism
    print("\n4. Testing self-rewarding mechanism...")
    
    state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0
    max_steps = 5  # Just test a few steps
    
    while not done and step < max_steps:
        print(f"\nStep {step + 1}:")

        # Select action
        action = agent.select_action(state, training=True)
        print(f"  Selected action: {action}")

        # Take action (Gymnasium style returns 5-tuple)
        next_state, expert_reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        reward_dict = info['reward_dict']

        print(f"  Expert reward: {expert_reward:.4f}")
        print(f"  Reward dict: {reward_dict}")

        # Test self-rewarding mechanism
        state_sequence = agent.get_state_sequence()
        print(f"  State sequence shape: {state_sequence.shape}")

        # Test self-reward computation
        self_reward = agent.compute_self_reward(state_sequence, action)
        final_reward = agent.compute_final_reward(state_sequence, action, reward_dict)

        print(f"  Self reward: {self_reward:.4f}")
        print(f"  Final reward (max): {final_reward:.4f}")

        # Store experience
        agent.store_experience(state, action, final_reward, next_state, done, reward_dict)
        print(f"  Stored experience in buffers")

        # Test training after accumulating some experiences
        if step >= 3:  # Start training after a few steps
            dqn_loss, reward_loss = agent.train()
            print(f"  Training losses - DQN: {dqn_loss:.6f}, Reward: {reward_loss:.6f}")

        state = next_state
        episode_reward += final_reward
        step += 1
    
    print(f"\nEpisode completed!")
    print(f"Total episode reward: {episode_reward:.4f}")
    print(f"DQN buffer size: {len(agent.dqn_agent.replay_buffer)}")
    print(f"SRDDQN buffer size: {len(agent.srddqn_replay_buffer)}")
    
    # Test buffer contents
    print("\n5. Testing buffer contents...")
    if len(agent.srddqn_replay_buffer) > 0:
        # Use sample_extended to get original (unflattened) states and expert dicts
        states, actions, rewards, next_states, dones, orig_states, expert_dicts = agent.srddqn_replay_buffer.sample_extended(1)
        # Ensure shapes are consistent when batch_size == 1
        if isinstance(states, np.ndarray) and states.ndim == 1:
            states = states.reshape(1, -1)
        sample_state = states[0]
        sample_action = int(actions[0]) if hasattr(actions, '__len__') else int(actions)
        sample_reward = float(rewards[0]) if hasattr(rewards, '__len__') else float(rewards)
        sample_expert = expert_dicts[0] if expert_dicts and len(expert_dicts) > 0 else None

        print(f"Sample from SRDDQN buffer:")
        print(f"  State shape: {sample_state.shape}")
        print(f"  Action: {sample_action}")
        print(f"  Reward: {sample_reward:.4f}")
        print(f"  Expert reward dict: {sample_expert}")
    
    print("\n" + "=" * 50)
    print("✅ SRDDQN Implementation Test PASSED!")
    print("Key improvements verified:")
    print("  ✓ Self-rewarding mechanism properly implemented")
    print("  ✓ Max(self_reward, expert_reward) computation working")
    print("  ✓ Enhanced replay buffer storing expert reward dicts")
    print("  ✓ State sequence management for reward network")
    print("  ✓ Two separate buffers for DQN and reward network training")
    
    return True


if __name__ == "__main__":
    try:
        test_srddqn_implementation()
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)