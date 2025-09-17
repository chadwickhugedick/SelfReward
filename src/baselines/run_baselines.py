import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.baselines.baseline_models import RandomAgent, BuyAndHoldAgent, MovingAverageAgent, DQNAgent, A2CAgent, PPOAgent
from src.models.agents.srddqn import SRDDQNAgent
from src.utils.comparison import compare_models, plot_portfolio_values, plot_action_distributions
from src.utils.device_utils import get_device

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_baseline(agent, env, episodes, render=False):
    """Run a baseline agent on the environment."""
    portfolio_values = []
    actions_history = []
    rewards_history = []
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        # Reset agent if it has a reset method
        if hasattr(agent, 'reset'):
            agent.reset()
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            
            # Store experience for training (if applicable)
            if hasattr(agent, 'remember'):
                agent.remember(state, action, reward, next_state, done)
            elif hasattr(agent, 'replay_buffer') and hasattr(agent.replay_buffer, 'push'):
                agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train the agent (if applicable)
            if hasattr(agent, 'train'):
                agent.train()
            
            state = next_state
            episode_reward += reward
            
            # Store portfolio value and action
            portfolio_values.append(info['portfolio_value'])
            actions_history.append(action)
            rewards_history.append(reward)
            
            if render:
                env.render()
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward:.4f}, Final Portfolio Value: {info['portfolio_value']:.4f}")
    
    # Calculate metrics
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
    
    metrics = {
        'final_portfolio_value': portfolio_values[-1],
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_reward': sum(rewards_history),
        'portfolio_values': portfolio_values,
        'actions_history': actions_history,
        'rewards_history': rewards_history
    }
    
    return metrics

def train_and_evaluate_baselines(config_path, test_data_path, models_dir):
    """Train and evaluate baseline models."""
    # Load configuration
    config = load_config(config_path)
    
    # Load test data
    data_processor = DataProcessor(config['data'])
    test_data = pd.read_csv(test_data_path)
    
    # Preprocess test data
    test_data = data_processor.add_technical_indicators(test_data)
    test_data = data_processor.normalize_data(test_data)
    
    # Create test environment
    env_config = config['environment']
    test_env = TradingEnvironment(
        data=test_data,
        window_size=env_config['window_size'],
        commission=env_config['commission'],
        initial_balance=env_config['initial_balance'],
        reward_scaling=env_config['reward_scaling'],
        reward_type=env_config['reward_type']
    )
    
    # Get state and action dimensions
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    
    # Create baseline agents
    random_agent = RandomAgent(action_dim)
    buy_hold_agent = BuyAndHoldAgent(action_dim)
    ma_agent = MovingAverageAgent(action_dim)
    
    # Create DQN agent
    dqn_config = config['model']['dqn']
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=dqn_config['hidden_dim'],
        learning_rate=dqn_config['learning_rate'],
        gamma=dqn_config['gamma'],
        epsilon_start=dqn_config['epsilon_start'],
        epsilon_end=dqn_config['epsilon_end'],
        epsilon_decay=dqn_config['epsilon_decay'],
        buffer_size=dqn_config['buffer_size'],
        batch_size=dqn_config['batch_size']
    )
    
    # Create A2C agent
    a2c_agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        actor_lr=0.0001,
        critic_lr=0.0001,
        gamma=0.99
    )
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        actor_lr=0.0001,
        critic_lr=0.0001,
        gamma=0.99,
        clip_ratio=0.2,
        epochs=10,
        batch_size=64
    )
    
    # Load SRDDQN agent if available
    srddqn_path = os.path.join(models_dir, 'srddqn_model.pth')
    srddqn_agent = None
    if os.path.exists(srddqn_path):
        # Create SRDDQN agent with the same configuration
        srddqn_config = config['model']
        srddqn_agent = SRDDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            reward_net_config=srddqn_config['reward_net'],
            dqn_config=srddqn_config['dqn'],
            device=get_device()
        )
        srddqn_agent.load(srddqn_path)
    
    # Define agents to evaluate
    agents = {
        'Random': random_agent,
        'Buy & Hold': buy_hold_agent,
        'Moving Average': ma_agent,
        'DQN': dqn_agent,
        'A2C': a2c_agent,
        'PPO': ppo_agent
    }
    
    if srddqn_agent is not None:
        agents['SRDDQN'] = srddqn_agent
    
    # Evaluate each agent
    results = {}
    for name, agent in agents.items():
        print(f"\nEvaluating {name} agent...")
        metrics = run_baseline(agent, test_env, episodes=1)  # One episode covers the entire test dataset
        results[name] = metrics
        print(f"{name} Results:")
        print(f"  Final Portfolio Value: {metrics['final_portfolio_value']:.4f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Total Reward: {metrics['total_reward']:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Compare models
    compare_models(results, save_path='results/baseline_comparison.png')
    
    # Plot portfolio values
    plot_portfolio_values(results, save_path='results/portfolio_values_comparison.png')
    
    # Plot action distributions
    plot_action_distributions(results, save_path='results/action_distributions.png')
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run and evaluate baseline models')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml', help='Path to configuration file')
    parser.add_argument('--test_data', type=str, default='data/processed/test_data.csv', help='Path to test data')
    parser.add_argument('--models_dir', type=str, default='models/saved', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    results = train_and_evaluate_baselines(
        config_path=args.config,
        test_data_path=args.test_data,
        models_dir=args.models_dir
    )