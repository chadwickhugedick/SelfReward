import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.utils.device_utils import get_device

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(config, model_path):
    """Evaluate a trained SRDDQN model."""
    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Set device
    device = get_device()
    
    # Initialize data processor
    data_processor = DataProcessor(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        data_dir=config['data']['data_dir'],
        window_size=config['data']['window_size'],
        feature_list=config['data']['features']
    )
    
    # Process data
    print("Processing data...")
    data_processor.download_data()
    data_processor.add_technical_indicators()
    data_processor.normalize_data()
    
    # Split data into train and test sets
    _, test_data = data_processor.split_data(
        split_ratio=config['data']['train_test_split']
    )
    
    # Create test environment
    test_env = TradingEnvironment(
        data=test_data,
        window_size=config['data']['window_size'],
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        reward_scaling=config['environment']['reward_scaling'],
        reward_mode=config['environment']['reward_mode'],
        reward_weights=config['environment']['reward_weights']
    )
    
    # Get state dimension
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    
    # Initialize SRDDQN agent
    agent = SRDDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=config['model']['reward_net']['seq_len'],
        hidden_dim=config['model']['dqn']['hidden_dim'],
        dqn_lr=config['model']['dqn']['learning_rate'],
        reward_net_lr=config['model']['reward_net']['learning_rate'],
        gamma=config['model']['dqn']['gamma'],
        epsilon_start=config['model']['dqn']['epsilon_start'],
        epsilon_end=config['model']['dqn']['epsilon_end'],
        epsilon_decay=config['model']['dqn']['epsilon_decay'],
        target_update=config['model']['dqn']['target_update'],
        buffer_size=config['model']['dqn']['buffer_size'],
        batch_size=config['model']['dqn']['batch_size'],
        reward_model_type=config['model']['reward_net']['model_type'],
        reward_labels=config['model']['reward_net']['reward_labels'],
        sync_steps=config['model']['reward_net']['sync_steps'],
        update_steps=config['model']['reward_net']['update_steps'],
        device=device
    )
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    
    # Create directory for evaluation results
    eval_results_dir = os.path.join(config['training']['results_dir'], 'evaluation')
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # Evaluate agent
    print("Evaluating agent...")
    evaluate_agent(agent, test_env, config, eval_results_dir)

def evaluate_agent(agent, env, config, results_dir):
    """Evaluate the agent on the test environment."""
    state = env.reset()
    done = False
    total_reward = 0
    actions = []
    portfolio_values = [env.portfolio_value]
    rewards = []
    states = []
    prices = []
    holdings = []
    cash_balances = []
    
    # Store initial state and holdings
    states.append(state)
    holdings.append(env.shares_held)
    cash_balances.append(env.balance)
    prices.append(env.current_price)
    
    # Run evaluation episode
    while not done:
        # Select action (no exploration)
        action = agent.select_action(state, training=False)
        actions.append(action)
        
        # Take action in environment
        next_state, reward_dict, done, info = env.step(action)
        
        # Use the first reward label by default
        reward = reward_dict[config['model']['reward_net']['reward_labels'][0]]
        rewards.append(reward)
        
        # Store state and portfolio information
        states.append(next_state)
        portfolio_values.append(env.portfolio_value)
        holdings.append(env.shares_held)
        cash_balances.append(env.balance)
        prices.append(env.current_price)
        
        # Update state and reward
        state = next_state
        total_reward += reward
    
    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Final Portfolio Value: {env.portfolio_value:.2f}")
    print(f"Initial Portfolio Value: {config['environment']['initial_balance']:.2f}")
    print(f"Return: {(env.portfolio_value / config['environment']['initial_balance'] - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {env.sharpe_ratio:.4f}")
    print(f"Max Drawdown: {env.max_drawdown:.4f}")
    print(f"Cumulative Return: {env.cumulative_return:.4f}")
    print(f"Annualized Return: {env.annualized_return:.4f}")
    
    # Create evaluation visualizations
    create_evaluation_plots(
        portfolio_values, actions, rewards, prices, holdings, cash_balances,
        results_dir, config
    )
    
    # Save evaluation metrics
    save_evaluation_metrics(
        env, total_reward, results_dir, config
    )
    
    # Save trading history
    save_trading_history(
        actions, prices, holdings, cash_balances, portfolio_values, rewards,
        results_dir
    )

def create_evaluation_plots(portfolio_values, actions, rewards, prices, holdings, cash_balances, results_dir, config):
    """Create evaluation plots."""
    # Convert action indices to labels
    action_labels = ['Sell', 'Hold', 'Buy']
    action_names = [action_labels[a] for a in actions]
    
    # Create time steps
    time_steps = np.arange(len(portfolio_values))
    
    # Plot portfolio value and price
    plt.figure(figsize=(12, 8))
    
    # Portfolio value subplot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_steps, portfolio_values, 'b-', label='Portfolio Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Price and actions subplot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(time_steps[1:], prices[1:], 'k-', label='Asset Price')
    
    # Mark buy and sell actions
    buys = [i for i, a in enumerate(actions) if a == 2]
    sells = [i for i, a in enumerate(actions) if a == 0]
    
    if buys:
        ax2.plot(buys, [prices[i+1] for i in buys], '^', markersize=8, color='g', label='Buy')
    if sells:
        ax2.plot(sells, [prices[i+1] for i in sells], 'v', markersize=8, color='r', label='Sell')
    
    ax2.set_title('Asset Price and Trading Actions')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Holdings and cash subplot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(time_steps, holdings, 'g-', label='Holdings')
    ax3.set_title('Asset Holdings Over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Shares Held')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'portfolio_performance.png'))
    plt.close()
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps[1:], rewards)
    plt.title('Rewards Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'rewards.png'))
    plt.close()
    
    # Plot cash balance
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, cash_balances)
    plt.title('Cash Balance Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cash ($)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'cash_balance.png'))
    plt.close()
    
    # Plot action distribution
    plt.figure(figsize=(10, 6))
    action_counts = [action_names.count(label) for label in action_labels]
    plt.bar(action_labels, action_counts)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.savefig(os.path.join(results_dir, 'action_distribution.png'))
    plt.close()

def save_evaluation_metrics(env, total_reward, results_dir, config):
    """Save evaluation metrics to CSV."""
    metrics = {
        'total_reward': total_reward,
        'final_portfolio_value': env.portfolio_value,
        'initial_portfolio_value': config['environment']['initial_balance'],
        'return_percentage': (env.portfolio_value / config['environment']['initial_balance'] - 1) * 100,
        'sharpe_ratio': env.sharpe_ratio,
        'max_drawdown': env.max_drawdown,
        'cumulative_return': env.cumulative_return,
        'annualized_return': env.annualized_return
    }
    
    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, 'evaluation_metrics.csv'),
        index=False
    )

def save_trading_history(actions, prices, holdings, cash_balances, portfolio_values, rewards, results_dir):
    """Save trading history to CSV."""
    # Convert action indices to labels
    action_labels = ['Sell', 'Hold', 'Buy']
    action_names = [action_labels[a] for a in actions]
    
    # Create DataFrame
    history = pd.DataFrame({
        'time_step': np.arange(1, len(prices)),
        'price': prices[1:],
        'action': action_names,
        'holdings': holdings[1:],
        'cash_balance': cash_balances[1:],
        'portfolio_value': portfolio_values[1:],
        'reward': rewards
    })
    
    # Save to CSV
    history.to_csv(os.path.join(results_dir, 'trading_history.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Evaluate SRDDQN agent for financial trading')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='models/best_model',
                        help='Path to trained model')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model
    evaluate_model(config, args.model)

if __name__ == '__main__':
    main()