import os
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import argparse

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file):
    """Set up logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_srddqn(agent, train_env, val_env, config, model_path):
    """Train the SRDDQN agent with the provided agent and environments."""
    # Set up logging
    os.makedirs('results', exist_ok=True)
    setup_logging('results/training_metrics.log')

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Training parameters
    num_episodes = config['training']['num_episodes']
    batch_size = config['training']['batch_size']

    # Initialize metrics tracking
    episode_rewards = []
    portfolio_values = []
    dqn_losses = []
    reward_net_losses = []
    episode_pnls = []
    episode_win_rates = []
    episode_total_trades = []
    episode_sharpe_ratios = []
    episode_max_drawdowns = []

    logging.info(f"Starting SRDDQN training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = train_env.reset()
        done = False
        episode_reward = 0.0
        episode_dqn_losses = []
        episode_reward_net_losses = []
        steps = 0
        trade_logs = []

        # Per-episode financial tracking
        initial_portfolio_value = train_env.initial_capital
        trades_executed = 0
        winning_trades = 0
        losing_trades = 0
        last_position = 0
        last_entry_price = 0

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = train_env.step(action)

            # Track trades and PnL
            current_price = info['current_price']
            current_portfolio_value = info['portfolio_value']

            # Detect trade execution
            if action == 1 and last_position == 0:  # Buy signal
                trades_executed += 1
                last_position = 1
                last_entry_price = current_price
                trade_logs.append(f"Step {steps} | BUY | Price: {current_price:.2f} | Portfolio: {current_portfolio_value:.2f}")

            elif action == 2 and last_position == 1:  # Sell signal
                trades_executed += 1
                last_position = 0
                # Calculate trade PnL
                trade_pnl = (current_price - last_entry_price) / last_entry_price
                if trade_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                trade_logs.append(f"Step {steps} | SELL | Price: {current_price:.2f} | Trade PnL: {trade_pnl:.4f} | Portfolio: {current_portfolio_value:.2f}")

            # Store experience
            agent.dqn_agent.replay_buffer.add(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward
            steps += 1

            # Train agent if buffer has enough samples
            if len(agent.dqn_agent.replay_buffer) > batch_size:
                dqn_loss, reward_net_loss = agent.train()
                if dqn_loss is not None:
                    episode_dqn_losses.append(dqn_loss)
                if reward_net_loss is not None:
                    episode_reward_net_losses.append(reward_net_loss)

        # Calculate episode financial metrics
        final_portfolio_value = train_env.portfolio_values[-1] if train_env.portfolio_values else initial_portfolio_value
        episode_pnl = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

        # Calculate win rate
        total_closed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0

        # Get portfolio stats
        stats = train_env.get_portfolio_stats()
        sharpe_ratio = stats.get('SR', 0.0)
        max_drawdown = stats.get('MDD', 0.0)

        # Store episode metrics
        episode_rewards.append(episode_reward)
        portfolio_values.append(final_portfolio_value)
        episode_pnls.append(episode_pnl)
        episode_win_rates.append(win_rate)
        episode_total_trades.append(trades_executed)
        episode_sharpe_ratios.append(sharpe_ratio)
        episode_max_drawdowns.append(max_drawdown)

        # Store average losses
        avg_dqn_loss = np.mean(episode_dqn_losses) if episode_dqn_losses else 0.0
        avg_reward_net_loss = np.mean(episode_reward_net_losses) if episode_reward_net_losses else 0.0
        dqn_losses.append(avg_dqn_loss)
        reward_net_losses.append(avg_reward_net_loss)

        # Log comprehensive metrics
        logging.info(f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"PnL: {episode_pnl:.4f} | "
                    f"Portfolio: {final_portfolio_value:.2f} | "
                    f"Trades: {trades_executed} | "
                    f"Win Rate: {win_rate:.2f} | "
                    f"Sharpe: {sharpe_ratio:.4f} | "
                    f"Max DD: {max_drawdown:.4f} | "
                    f"DQN Loss: {avg_dqn_loss:.6f} | "
                    f"Reward Loss: {avg_reward_net_loss:.6f} | "
                    f"Epsilon: {agent.dqn_agent.epsilon:.4f}")

        # Log batched trade logs
        for log in trade_logs:
            logging.info(log)

    # Save model
    agent.save(model_path)
    logging.info(f"Model saved to {model_path}")

    logging.info("Training completed!")

    # Return comprehensive training metrics
    return {
        'episode_rewards': episode_rewards,
        'portfolio_values': portfolio_values,
        'dqn_losses': dqn_losses,
        'reward_net_losses': reward_net_losses,
        'episode_pnls': episode_pnls,
        'episode_win_rates': episode_win_rates,
        'episode_total_trades': episode_total_trades,
        'episode_sharpe_ratios': episode_sharpe_ratios,
        'episode_max_drawdowns': episode_max_drawdowns
    }

def main():
    parser = argparse.ArgumentParser(description='Train SRDDQN agent for financial trading')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Train agent
    train_srddqn(config)

if __name__ == '__main__':
    main()