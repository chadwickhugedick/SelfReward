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

# Mixed precision training support
try:
    from torch.amp import autocast, GradScaler
    MIXED_PRECISION_AVAILABLE = True
    NEW_AMP_API = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        MIXED_PRECISION_AVAILABLE = True
        NEW_AMP_API = False
    except ImportError:
        MIXED_PRECISION_AVAILABLE = False
        NEW_AMP_API = False
        autocast = None
        GradScaler = None

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
    
    # Initialize mixed precision training if available and using GPU
    use_mixed_precision = (MIXED_PRECISION_AVAILABLE and 
                          torch.cuda.is_available() and 
                          config.get('mixed_precision', True))
    
    if use_mixed_precision:
        try:
            if NEW_AMP_API:
                # Try the new API first
                scaler = GradScaler('cuda')
            else:
                # Use old API
                scaler = GradScaler()
        except (TypeError, AttributeError):
            # Fall back to old API
            scaler = GradScaler()
        logging.info("Mixed precision training enabled")
    else:
        scaler = None
        logging.info("Mixed precision training disabled")

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
    episode_epsilons = []

    logging.info(f"Starting SRDDQN training for {num_episodes} episodes...")

    # Support vectorized environments (SyncVectorEnv) with batch stepping
    # Detect gymnasium VectorEnv (has attribute `num_envs`) or our previous SyncVectorEnv
    is_vectorized = hasattr(train_env, 'num_envs')
    num_envs = getattr(train_env, 'num_envs', 1) if is_vectorized else 1

    def _get_env_info(info_batch, env_idx, num_envs):
        """Normalize various `infos` shapes into a per-env dict.

        Handles:
        - list/tuple/ndarray of per-env dicts
        - dict of arrays (key -> array of length num_envs)
        - dict of int->dict mappings
        - fallback to empty dict when unavailable
        """
        # Case: dict of arrays or dict of int->dict
        if isinstance(info_batch, dict):
            # If keys are integers (0..n-1), try direct indexing
            try:
                if env_idx in info_batch:
                    return info_batch[env_idx]
            except Exception:
                pass

            # If values are sequence-like of length num_envs, build a per-env dict
            per_env = {}
            any_seq = False
            for k, v in info_batch.items():
                if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                    try:
                        if len(v) == num_envs:
                            per_env[k] = v[env_idx]
                            any_seq = True
                        else:
                            # not per-env sequence; skip
                            continue
                    except Exception:
                        continue
                else:
                    # scalar value, copy as-is
                    per_env[k] = v

            if any_seq:
                return per_env

            # As a last resort, return first value if it's a dict
            try:
                first = next(iter(info_batch.values()))
                if isinstance(first, dict):
                    return first
            except Exception:
                pass

            return {}

        # Case: list/tuple/ndarray
        if isinstance(info_batch, (list, tuple)):
            if env_idx < len(info_batch):
                return info_batch[env_idx] if isinstance(info_batch[env_idx], dict) else {}
            return {}

        # numpy array or scalar: no structured info available
        return {}

    for episode in range(num_episodes):
        # Reset env(s)
        if is_vectorized:
            state, info_batch = train_env.reset()
        else:
            state, info = train_env.reset()
        # Reset agent episode-level counters
        if hasattr(agent, 'on_episode_start'):
            agent.on_episode_start()
        # For vectorized envs, done is per-env
        if is_vectorized:
            done = [False] * num_envs
        else:
            done = False
        episode_reward = 0.0
        episode_dqn_losses = []
        episode_reward_net_losses = []
        steps = 0
        trade_logs = []

        # Per-episode financial tracking
        if is_vectorized:
            # try to extract initial portfolio value from reset infos
            try:
                initial_info = _get_env_info(info_batch, 0, num_envs)
            except Exception:
                initial_info = {}
            initial_portfolio_value = initial_info.get('portfolio_value', 0.0)
        else:
            initial_portfolio_value = getattr(train_env, 'initial_capital', 0.0)
        trades_executed = 0
        winning_trades = 0
        losing_trades = 0
        last_position = 0
        last_entry_price = 0

        # Episode loop: for vectorized envs we run until all envs are done
        while (not is_vectorized and not done) or (is_vectorized and not all(done)):
            # Select action
            if is_vectorized:
                # state is a batch of observations: shape (num_envs, ...)
                actions = []
                for env_idx, s in enumerate(state):
                    actions.append(agent.select_action(s, env_idx=env_idx))

                # Gymnasium VectorEnv.step returns (obs, rewards, dones, truncs, infos)
                next_state, expert_reward_batch, done_batch, truncs, info_batch = train_env.step(actions)
                # treat either done or trunc as terminal
                done_batch = [d or t for d, t in zip(done_batch, truncs)]
            else:
                action = agent.select_action(state)
                next_state, expert_reward, terminated, truncated, info = train_env.step(action)
                done = bool(terminated or truncated)

            if is_vectorized:
                # Process each env's transition
                for env_idx in range(num_envs):
                    a = actions[env_idx]
                    inf = _get_env_info(info_batch, env_idx, num_envs)
                    reward_dict = inf.get('reward_dict', {})
                    # Compute final reward using per-env state sequence
                    current_state_sequence = agent.get_state_sequence(env_idx=env_idx)
                    final_reward = agent.compute_final_reward(current_state_sequence, a, reward_dict)

                    # Store experience in SRDDQN buffers
                    agent.store_experience(state[env_idx], a, final_reward, next_state[env_idx], done_batch[env_idx], reward_dict)
            else:
                # Get reward dictionary from info
                reward_dict = info.get('reward_dict', {}) if isinstance(info, dict) else {}

                # Compute final reward using self-rewarding mechanism
                # Get current state sequence for reward computation
                current_state_sequence = agent.get_state_sequence()
                final_reward = agent.compute_final_reward(current_state_sequence, action, reward_dict)

            # Track trades and PnL
            # For vectorized envs, take env 0's info for aggregated logging
            if is_vectorized:
                main_info = _get_env_info(info_batch, 0, num_envs)
            else:
                main_info = info

            current_price = main_info.get('current_price', 0.0)
            current_portfolio_value = main_info.get('portfolio_value', initial_portfolio_value)

            # Detect trade execution (only in single-env mode)
            if not is_vectorized:
                if action == 1 and last_position == 0:  # Buy signal
                    trades_executed += 1
                    last_position = 1
                    last_entry_price = current_price
                    trade_logs.append(f"Step {steps} | BUY | Price: {current_price:.2f} | Portfolio: {current_portfolio_value:.2f}")

                elif action == 2 and last_position == 1:  # Sell signal
                    trades_executed += 1
                    last_position = 0
                    # Calculate trade PnL
                    trade_pnl = (current_price - last_entry_price) / last_entry_price if last_entry_price != 0 else 0.0
                    if trade_pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    trade_logs.append(f"Step {steps} | SELL | Price: {current_price:.2f} | Trade PnL: {trade_pnl:.4f} | Portfolio: {current_portfolio_value:.2f}")

            if not is_vectorized:
                # Update state for single env
                agent.store_experience(state, action, final_reward, next_state, done, reward_dict)
                state = next_state
                episode_reward += final_reward
                steps += 1
            else:
                # For vectorized envs, advance state batch and update metrics
                state = next_state
                # Compute per-env final rewards for logging (avoid KeyError by using normalized infos)
                per_env_rewards = []
                for idx, a in enumerate(actions):
                    inf_i = _get_env_info(info_batch, idx, num_envs)
                    rd = inf_i.get('reward_dict', {})
                    seq = agent.get_state_sequence(env_idx=idx)
                    per_env_rewards.append(agent.compute_final_reward(seq, a, rd))
                episode_reward += sum(per_env_rewards)
                steps += num_envs

            # Train agent if buffer has enough samples
            if len(agent.dqn_agent.replay_buffer) > batch_size:
                if use_mixed_precision:
                    # Train with mixed precision
                    if NEW_AMP_API:
                        with autocast('cuda'):
                            dqn_loss, reward_net_loss = agent.train()
                    else:
                        with autocast():
                            dqn_loss, reward_net_loss = agent.train()
                else:
                    # Train with normal precision
                    dqn_loss, reward_net_loss = agent.train()
                    
                if dqn_loss is not None:
                    episode_dqn_losses.append(dqn_loss)
                if reward_net_loss is not None:
                    episode_reward_net_losses.append(reward_net_loss)

        # Calculate episode financial metrics
        if is_vectorized:
            try:
                final_info = _get_env_info(info_batch, 0, num_envs)
            except Exception:
                final_info = {}
            final_portfolio_value = final_info.get('portfolio_value', initial_portfolio_value)
        else:
            final_portfolio_value = train_env.portfolio_values[-1] if train_env.portfolio_values else initial_portfolio_value
        episode_pnl = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value

        # Calculate win rate
        total_closed_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0

        # Get portfolio stats
        if not is_vectorized:
            stats = train_env.get_portfolio_stats()
            sharpe_ratio = stats.get('SR', 0.0)
            max_drawdown = stats.get('MDD', 0.0)
        else:
            stats = final_info.get('stats', {}) if isinstance(final_info, dict) else {}
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
        episode_epsilons.append(agent.dqn_agent.epsilon)

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
        'episode_max_drawdowns': episode_max_drawdowns,
        'episode_epsilons': episode_epsilons
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