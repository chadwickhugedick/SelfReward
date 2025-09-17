import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from gymnasium.vector import AsyncVectorEnv
from src.environment.vectorized_env import SyncVectorEnv
from src.models.agents.srddqn import SRDDQNAgent
from src.train import train_srddqn
from src.evaluate import evaluate_agent
from src.baselines.run_baselines import train_and_evaluate_baselines
from src.utils.device_utils import get_device

def main(args):
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = get_device(use_cpu=args.cpu)
    
    # Process data (DataProcessor reads multi_horizon from top-level config['data'])
    data_processor = DataProcessor(config)

    if args.parquet_file:
        print(f"Loading data from parquet file: {args.parquet_file}")
        if args.resample:
            print(f"Will resample to {args.resample} timeframe")
        
        # Use parquet file instead of downloading
        train_data, val_data, test_data = data_processor.prepare_data_from_parquet(
            args.parquet_file, 
            resample_timeframe=args.resample
        )
        
        if train_data is None:
            print("Failed to load data from parquet file")
            return
            
        # Save processed data
        train_data.to_csv('data/processed/train_data.csv', index=True)  # Keep datetime index
        val_data.to_csv('data/processed/val_data.csv', index=True)
        test_data.to_csv('data/processed/test_data.csv', index=True)
        print("Processed parquet data saved to data/processed/")
        
    elif args.multi_index:
        print("Multi-index training mode enabled. Training on all six indices.")
        # Process data for all tickers
        multi_data = data_processor.prepare_multi_ticker_data()

        if not multi_data:
            print("Failed to download data for any tickers.")
            return

        # For multi-index training, we'll use the first ticker's data for now
        # In a full implementation, you'd train on all indices
        first_ticker = list(multi_data.keys())[0]
        train_data = multi_data[first_ticker]['train_data']
        val_data = multi_data[first_ticker]['val_data']
        test_data = multi_data[first_ticker]['test_data']

        print(f"Using data from {first_ticker} for training (multi-index mode)")
    else:
        # Single ticker mode
        if args.download_data:
            print("Downloading data...")
            # Get interval from config if available
            interval = config['data'].get('interval')
            data = data_processor.download_data(
            ticker=config['data']['ticker'],  # Use ticker from config
            start_date=config['data']['train_start_date'],
            end_date=config['data']['test_end_date'],
            interval=interval
        )
            data.to_csv('data/raw/market_data.csv', index=False)
            print(f"Data saved to data/raw/market_data.csv")
        else:
            # Load data if it exists
            try:
                data = pd.read_csv('data/raw/market_data.csv')
                print("Loaded data from data/raw/market_data.csv")
            except FileNotFoundError:
                print("No data found. Please use --download_data flag to download data first.")
                return

        # Add technical indicators BEFORE splitting to avoid leakage but do not normalize yet
        data_with_indicators = data_processor.add_technical_indicators(data)

        # Split raw feature data first (prevents normalization leakage)
        train_raw, val_raw, test_raw = data_processor.split_data(data_with_indicators)

        # Fit scaler ONLY on training data
        train_data = data_processor.normalize_data(train_raw)
        val_data = pd.DataFrame(
            data_processor.scaler.transform(val_raw),
            columns=val_raw.columns,
            index=val_raw.index
        )
        test_data = pd.DataFrame(
            data_processor.scaler.transform(test_raw),
            columns=test_raw.columns,
            index=test_raw.index
        )

        # Save processed splits
        train_data.to_csv('data/processed/train_data.csv', index=False)
        val_data.to_csv('data/processed/val_data.csv', index=False)
        test_data.to_csv('data/processed/test_data.csv', index=False)
        print("Processed data saved to data/processed/ (no normalization leakage)")
    
    # Create environments
    env_config = config['environment']
    
    # Optionally create vectorized environments for faster sampling
    num_envs = config.get('training', {}).get('num_envs', 1)
    # Allow CLI override
    if getattr(args, 'num_envs', None) is not None:
        num_envs = int(args.num_envs)
    debug_sync = getattr(args, 'debug_sync', False)
    expert_selection_cfg = config.get('training', {}).get('expert_selection', None)
    if num_envs > 1:
        def make_env(data):
            def _thunk():
                return TradingEnvironment(data=data,
                                          window_size=env_config['window_size'],
                                          initial_capital=env_config.get('initial_capital', 500000),
                                          transaction_cost=env_config.get('transaction_cost', 0.003),
                                          expert_selection=expert_selection_cfg)
            return _thunk
        factories = [make_env(train_data) for _ in range(num_envs)]
        if debug_sync:
            train_env = SyncVectorEnv(factories)
            val_env = SyncVectorEnv([make_env(val_data) for _ in range(num_envs)])
            test_env = SyncVectorEnv([make_env(test_data) for _ in range(num_envs)])
        else:
            train_env = AsyncVectorEnv([make_env(train_data) for _ in range(num_envs)])
            val_env = AsyncVectorEnv([make_env(val_data) for _ in range(num_envs)])
            test_env = AsyncVectorEnv([make_env(test_data) for _ in range(num_envs)])
    else:
        train_env = TradingEnvironment(
            data=train_data,
            window_size=env_config['window_size'],
            initial_capital=env_config.get('initial_capital', 500000),
            transaction_cost=env_config.get('transaction_cost', 0.003),
            expert_selection=expert_selection_cfg
        )
        
        val_env = TradingEnvironment(
            data=val_data,
            window_size=env_config['window_size'],
            initial_capital=env_config.get('initial_capital', 500000),
            transaction_cost=env_config.get('transaction_cost', 0.003),
            expert_selection=expert_selection_cfg
        )
        
        test_env = TradingEnvironment(
            data=test_data,
            window_size=env_config['window_size'],
            initial_capital=env_config.get('initial_capital', 500000),
            transaction_cost=env_config.get('transaction_cost', 0.003),
            expert_selection=expert_selection_cfg
        )
    
    # Get state and action dimensions
    state_dim = np.prod(train_env.observation_space.shape)  # Total size for DQN
    feature_dim = train_env.observation_space.shape[1]  # Feature dimension per time step for reward network
    action_dim = train_env.action_space.n
    
    # Create the directory if it doesn't exist
    os.makedirs('models/saved', exist_ok=True)
    
    # Check for pre-trained reward network
    pretrained_reward_path = os.path.join('models/saved', 'pretrained_reward_network.pth')
    pretrained_reward_exists = os.path.exists(pretrained_reward_path)
    
    # Create agent
    agent = SRDDQNAgent(
        state_dim=state_dim,  # Flattened state dimension for DQN
        action_dim=action_dim,
        seq_len=config['environment']['window_size'],
        feature_dim=feature_dim,  # Explicit feature dimension per timestep
        hidden_dim=config['model']['dqn']['hidden_size'],
        dqn_lr=config['model']['dqn']['learning_rate'],
        reward_net_lr=config['model']['reward_net']['learning_rate'],
        gamma=config['model']['dqn']['gamma'],
        epsilon_start=config['model']['dqn']['epsilon_start'],
        epsilon_end=config['model']['dqn']['epsilon_end'],
        epsilon_decay=config['model']['dqn']['epsilon_decay'],
        target_update=config['model']['dqn']['target_update'],
        tau=config['model']['dqn']['tau'],
        buffer_size=config['training']['replay_buffer_size'],
        batch_size=config['training']['batch_size'],
        reward_model_type=config['model']['reward_net']['model_type'],
        reward_labels=config['training']['reward_labels'],
        sync_steps=1,  # Default value
        update_steps=1,  # Default value
        device=device,
        pretrained_reward_path=pretrained_reward_path if pretrained_reward_exists else None,
        reward_net_num_layers=config['model']['reward_net'].get('num_layers', 2),
        reward_net_dropout=config['model']['reward_net'].get('dropout', 0.1),
        reward_update_interval=config['training'].get('reward_update_interval', 50),
        reward_batch_multiplier=config['training'].get('reward_batch_multiplier', 4),
        reward_warmup_steps=config['training'].get('reward_warmup_steps', 500),
        max_reward_train_per_episode=config['training'].get('max_reward_train_per_episode', 5)
        ,num_envs=num_envs
    )
    
    # Train or load model  
    model_path = os.path.join('models/saved', 'srddqn_model.pth')
    
    # Run pre-training if requested
    if args.pretrain:
        print("\nRunning Phase 1: Pre-training reward network...")
        from src.pretrain_reward_network import pretrain_reward_network
        
        # Use training data for pre-training
        pretrained_reward_network = pretrain_reward_network(config, train_data, pretrained_reward_path)
        print("Phase 1 completed: Reward network pre-trained!")
    
    if args.train:
        print("\nPhase 2: Training SRDDQN agent with self-rewarding mechanism...")
        train_metrics = train_srddqn(
            agent=agent,
            train_env=train_env,
            val_env=val_env,
            config=config,
            model_path=model_path
        )
        
        # Plot comprehensive training curves
        plt.figure(figsize=(20, 15))

        # Row 1: Core RL Metrics
        plt.subplot(3, 4, 1)
        plt.plot(train_metrics['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(3, 4, 2)
        plt.plot(train_metrics['dqn_losses'])
        plt.title('DQN Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.subplot(3, 4, 3)
        plt.plot(train_metrics['reward_net_losses'])
        plt.title('Reward Network Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.subplot(3, 4, 4)
        plt.plot(train_metrics.get('episode_epsilons', []))
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon (actual)')

        # Row 2: Financial Performance Metrics
        plt.subplot(3, 4, 5)
        plt.plot(train_metrics['portfolio_values'])
        plt.title('Portfolio Values')
        plt.xlabel('Episode')
        plt.ylabel('Portfolio Value ($)')

        plt.subplot(3, 4, 6)
        plt.plot(train_metrics['episode_pnls'])
        plt.title('Episode PnL (%)')
        plt.xlabel('Episode')
        plt.ylabel('PnL')

        plt.subplot(3, 4, 7)
        plt.plot(train_metrics['episode_win_rates'])
        plt.title('Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')

        plt.subplot(3, 4, 8)
        plt.plot(train_metrics['episode_total_trades'])
        plt.title('Total Trades per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Trades')

        # Row 3: Risk and Performance Metrics
        plt.subplot(3, 4, 9)
        plt.plot(train_metrics['episode_sharpe_ratios'])
        plt.title('Sharpe Ratio')
        plt.xlabel('Episode')
        plt.ylabel('Sharpe Ratio')

        plt.subplot(3, 4, 10)
        plt.plot(train_metrics['episode_max_drawdowns'])
        plt.title('Maximum Drawdown')
        plt.xlabel('Episode')
        plt.ylabel('Max Drawdown')

        # Create summary statistics plot
        plt.subplot(3, 4, 11)
        metrics_summary = {
            'Avg Reward': np.mean(train_metrics['episode_rewards']),
            'Avg PnL (%)': np.mean(train_metrics['episode_pnls']) * 100,
            'Avg Win Rate': np.mean(train_metrics['episode_win_rates']),
            'Total Trades': np.sum(train_metrics['episode_total_trades']),
            'Final Portfolio': train_metrics['portfolio_values'][-1],
            'Avg Sharpe': np.mean(train_metrics['episode_sharpe_ratios'])
        }

        plt.bar(range(len(metrics_summary)), list(metrics_summary.values()))
        plt.xticks(range(len(metrics_summary)), list(metrics_summary.keys()), rotation=45, ha='right')
        plt.title('Training Summary Statistics')
        plt.ylabel('Value')

        # Portfolio growth over time
        plt.subplot(3, 4, 12)
        cumulative_portfolio = np.cumprod(1 + np.array(train_metrics['episode_pnls']))
        plt.plot(cumulative_portfolio)
        plt.title('Cumulative Portfolio Growth')
        plt.xlabel('Episode')
        plt.ylabel('Growth Factor')

        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        print("Enhanced training curves saved to results/training_curves.png")

        # Print training summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(train_metrics['episode_rewards'])}")
        print(f"Average Episode Reward: {np.mean(train_metrics['episode_rewards']):.4f}")
        print(f"Average PnL: {np.mean(train_metrics['episode_pnls'])*100:.2f}%")
        print(f"Average Win Rate: {np.mean(train_metrics['episode_win_rates'])*100:.1f}%")
        print(f"Total Trades Executed: {np.sum(train_metrics['episode_total_trades'])}")
        print(f"Final Portfolio Value: ${train_metrics['portfolio_values'][-1]:.2f}")
        print(f"Average Sharpe Ratio: {np.mean(train_metrics['episode_sharpe_ratios']):.4f}")
        print(f"Average Max Drawdown: {np.mean(train_metrics['episode_max_drawdowns'])*100:.2f}%")
        print(f"Best Episode PnL: {np.max(train_metrics['episode_pnls'])*100:.2f}%")
        print(f"Worst Episode PnL: {np.min(train_metrics['episode_pnls'])*100:.2f}%")
        print("="*60)
    elif os.path.exists(f"{model_path}_dqn.pth") and os.path.exists(f"{model_path}_reward.pth"):
        print(f"\nLoading model from {model_path}")
        agent.load(model_path)
    else:
        print(f"\nNo model found at {model_path}. Use --train flag to train a new model.")
        return
    
    # Evaluate model
    if args.evaluate:
        print("\nEvaluating SRDDQN agent...")
        
        # Create evaluation config from main config
        eval_config = {
            'hidden_size': config['model']['dqn']['hidden_size'],
            'learning_rate': config['model']['dqn']['learning_rate'],
            'gamma': config['model']['dqn']['gamma'],
            'epsilon_decay': config['model']['dqn']['epsilon_decay'],
            'initial_capital': config['environment'].get('initial_capital', 500000),
            'transaction_cost': config['environment'].get('transaction_cost', 0.003),
            'window_size': config['environment']['window_size']
        }
        
        eval_metrics = evaluate_agent(
            model_path='models/saved/srddqn_model.pth',
            data_path='data/processed/test_data.csv',
            config=eval_config,
            results_dir='results'
        )
        
        # Also compare with paper benchmarks
        from src.evaluate import compare_with_baselines
        compare_with_baselines(eval_metrics, eval_config)
    
    # Run baseline comparisons
    if args.run_baselines:
        print("\nRunning baseline comparisons...")
        baseline_results = train_and_evaluate_baselines(
            config_path=args.config,
            test_data_path='data/processed/test_data.csv',
            models_dir='models/saved'
        )
        
        print("\nBaseline comparison completed. Results saved to results/ directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SRDDQN Trading System')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml', help='Path to configuration file')
    parser.add_argument('--download_data', action='store_true', help='Download and process new data')
    parser.add_argument('--parquet_file', type=str, help='Path to parquet file to use instead of downloading data')
    parser.add_argument('--resample', type=str, help='Resample timeframe (e.g., "5T" for 5min, "15T" for 15min, "1H" for 1hour)')
    parser.add_argument('--pretrain', action='store_true', help='Run Phase 1: Pre-train the reward network')
    parser.add_argument('--train', action='store_true', help='Run Phase 2: Train the SRDDQN agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the SRDDQN agent')
    parser.add_argument('--run_baselines', action='store_true', help='Run baseline model comparisons')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--multi_index', action='store_true', help='Train on all six indices (DJI, IXIC, SP500, HSI, FCHI, KS11)')
    parser.add_argument('--debug-sync', action='store_true', help='Use in-process SyncVectorEnv for debugging on Windows')
    parser.add_argument('--num-envs', type=int, default=None, help='Override number of environments for vectorization')
    
    args = parser.parse_args()
    
    main(args)