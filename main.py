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
    
    # Process data
    data_processor = DataProcessor(config['data'])

    if args.multi_index:
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
            data = data_processor.download_data(
            ticker=config['data']['ticker'],  # Use ticker from config
            start_date=config['data']['train_start_date'],
            end_date=config['data']['test_end_date']
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

        # Add technical indicators
        data = data_processor.add_technical_indicators(data)

        # Normalize data
        data = data_processor.normalize_data(data)

        # Split data
        train_data, val_data, test_data = data_processor.split_data(data)

        # Save processed data
        train_data.to_csv('data/processed/train_data.csv', index=False)
        val_data.to_csv('data/processed/val_data.csv', index=False)
        test_data.to_csv('data/processed/test_data.csv', index=False)
        print("Processed data saved to data/processed/")
    
    # Create environments
    env_config = config['environment']
    
    train_env = TradingEnvironment(
        data=train_data,
        window_size=env_config['window_size'],
        initial_capital=env_config.get('initial_capital', 500000),
        transaction_cost=env_config.get('transaction_cost', 0.003)
    )
    
    val_env = TradingEnvironment(
        data=val_data,
        window_size=env_config['window_size'],
        initial_capital=env_config.get('initial_capital', 500000),
        transaction_cost=env_config.get('transaction_cost', 0.003)
    )
    
    test_env = TradingEnvironment(
        data=test_data,
        window_size=env_config['window_size'],
        initial_capital=env_config.get('initial_capital', 500000),
        transaction_cost=env_config.get('transaction_cost', 0.003)
    )
    
    # Get state and action dimensions
    state_dim = np.prod(train_env.observation_space.shape)  # Total size for DQN
    feature_dim = train_env.observation_space.shape[1]  # Feature dimension per time step for reward network
    action_dim = train_env.action_space.n
    
    # Create agent
    agent = SRDDQNAgent(
        state_dim=state_dim,  # Flattened state dimension for DQN
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
        buffer_size=config['training']['replay_buffer_size'],
        batch_size=config['training']['batch_size'],
        reward_model_type=config['model']['reward_net']['model_type'],
        reward_labels=config['training']['reward_labels'],
        sync_steps=1,  # Default value
        update_steps=1,  # Default value
        device=device,
        feature_dim=feature_dim  # Feature dimension for reward network
    )
    
    # Train or load model
    model_path = os.path.join('models/saved', 'srddqn_model.pth')
    
    if args.train:
        print("\nTraining SRDDQN agent...")
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
        episodes = range(1, len(train_metrics['episode_rewards']) + 1)
        epsilon_values = [agent.dqn_agent.epsilon_start * (agent.dqn_agent.epsilon_decay ** (episode * 100))
                         for episode in episodes]
        plt.plot(epsilon_values)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')

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
    elif os.path.exists(model_path):
        print(f"\nLoading model from {model_path}")
        agent.load(model_path)
    else:
        print(f"\nNo model found at {model_path}. Use --train flag to train a new model.")
        return
    
    # Evaluate model
    if args.evaluate:
        print("\nEvaluating SRDDQN agent...")
        eval_metrics = evaluate_agent(
            agent=agent,
            env=test_env,
            render=args.render
        )
        
        print("\nEvaluation Results:")
        print(f"Final Portfolio Value: {eval_metrics['final_portfolio_value']:.4f}")
        print(f"Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {eval_metrics['max_drawdown']:.4f}")
        print(f"Total Reward: {eval_metrics['total_reward']:.4f}")
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/evaluation_{timestamp}.txt"
        
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results:\n")
            f.write(f"Final Portfolio Value: {eval_metrics['final_portfolio_value']:.4f}\n")
            f.write(f"Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Max Drawdown: {eval_metrics['max_drawdown']:.4f}\n")
            f.write(f"Total Reward: {eval_metrics['total_reward']:.4f}\n")
        
        print(f"Evaluation results saved to {results_path}")
    
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
    parser.add_argument('--train', action='store_true', help='Train the SRDDQN agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the SRDDQN agent')
    parser.add_argument('--run_baselines', action='store_true', help='Run baseline model comparisons')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--multi_index', action='store_true', help='Train on all six indices (DJI, IXIC, SP500, HSI, FCHI, KS11)')
    
    args = parser.parse_args()
    
    main(args)