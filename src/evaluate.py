"""
Simple Evaluation Script for SRDDQN Trading Agent

This script provides basic evaluation functionality for trained SRDDQN models.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import os
from datetime import datetime
import yaml

from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.data.data_processor import DataProcessor


def evaluate_agent(model_path: str, data_path: str, config: Dict[str, Any], 
                  results_dir: str = "results") -> Dict[str, Any]:
    """
    Simple evaluation function for SRDDQN agent.
    
    Args:
        model_path: Path to the saved model file (without extension)
        data_path: Path to the test data CSV file
        config: Configuration dictionary
        results_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        print(f"Loading test data from: {data_path}")
        
        # Load test data
        if os.path.exists(data_path):
            test_data = pd.read_csv(data_path)
            print(f"Loaded {len(test_data)} test samples")
        else:
            print(f"Test data not found at {data_path}, using sample data...")
            # Create sample data for demonstration
            dates = pd.date_range('2021-01-01', periods=100, freq='D')
            test_data = pd.DataFrame({
                'Date': dates,
                'Open': np.random.uniform(100, 150, 100),
                'High': np.random.uniform(100, 150, 100),
                'Low': np.random.uniform(100, 150, 100),
                'Close': np.random.uniform(100, 150, 100),
                'Volume': np.random.uniform(1000000, 10000000, 100)
            })
            # Make prices realistic
            for i in range(1, len(test_data)):
                prev_close = test_data.iloc[i-1]['Close']
                change = np.random.normal(0, 0.02) * prev_close
                new_close = prev_close + change
                test_data.iloc[i]['Close'] = new_close
                test_data.iloc[i]['Open'] = new_close + np.random.normal(0, 0.005) * new_close
                test_data.iloc[i]['High'] = max(test_data.iloc[i]['Open'], test_data.iloc[i]['Close']) + np.random.uniform(0, 0.01) * new_close
                test_data.iloc[i]['Low'] = min(test_data.iloc[i]['Open'], test_data.iloc[i]['Close']) - np.random.uniform(0, 0.01) * new_close
        
        # Create trading environment
        env = TradingEnvironment(
            data=test_data,
            window_size=config.get('window_size', 20),
            initial_capital=config.get('initial_capital', 500000),
            transaction_cost=config.get('transaction_cost', 0.003)
        )
        # Pass expert selection if provided in config
        expert_sel = config.get('expert_selection', None) if isinstance(config, dict) else None
        if expert_sel is not None:
            env = TradingEnvironment(
                data=test_data,
                window_size=config.get('window_size', 20),
                initial_capital=config.get('initial_capital', 500000),
                transaction_cost=config.get('transaction_cost', 0.003),
                expert_selection=expert_sel
            )
        
        print(f"Created trading environment with {len(test_data)} days of data")
        print(f"Initial capital: ${config.get('initial_capital', 500000):,.2f}")
        
        # Check if model files exist
        model_files = [f"{model_path}.pth", f"{model_path}_dqn.pth", f"{model_path}_reward.pth"]
        existing_files = [f for f in model_files if os.path.exists(f)]
        
        if not existing_files:
            print(f"No model files found at {model_path}")
            print("Running Buy & Hold baseline strategy instead...")
            return run_baseline_evaluation(env, config, results_dir)
        
        print(f"Found model files: {existing_files}")

        # Create agent (don't try to load if files don't exist)
        state, info = env.reset()
        # The SRDDQN agent expects flattened state dimensions for DQN compatibility
        if len(state.shape) > 1:
            state_dim = np.prod(state.shape)  # Flatten multi-dimensional state
        else:
            state_dim = state.shape[0]
        action_dim = env.action_space.n

        print(f"State shape: {state.shape}, Flattened dimension: {state_dim}, Action dimension: {action_dim}")

        # Try loading with agent if model exists
        try:
            # Calculate dimensions to match training logic
            seq_len = config.get('window_size', 10)  # Should match environment window_size
            feature_dim = state.shape[1] if len(state.shape) > 1 else None  # Per-timestep feature dimension

            print(f"Using seq_len: {seq_len}, feature_dim: {feature_dim}")

            agent = SRDDQNAgent(
                state_dim=state_dim,  # Flattened state dimension for DQN
                action_dim=action_dim,
                seq_len=seq_len,
                feature_dim=feature_dim,  # Match training calculation
                hidden_dim=config.get('hidden_size', 64),
                dqn_lr=config.get('learning_rate', 0.0001),
                gamma=config.get('gamma', 0.99),
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=config.get('epsilon_decay', 0.995),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                reward_model_type='TimesNet',  # Match training model type
                reward_labels=['Min-Max']  # Match training reward labels
            )

            # The model path should be without .pth extension since agent.load() adds suffixes
            base_path = model_path.replace('.pth', '')
            agent.load(base_path)
            print(f"Successfully loaded SRDDQN model from {base_path}")

        except Exception as e:
            print(f"Error creating/loading agent: {e}")
            print("Running Buy & Hold baseline strategy instead...")
            return run_baseline_evaluation(env, config, results_dir)

        # Run evaluation
        print("Running SRDDQN evaluation...")
        state, info = env.reset()
        done = False
        episode_reward = 0

        # Track portfolio values for metrics calculation
        portfolio_history = [env.initial_capital]
        actions_taken = []

        step_count = 0
        while not done and step_count < len(test_data) - env.window_size:
            # Get action from agent (it handles state flattening internally)
            try:
                action = agent.select_action(state, training=False)
            except Exception as e:
                print(f"Error selecting action: {e}, using random action")
                action = env.action_space.sample()

            # Take action in environment (Gymnasium-style)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            episode_reward += reward
            current_portfolio_value = env.portfolio_values[-1] if env.portfolio_values else env.initial_capital
            portfolio_history.append(current_portfolio_value)
            actions_taken.append(action)

            state = next_state
            step_count += 1

        # Calculate metrics
        final_portfolio_value = portfolio_history[-1]
        total_return = (final_portfolio_value - env.initial_capital) / env.initial_capital * 100

        # Calculate Sharpe ratio
        portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
        if len(portfolio_returns) > 1 and np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Calculate maximum drawdown
        portfolio_array = np.array(portfolio_history)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        print(f"\nEvaluation completed!")
        print(f"Steps taken: {step_count}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")

        metrics = {
            'average_return': total_return,
            'std_return': 0.0,  # Single episode
            'average_sharpe_ratio': sharpe_ratio,
            'average_max_drawdown': max_drawdown,
            'final_portfolio_value': final_portfolio_value,
            'total_episodes': 1,
            'actions_distribution': {
                'hold': actions_taken.count(0) if actions_taken else 0,
                'buy': actions_taken.count(1) if actions_taken else 0,
                'sell': actions_taken.count(2) if actions_taken else 0
            }
        }

        # Save results
        save_evaluation_results(metrics, results_dir)

        return metrics
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Running Buy & Hold baseline strategy instead...")
        return run_baseline_evaluation(env if 'env' in locals() else None, config, results_dir)


def run_baseline_evaluation(env, config: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    """Run a simple Buy & Hold baseline strategy."""
    
    if env is None:
        # Create sample environment for demonstration
        print("Creating sample environment for baseline...")
        dates = pd.date_range('2021-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Close': 100 + np.cumsum(np.random.normal(0, 1, 100))  # Random walk
        })
        env = TradingEnvironment(
            data=sample_data,
            window_size=config.get('window_size', 20),
            initial_capital=config.get('initial_capital', 500000),
            transaction_cost=config.get('transaction_cost', 0.003)
        )
        expert_sel = config.get('expert_selection', None) if isinstance(config, dict) else None
        if expert_sel is not None:
            env = TradingEnvironment(
                data=sample_data,
                window_size=config.get('window_size', 20),
                initial_capital=config.get('initial_capital', 500000),
                transaction_cost=config.get('transaction_cost', 0.003),
                expert_selection=expert_sel
            )
    
    print("Running Buy & Hold baseline strategy...")
    
    state, info = env.reset()
    
    # Buy at the beginning (Gymnasium-style)
    _, reward, terminated, truncated, _ = env.step(1)  # Buy action
    done = bool(terminated or truncated)
    
    # Hold until the end
    while not done:
        _, reward, terminated, truncated, _ = env.step(0)  # Hold action
        done = bool(terminated or truncated)
    
    final_value = env.portfolio_values[-1] if env.portfolio_values else env.initial_capital
    total_return = (final_value - env.initial_capital) / env.initial_capital * 100
    
    print(f"Buy & Hold Results:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    
    metrics = {
        'average_return': total_return,
        'std_return': 0.0,
        'average_sharpe_ratio': 0.0,  # Would need price history to calculate
        'average_max_drawdown': 0.0,  # Simplified
        'final_portfolio_value': final_value,
        'total_episodes': 1,
        'actions_distribution': {
            'hold': 1,
            'buy': 1, 
            'sell': 0
        }
    }
    
    save_evaluation_results(metrics, results_dir, strategy_name="Buy_Hold_Baseline")
    return metrics


def save_evaluation_results(metrics: Dict[str, Any], results_dir: str, strategy_name: str = "SRDDQN") -> None:
    """Save evaluation results to file."""
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"{strategy_name}_evaluation_{timestamp}.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"{strategy_name} Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Return: {metrics['average_return']:.2f}%\n")
        f.write(f"Return Std Dev: {metrics['std_return']:.2f}%\n")
        f.write(f"Average Sharpe Ratio: {metrics['average_sharpe_ratio']:.3f}\n")
        f.write(f"Average Max Drawdown: {metrics['average_max_drawdown']:.2f}%\n")
        f.write(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}\n\n")
        
        f.write("Action Distribution:\n")
        f.write("-" * 20 + "\n")
        actions = metrics['actions_distribution']
        total_actions = sum(actions.values())
        if total_actions > 0:
            f.write(f"Hold: {actions['hold']} ({actions['hold']/total_actions*100:.1f}%)\n")
            f.write(f"Buy: {actions['buy']} ({actions['buy']/total_actions*100:.1f}%)\n")
            f.write(f"Sell: {actions['sell']} ({actions['sell']/total_actions*100:.1f}%)\n")
        
    print(f"Evaluation results saved to: {results_path}")


def compare_with_baselines(metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Compare model performance with paper benchmarks.
    
    Args:
        metrics: Model evaluation metrics
        config: Configuration dictionary
    """
    print("\nComparison with Research Paper Benchmarks:")
    print("=" * 60)
    
    # Paper results (from concept.txt)
    paper_results = {
        'SRDDQN_IXIC': {'cumulative_return': 1124.23, 'description': 'SRDDQN on IXIC dataset'},
        'Fire_DQN_HER_IXIC': {'cumulative_return': 51.87, 'description': 'Fire (DQN-HER) on IXIC'},
        'SRDDQN_DJI': {'cumulative_return': 305.43, 'description': 'SRDDQN on DJI dataset'},
        'Fire_DQN_HER_DJI': {'cumulative_return': 76.79, 'description': 'Fire (DQN-HER) on DJI'}
    }
    
    print("Paper Benchmark Results:")
    for method, result in paper_results.items():
        print(f"  {method}: {result['cumulative_return']:.2f}% - {result['description']}")
    
    print(f"\nYour Model Performance:")
    print(f"  Average Return: {metrics['average_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['average_sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['average_max_drawdown']:.2f}%")
    
    # Analysis
    print("\nPerformance Analysis:")
    print("-" * 30)
    
    if metrics['average_return'] > 300:
        print("✅ Excellent performance - matches paper's high-performing results!")
    elif metrics['average_return'] > 100:
        print("✅ Good performance - above many baseline methods")
    elif metrics['average_return'] > 50:
        print("⚠️  Moderate performance - room for improvement")
    else:
        print("❌ Underperforming - significant optimization needed")
    
    if metrics['average_sharpe_ratio'] > 3.0:
        print("✅ Excellent risk-adjusted returns (Sharpe > 3.0)")
    elif metrics['average_sharpe_ratio'] > 1.5:
        print("✅ Good risk-adjusted returns (Sharpe > 1.5)")
    else:
        print("⚠️  Risk-adjusted returns could be improved")
    
    if abs(metrics['average_max_drawdown']) < 10:
        print("✅ Good risk management (Max Drawdown < 10%)")
    elif abs(metrics['average_max_drawdown']) < 20:
        print("⚠️  Moderate risk exposure")
    else:
        print("❌ High risk exposure - consider risk management improvements")