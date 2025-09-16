import os
import yaml
import numpy as np
import pandas as pd
import torch
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.train import train_agent
from src.evaluate import evaluate_agent

class HyperparameterTuner:
    """Class for tuning hyperparameters of the SRDDQN model."""
    
    def __init__(self, base_config_path, data_path, results_dir='results/hyperparameter_tuning'):
        """Initialize the hyperparameter tuner.
        
        Args:
            base_config_path (str): Path to the base configuration file
            data_path (str): Path to the processed data
            results_dir (str): Directory to save tuning results
        """
        self.base_config_path = base_config_path
        self.data_path = data_path
        self.results_dir = results_dir
        
        # Load base configuration
        with open(base_config_path, 'r') as file:
            self.base_config = yaml.safe_load(file)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.data = pd.read_csv(data_path)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.base_config['data'])
        
        # Split data
        self.train_data, self.val_data, self.test_data = self.data_processor.split_data(self.data)
    
    def create_environments(self, env_config):
        """Create training, validation, and test environments.
        
        Args:
            env_config (dict): Environment configuration
            
        Returns:
            tuple: (train_env, val_env, test_env)
        """
        train_env = TradingEnvironment(
            data=self.train_data,
            window_size=env_config['window_size'],
            commission=env_config['commission'],
            initial_balance=env_config['initial_balance'],
            reward_scaling=env_config['reward_scaling'],
            reward_type=env_config['reward_type']
        )
        
        val_env = TradingEnvironment(
            data=self.val_data,
            window_size=env_config['window_size'],
            commission=env_config['commission'],
            initial_balance=env_config['initial_balance'],
            reward_scaling=env_config['reward_scaling'],
            reward_type=env_config['reward_type']
        )
        
        test_env = TradingEnvironment(
            data=self.test_data,
            window_size=env_config['window_size'],
            commission=env_config['commission'],
            initial_balance=env_config['initial_balance'],
            reward_scaling=env_config['reward_scaling'],
            reward_type=env_config['reward_type']
        )
        
        return train_env, val_env, test_env
    
    def create_agent(self, state_dim, action_dim, reward_net_config, dqn_config):
        """Create an SRDDQN agent.
        
        Args:
            state_dim (int): State dimension
            action_dim (int): Action dimension
            reward_net_config (dict): Reward network configuration
            dqn_config (dict): DQN configuration
            
        Returns:
            SRDDQNAgent: The created agent
        """
        return SRDDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            reward_net_config=reward_net_config,
            dqn_config=dqn_config,
            device=self.device
        )
    
    def grid_search(self, param_grid):
        """Perform grid search over hyperparameters.
        
        Args:
            param_grid (dict): Dictionary of hyperparameter grids
            
        Returns:
            pd.DataFrame: Results of the grid search
        """
        # Generate all combinations of hyperparameters
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        results = []
        
        for i, combination in enumerate(combinations):
            print(f"\nTesting combination {i+1}/{len(combinations)}")
            
            # Create configuration for this combination
            config = self.base_config.copy()
            
            # Update configuration with current hyperparameters
            for k, v in zip(keys, combination):
                # Parse the key path and update nested dictionaries
                key_path = k.split('.')
                current = config
                for key in key_path[:-1]:
                    current = current[key]
                current[key_path[-1]] = v
            
            # Create environments
            train_env, val_env, test_env = self.create_environments(config['environment'])
            
            # Get state and action dimensions
            state_dim = train_env.observation_space.shape[0]
            action_dim = train_env.action_space.n
            
            # Create agent
            agent = self.create_agent(
                state_dim=state_dim,
                action_dim=action_dim,
                reward_net_config=config['model']['reward_net'],
                dqn_config=config['model']['dqn']
            )
            
            # Train agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.results_dir, f"model_{timestamp}.pth")
            
            train_metrics = train_agent(
                agent=agent,
                train_env=train_env,
                val_env=val_env,
                config=config['training'],
                model_path=model_path
            )
            
            # Evaluate agent
            eval_metrics = evaluate_agent(
                agent=agent,
                env=test_env,
                render=False
            )
            
            # Store results
            result = {
                'combination_id': i,
                'final_portfolio_value': eval_metrics['final_portfolio_value'],
                'sharpe_ratio': eval_metrics['sharpe_ratio'],
                'max_drawdown': eval_metrics['max_drawdown'],
                'total_reward': eval_metrics['total_reward'],
                'model_path': model_path
            }
            
            # Add hyperparameters to results
            for k, v in zip(keys, combination):
                result[k] = v
            
            results.append(result)
            
            # Save current results
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.results_dir, 'grid_search_results.csv'), index=False)
            
            # Print current best result
            best_idx = results_df['sharpe_ratio'].idxmax()
            best_result = results_df.iloc[best_idx]
            print(f"\nCurrent best result (combination {best_result['combination_id']})")
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
            print(f"Final Portfolio Value: {best_result['final_portfolio_value']:.4f}")
            print(f"Max Drawdown: {best_result['max_drawdown']:.4f}")
            print(f"Total Reward: {best_result['total_reward']:.4f}")
        
        return pd.DataFrame(results)
    
    def random_search(self, param_distributions, n_iter=10):
        """Perform random search over hyperparameters.
        
        Args:
            param_distributions (dict): Dictionary of hyperparameter distributions
            n_iter (int): Number of random combinations to try
            
        Returns:
            pd.DataFrame: Results of the random search
        """
        results = []
        
        for i in range(n_iter):
            print(f"\nTesting combination {i+1}/{n_iter}")
            
            # Create configuration for this combination
            config = self.base_config.copy()
            
            # Sample random hyperparameters
            sampled_params = {}
            for k, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # Categorical distribution
                    value = np.random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Uniform distribution
                    low, high = distribution
                    if isinstance(low, int) and isinstance(high, int):
                        value = np.random.randint(low, high + 1)
                    else:
                        value = np.random.uniform(low, high)
                else:
                    raise ValueError(f"Unsupported distribution for parameter {k}")
                
                # Parse the key path and update nested dictionaries
                key_path = k.split('.')
                current = config
                for key in key_path[:-1]:
                    current = current[key]
                current[key_path[-1]] = value
                sampled_params[k] = value
            
            # Create environments
            train_env, val_env, test_env = self.create_environments(config['environment'])
            
            # Get state and action dimensions
            state_dim = train_env.observation_space.shape[0]
            action_dim = train_env.action_space.n
            
            # Create agent
            agent = self.create_agent(
                state_dim=state_dim,
                action_dim=action_dim,
                reward_net_config=config['model']['reward_net'],
                dqn_config=config['model']['dqn']
            )
            
            # Train agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.results_dir, f"model_{timestamp}.pth")
            
            train_metrics = train_agent(
                agent=agent,
                train_env=train_env,
                val_env=val_env,
                config=config['training'],
                model_path=model_path
            )
            
            # Evaluate agent
            eval_metrics = evaluate_agent(
                agent=agent,
                env=test_env,
                render=False
            )
            
            # Store results
            result = {
                'combination_id': i,
                'final_portfolio_value': eval_metrics['final_portfolio_value'],
                'sharpe_ratio': eval_metrics['sharpe_ratio'],
                'max_drawdown': eval_metrics['max_drawdown'],
                'total_reward': eval_metrics['total_reward'],
                'model_path': model_path
            }
            
            # Add hyperparameters to results
            for k, v in sampled_params.items():
                result[k] = v
            
            results.append(result)
            
            # Save current results
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.results_dir, 'random_search_results.csv'), index=False)
            
            # Print current best result
            best_idx = results_df['sharpe_ratio'].idxmax()
            best_result = results_df.iloc[best_idx]
            print(f"\nCurrent best result (combination {best_result['combination_id']})")
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
            print(f"Final Portfolio Value: {best_result['final_portfolio_value']:.4f}")
            print(f"Max Drawdown: {best_result['max_drawdown']:.4f}")
            print(f"Total Reward: {best_result['total_reward']:.4f}")
        
        return pd.DataFrame(results)
    
    def visualize_results(self, results_df, save_path=None):
        """Visualize the results of hyperparameter tuning.
        
        Args:
            results_df (pd.DataFrame): Results of the hyperparameter search
            save_path (str, optional): Path to save the visualization
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot Sharpe ratio vs. final portfolio value
        axes[0, 0].scatter(results_df['sharpe_ratio'], results_df['final_portfolio_value'])
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_ylabel('Final Portfolio Value')
        axes[0, 0].set_title('Sharpe Ratio vs. Final Portfolio Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot max drawdown vs. sharpe ratio
        axes[0, 1].scatter(results_df['max_drawdown'], results_df['sharpe_ratio'])
        axes[0, 1].set_xlabel('Max Drawdown')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Max Drawdown vs. Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Find the most important hyperparameters
        # Exclude metrics and paths from correlation analysis
        exclude_cols = ['combination_id', 'final_portfolio_value', 'sharpe_ratio', 
                        'max_drawdown', 'total_reward', 'model_path']
        hyperparam_cols = [col for col in results_df.columns if col not in exclude_cols]
        
        # Calculate correlation with Sharpe ratio
        correlations = []
        for col in hyperparam_cols:
            if results_df[col].dtype in [np.float64, np.int64]:
                corr = results_df[col].corr(results_df['sharpe_ratio'])
                correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Plot top correlations
        if correlations:
            top_n = min(5, len(correlations))
            top_params = [c[0] for c in correlations[:top_n]]
            top_corrs = [c[1] for c in correlations[:top_n]]
            
            axes[1, 0].barh(top_params, top_corrs)
            axes[1, 0].set_xlabel('Correlation with Sharpe Ratio')
            axes[1, 0].set_title('Top Hyperparameter Correlations')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot the most important hyperparameter vs. Sharpe ratio
            if top_params:
                most_important = top_params[0]
                axes[1, 1].scatter(results_df[most_important], results_df['sharpe_ratio'])
                axes[1, 1].set_xlabel(most_important)
                axes[1, 1].set_ylabel('Sharpe Ratio')
                axes[1, 1].set_title(f'{most_important} vs. Sharpe Ratio')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_best_config(self, results_df, metric='sharpe_ratio'):
        """Get the best configuration based on a metric.
        
        Args:
            results_df (pd.DataFrame): Results of the hyperparameter search
            metric (str): Metric to use for ranking (default: 'sharpe_ratio')
            
        Returns:
            dict: Best configuration
        """
        # Find the best result
        if metric == 'max_drawdown':
            best_idx = results_df['max_drawdown'].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        
        best_result = results_df.iloc[best_idx]
        
        # Create configuration for the best result
        best_config = self.base_config.copy()
        
        # Update configuration with best hyperparameters
        exclude_cols = ['combination_id', 'final_portfolio_value', 'sharpe_ratio', 
                        'max_drawdown', 'total_reward', 'model_path']
        hyperparam_cols = [col for col in results_df.columns if col not in exclude_cols]
        
        for k in hyperparam_cols:
            # Parse the key path and update nested dictionaries
            key_path = k.split('.')
            current = best_config
            for key in key_path[:-1]:
                current = current[key]
            current[key_path[-1]] = best_result[k]
        
        return best_config
    
    def save_best_config(self, best_config, save_path):
        """Save the best configuration to a YAML file.
        
        Args:
            best_config (dict): Best configuration
            save_path (str): Path to save the configuration
        """
        with open(save_path, 'w') as file:
            yaml.dump(best_config, file, default_flow_style=False)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SRDDQN')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml', help='Path to base configuration file')
    parser.add_argument('--data', type=str, default='data/processed/market_data.csv', help='Path to processed data')
    parser.add_argument('--results_dir', type=str, default='results/hyperparameter_tuning', help='Directory to save tuning results')
    parser.add_argument('--method', type=str, choices=['grid', 'random'], default='random', help='Search method')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations for random search')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        base_config_path=args.config,
        data_path=args.data,
        results_dir=args.results_dir
    )
    
    if args.method == 'grid':
        # Define parameter grid for grid search
        param_grid = {
            'model.dqn.learning_rate': [0.0001, 0.0005, 0.001],
            'model.dqn.gamma': [0.95, 0.99],
            'model.dqn.hidden_dim': [64, 128, 256],
            'model.reward_net.learning_rate': [0.0001, 0.0005, 0.001],
            'model.reward_net.hidden_dim': [64, 128, 256],
            'model.reward_net.feature_extractor': ['timesnet', 'wftnet', 'nlinear'],
            'environment.reward_type': ['minmax_return_sharpe', 'return_sharpe', 'minmax_return']
        }
        
        # Run grid search
        results = tuner.grid_search(param_grid)
    else:
        # Define parameter distributions for random search
        param_distributions = {
            'model.dqn.learning_rate': (0.00001, 0.001),
            'model.dqn.gamma': (0.9, 0.999),
            'model.dqn.hidden_dim': (32, 256),
            'model.dqn.buffer_size': (5000, 20000),
            'model.dqn.batch_size': [32, 64, 128, 256],
            'model.reward_net.learning_rate': (0.00001, 0.001),
            'model.reward_net.hidden_dim': (32, 256),
            'model.reward_net.feature_extractor': ['timesnet', 'wftnet', 'nlinear', 'nlinear_attention'],
            'environment.reward_type': ['minmax_return_sharpe', 'return_sharpe', 'minmax_return'],
            'environment.commission': (0.0001, 0.003),
            'training.episodes': [100, 200, 300],
            'training.update_target_every': [10, 20, 50]
        }
        
        # Run random search
        results = tuner.random_search(param_distributions, n_iter=args.n_iter)
    
    # Visualize results
    tuner.visualize_results(results, save_path=os.path.join(args.results_dir, 'tuning_results.png'))
    
    # Get and save best configuration
    best_config = tuner.get_best_config(results)
    tuner.save_best_config(best_config, os.path.join(args.results_dir, 'best_config.yaml'))
    
    print("\nHyperparameter tuning completed!")
    print(f"Results saved to {args.results_dir}")