import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.train import train_agent
from src.evaluate import evaluate_agent

class SensitivityAnalyzer:
    """Class for analyzing the sensitivity of the SRDDQN model to different hyperparameters."""
    
    def __init__(self, base_config_path, data_path, results_dir='results/sensitivity_analysis'):
        """Initialize the sensitivity analyzer.
        
        Args:
            base_config_path (str): Path to the base configuration file
            data_path (str): Path to the processed data
            results_dir (str): Directory to save analysis results
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
    
    def analyze_parameter_sensitivity(self, parameter_name, parameter_values, n_runs=3):
        """Analyze the sensitivity of the model to a specific parameter.
        
        Args:
            parameter_name (str): Name of the parameter to analyze (e.g., 'model.dqn.learning_rate')
            parameter_values (list): List of values to test for the parameter
            n_runs (int): Number of runs for each parameter value
            
        Returns:
            pd.DataFrame: Results of the sensitivity analysis
        """
        results = []
        
        for value in parameter_values:
            print(f"\nTesting {parameter_name} = {value}")
            
            for run in range(n_runs):
                print(f"Run {run+1}/{n_runs}")
                
                # Create configuration for this run
                config = self.base_config.copy()
                
                # Update configuration with current parameter value
                key_path = parameter_name.split('.')
                current = config
                for key in key_path[:-1]:
                    current = current[key]
                current[key_path[-1]] = value
                
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
                model_path = os.path.join(self.results_dir, f"{parameter_name.replace('.', '_')}_{value}_run_{run}.pth")
                
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
                    'parameter_name': parameter_name,
                    'parameter_value': value,
                    'run': run,
                    'final_portfolio_value': eval_metrics['final_portfolio_value'],
                    'sharpe_ratio': eval_metrics['sharpe_ratio'],
                    'max_drawdown': eval_metrics['max_drawdown'],
                    'total_reward': eval_metrics['total_reward'],
                    'model_path': model_path
                }
                
                results.append(result)
                
                # Save current results
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(self.results_dir, f"{parameter_name.replace('.', '_')}_sensitivity.csv"), index=False)
        
        return pd.DataFrame(results)
    
    def analyze_multiple_parameters(self, parameters_dict, n_runs=3):
        """Analyze the sensitivity of the model to multiple parameters.
        
        Args:
            parameters_dict (dict): Dictionary mapping parameter names to lists of values
            n_runs (int): Number of runs for each parameter value
            
        Returns:
            dict: Dictionary mapping parameter names to DataFrames with results
        """
        all_results = {}
        
        for param_name, param_values in parameters_dict.items():
            print(f"\nAnalyzing sensitivity to {param_name}")
            results = self.analyze_parameter_sensitivity(param_name, param_values, n_runs)
            all_results[param_name] = results
        
        return all_results
    
    def visualize_sensitivity(self, results_df, parameter_name, metric='sharpe_ratio', save_path=None):
        """Visualize the sensitivity of the model to a specific parameter.
        
        Args:
            results_df (pd.DataFrame): Results of the sensitivity analysis
            parameter_name (str): Name of the parameter analyzed
            metric (str): Metric to visualize (default: 'sharpe_ratio')
            save_path (str, optional): Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Group by parameter value and calculate statistics
        grouped = results_df.groupby('parameter_value')
        mean_values = grouped[metric].mean()
        std_values = grouped[metric].std()
        
        # Plot mean and standard deviation
        x = mean_values.index
        y = mean_values.values
        yerr = std_values.values
        
        plt.errorbar(x, y, yerr=yerr, marker='o', linestyle='-', capsize=5, markersize=8)
        
        # Add individual points
        for value in results_df['parameter_value'].unique():
            points = results_df[results_df['parameter_value'] == value][metric]
            plt.scatter([value] * len(points), points, alpha=0.5, color='gray')
        
        # Format plot
        plt.xlabel(parameter_name.split('.')[-1])
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Sensitivity of {metric.replace("_", " ").title()} to {parameter_name.split(".")[-1]}')
        plt.grid(True, alpha=0.3)
        
        # Add best value annotation
        best_value_idx = mean_values.idxmax() if metric != 'max_drawdown' else mean_values.idxmin()
        best_value = best_value_idx
        best_metric = mean_values[best_value_idx]
        
        plt.annotate(f'Best value: {best_value}\n{metric}: {best_metric:.4f}',
                    xy=(best_value, best_metric),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def visualize_all_sensitivities(self, all_results, metrics=['sharpe_ratio', 'final_portfolio_value', 'max_drawdown']):
        """Visualize the sensitivity of the model to all analyzed parameters.
        
        Args:
            all_results (dict): Dictionary mapping parameter names to DataFrames with results
            metrics (list): List of metrics to visualize
        """
        for param_name, results_df in all_results.items():
            for metric in metrics:
                save_path = os.path.join(self.results_dir, f"{param_name.replace('.', '_')}_{metric}.png")
                self.visualize_sensitivity(results_df, param_name, metric, save_path)
    
    def create_sensitivity_heatmap(self, param1_name, param1_values, param2_name, param2_values, metric='sharpe_ratio', n_runs=3):
        """Create a heatmap showing the sensitivity of the model to two parameters.
        
        Args:
            param1_name (str): Name of the first parameter
            param1_values (list): List of values for the first parameter
            param2_name (str): Name of the second parameter
            param2_values (list): List of values for the second parameter
            metric (str): Metric to visualize (default: 'sharpe_ratio')
            n_runs (int): Number of runs for each parameter combination
            
        Returns:
            np.ndarray: Heatmap data
        """
        # Initialize heatmap data
        heatmap_data = np.zeros((len(param1_values), len(param2_values)))
        
        # Run experiments for all combinations
        results = []
        
        for i, value1 in enumerate(param1_values):
            for j, value2 in enumerate(param2_values):
                print(f"\nTesting {param1_name} = {value1}, {param2_name} = {value2}")
                
                metric_values = []
                
                for run in range(n_runs):
                    print(f"Run {run+1}/{n_runs}")
                    
                    # Create configuration for this run
                    config = self.base_config.copy()
                    
                    # Update configuration with current parameter values
                    key_path1 = param1_name.split('.')
                    current = config
                    for key in key_path1[:-1]:
                        current = current[key]
                    current[key_path1[-1]] = value1
                    
                    key_path2 = param2_name.split('.')
                    current = config
                    for key in key_path2[:-1]:
                        current = current[key]
                    current[key_path2[-1]] = value2
                    
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
                    model_path = os.path.join(
                        self.results_dir, 
                        f"{param1_name.replace('.', '_')}_{value1}_{param2_name.replace('.', '_')}_{value2}_run_{run}.pth"
                    )
                    
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
                        param1_name: value1,
                        param2_name: value2,
                        'run': run,
                        'final_portfolio_value': eval_metrics['final_portfolio_value'],
                        'sharpe_ratio': eval_metrics['sharpe_ratio'],
                        'max_drawdown': eval_metrics['max_drawdown'],
                        'total_reward': eval_metrics['total_reward'],
                        'model_path': model_path
                    }
                    
                    results.append(result)
                    metric_values.append(eval_metrics[metric])
                
                # Calculate mean metric value for this combination
                mean_value = np.mean(metric_values)
                heatmap_data[i, j] = mean_value
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(
                self.results_dir, 
                f"{param1_name.replace('.', '_')}_{param2_name.replace('.', '_')}_heatmap.csv"
            ), 
            index=False
        )
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 10))
        
        # Determine vmin and vmax based on metric
        if metric == 'max_drawdown':
            # For max_drawdown, lower is better
            vmin, vmax = heatmap_data.min(), heatmap_data.max()
            cmap = 'RdYlGn_r'  # Reversed colormap for max_drawdown
        else:
            # For other metrics, higher is better
            vmin, vmax = heatmap_data.min(), heatmap_data.max()
            cmap = 'RdYlGn'
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.4f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            xticklabels=param2_values,
            yticklabels=param1_values
        )
        
        plt.xlabel(param2_name.split('.')[-1])
        plt.ylabel(param1_name.split('.')[-1])
        plt.title(f'Sensitivity of {metric.replace("_", " ").title()} to {param1_name.split(".")[-1]} and {param2_name.split(".")[-1]}')
        
        # Save heatmap
        plt.savefig(
            os.path.join(
                self.results_dir, 
                f"{param1_name.replace('.', '_')}_{param2_name.replace('.', '_')}_{metric}_heatmap.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )
        
        return heatmap_data

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity analysis for SRDDQN')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml', help='Path to base configuration file')
    parser.add_argument('--data', type=str, default='data/processed/market_data.csv', help='Path to processed data')
    parser.add_argument('--results_dir', type=str, default='results/sensitivity_analysis', help='Directory to save analysis results')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs for each parameter value')
    parser.add_argument('--analysis_type', type=str, choices=['single', 'multiple', 'heatmap'], default='single', help='Type of sensitivity analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(
        base_config_path=args.config,
        data_path=args.data,
        results_dir=args.results_dir
    )
    
    if args.analysis_type == 'single':
        # Analyze sensitivity to a single parameter
        parameter_name = 'model.dqn.learning_rate'
        parameter_values = [0.0001, 0.0005, 0.001, 0.005]
        
        results = analyzer.analyze_parameter_sensitivity(parameter_name, parameter_values, args.n_runs)
        
        # Visualize results
        for metric in ['sharpe_ratio', 'final_portfolio_value', 'max_drawdown']:
            save_path = os.path.join(args.results_dir, f"{parameter_name.replace('.', '_')}_{metric}.png")
            analyzer.visualize_sensitivity(results, parameter_name, metric, save_path)
    
    elif args.analysis_type == 'multiple':
        # Analyze sensitivity to multiple parameters
        parameters_dict = {
            'model.dqn.learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'model.dqn.gamma': [0.9, 0.95, 0.99],
            'model.reward_net.feature_extractor': ['timesnet', 'wftnet', 'nlinear', 'nlinear_attention'],
            'environment.reward_type': ['minmax_return_sharpe', 'return_sharpe', 'minmax_return']
        }
        
        all_results = analyzer.analyze_multiple_parameters(parameters_dict, args.n_runs)
        
        # Visualize all results
        analyzer.visualize_all_sensitivities(all_results)
    
    elif args.analysis_type == 'heatmap':
        # Create heatmap for two parameters
        param1_name = 'model.dqn.learning_rate'
        param1_values = [0.0001, 0.0005, 0.001]
        
        param2_name = 'model.dqn.gamma'
        param2_values = [0.9, 0.95, 0.99]
        
        for metric in ['sharpe_ratio', 'final_portfolio_value', 'max_drawdown']:
            analyzer.create_sensitivity_heatmap(
                param1_name, param1_values,
                param2_name, param2_values,
                metric, args.n_runs
            )
    
    print("\nSensitivity analysis completed!")
    print(f"Results saved to {args.results_dir}")