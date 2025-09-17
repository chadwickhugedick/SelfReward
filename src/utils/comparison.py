"""
Comparison utilities for model evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


def compare_models(results: Dict[str, Dict[str, float]], save_path: str = None):
    """
    Compare model performance across different metrics.
    
    Args:
        results (dict): Dictionary of model results
        save_path (str): Path to save comparison plot
    """
    if not results:
        print("No results to compare")
        return
    
    # Convert results to DataFrame for easier plotting
    df_data = []
    for model_name, metrics in results.items():
        for metric, value in metrics.items():
            df_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': value
            })
    
    df = pd.DataFrame(df_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Plot each metric
    metrics = df['Metric'].unique()
    for i, metric in enumerate(metrics[:4]):  # Plot up to 4 metrics
        row = i // 2
        col = i % 2
        
        metric_data = df[df['Metric'] == metric]
        axes[row, col].bar(metric_data['Model'], metric_data['Value'])
        axes[row, col].set_title(f'{metric} Comparison')
        axes[row, col].set_ylabel(metric)
        axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_portfolio_values(portfolio_data: Dict[str, List[float]], save_path: str = None):
    """
    Plot portfolio value evolution for different models.
    
    Args:
        portfolio_data (dict): Dictionary with model names as keys and portfolio values as lists
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, values in portfolio_data.items():
        plt.plot(values, label=model_name, linewidth=2)
    
    plt.title('Portfolio Value Evolution Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio evolution plot saved to {save_path}")
    
    plt.show()


def plot_action_distributions(action_data: Dict[str, List[int]], save_path: str = None):
    """
    Plot action distribution for different models.
    
    Args:
        action_data (dict): Dictionary with model names as keys and action lists as values
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, len(action_data), figsize=(5 * len(action_data), 6))
    if len(action_data) == 1:
        axes = [axes]
    
    action_names = ['Hold', 'Buy', 'Sell']
    
    for i, (model_name, actions) in enumerate(action_data.items()):
        action_counts = np.bincount(actions, minlength=3)
        action_percentages = action_counts / len(actions) * 100
        
        axes[i].pie(action_percentages, labels=action_names, autopct='%1.1f%%')
        axes[i].set_title(f'{model_name} Action Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action distribution plot saved to {save_path}")
    
    plt.show()


def calculate_metrics(portfolio_values: List[float], returns: List[float] = None):
    """
    Calculate performance metrics from portfolio values.
    
    Args:
        portfolio_values (list): List of portfolio values over time
        returns (list): List of returns (optional, calculated if not provided)
    
    Returns:
        dict: Dictionary of calculated metrics
    """
    if len(portfolio_values) < 2:
        return {}
    
    # Calculate returns if not provided
    if returns is None:
        returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                  for i in range(1, len(portfolio_values))]
    
    if len(returns) == 0:
        return {}
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    avg_return = np.mean(returns)
    volatility = np.std(returns)
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'Total Return': total_return,
        'Average Return': avg_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print a formatted comparison table of results.
    
    Args:
        results (dict): Dictionary of model results
    """
    if not results:
        print("No results to display")
        return
    
    # Get all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    
    # Create comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Header
    header = f"{'Model':<20}"
    for metric in sorted(all_metrics):
        header += f"{metric:<15}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for model_name, metrics in results.items():
        row = f"{model_name:<20}"
        for metric in sorted(all_metrics):
            value = metrics.get(metric, 0)
            if isinstance(value, float):
                row += f"{value:<15.4f}"
            else:
                row += f"{value:<15}"
        print(row)
    
    print("="*80)