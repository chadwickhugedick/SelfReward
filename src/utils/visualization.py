import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

def plot_portfolio_performance(portfolio_values, benchmark_values=None, save_path=None):
    """Plot portfolio performance over time.
    
    Args:
        portfolio_values (list): List of portfolio values over time
        benchmark_values (list, optional): List of benchmark values for comparison
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value', linewidth=2)
    
    if benchmark_values is not None:
        plt.plot(benchmark_values, label='Benchmark', linewidth=2, linestyle='--')
    
    plt.title('Portfolio Performance', fontsize=16)
    plt.xlabel('Trading Steps', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trading_actions(actions_history, prices, save_path=None):
    """Plot trading actions (buy/sell/hold) along with price movement.
    
    Args:
        actions_history (list): List of actions taken (0: Sell, 1: Hold, 2: Buy)
        prices (list): List of asset prices
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(prices, color='black', alpha=0.6, label='Price')
    
    # Find buy and sell points
    buys = [i for i, a in enumerate(actions_history) if a == 2]
    sells = [i for i, a in enumerate(actions_history) if a == 0]
    holds = [i for i, a in enumerate(actions_history) if a == 1]
    
    # Plot actions
    if buys:
        plt.scatter(buys, [prices[i] for i in buys], marker='^', color='green', s=100, label='Buy')
    if sells:
        plt.scatter(sells, [prices[i] for i in sells], marker='v', color='red', s=100, label='Sell')
    
    plt.title('Trading Actions and Price Movement', fontsize=16)
    plt.xlabel('Trading Steps', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reward_distribution(rewards_history, save_path=None):
    """Plot the distribution of rewards.
    
    Args:
        rewards_history (list): List of rewards received
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(rewards_history, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axvline(x=np.mean(rewards_history), color='g', linestyle='-', label=f'Mean: {np.mean(rewards_history):.4f}')
    
    plt.title('Reward Distribution', fontsize=16)
    plt.xlabel('Reward', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_metrics(metrics, save_path=None):
    """Plot training metrics over time.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio values
    plt.subplot(2, 2, 2)
    plt.plot(metrics['portfolio_values'])
    plt.title('Portfolio Values', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot DQN losses
    plt.subplot(2, 2, 3)
    plt.plot(metrics['dqn_losses'])
    plt.title('DQN Losses', fontsize=14)
    plt.xlabel('Update Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot reward network losses
    plt.subplot(2, 2, 4)
    plt.plot(metrics['reward_net_losses'])
    plt.title('Reward Network Losses', fontsize=14)
    plt.xlabel('Update Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_action_distribution(actions_history, save_path=None):
    """Plot the distribution of actions taken.
    
    Args:
        actions_history (list): List of actions taken (0: Sell, 1: Hold, 2: Buy)
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    action_counts = {
        'Sell': actions_history.count(0),
        'Hold': actions_history.count(1),
        'Buy': actions_history.count(2)
    }
    
    # Calculate percentages
    total_actions = len(actions_history)
    action_percentages = {k: (v / total_actions) * 100 for k, v in action_counts.items()}
    
    # Plot
    bars = plt.bar(action_percentages.keys(), action_percentages.values(), color=['red', 'gray', 'green'])
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Action Distribution', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_portfolio_metrics(metrics, save_path=None):
    """Plot portfolio performance metrics.
    
    Args:
        metrics (dict): Dictionary containing portfolio metrics
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for key metrics
    metric_names = ['Sharpe Ratio', 'Max Drawdown', 'Total Return (%)']
    metric_values = [
        metrics['sharpe_ratio'],
        metrics['max_drawdown'],
        (metrics['final_portfolio_value'] / metrics['initial_portfolio_value'] - 1) * 100
    ]
    
    colors = ['green', 'red', 'blue']
    
    bars = plt.bar(metric_names, metric_values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Portfolio Performance Metrics', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reward_components(reward_components, save_path=None):
    """Plot the components of the reward function.
    
    Args:
        reward_components (dict): Dictionary containing reward components over time
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    for component, values in reward_components.items():
        plt.plot(values, label=component)
    
    plt.title('Reward Components Over Time', fontsize=16)
    plt.xlabel('Trading Steps', fontsize=12)
    plt.ylabel('Reward Component Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_q_values(q_values_history, save_path=None):
    """Plot the Q-values for each action over time.
    
    Args:
        q_values_history (list): List of Q-values for each action at each step
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(14, 8))
    
    # Extract Q-values for each action
    q_sell = [q[0] for q in q_values_history]
    q_hold = [q[1] for q in q_values_history]
    q_buy = [q[2] for q in q_values_history]
    
    plt.plot(q_sell, label='Q(Sell)', color='red')
    plt.plot(q_hold, label='Q(Hold)', color='gray')
    plt.plot(q_buy, label='Q(Buy)', color='green')
    
    plt.title('Q-Values Over Time', fontsize=16)
    plt.xlabel('Trading Steps', fontsize=12)
    plt.ylabel('Q-Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_names, importance_scores, save_path=None):
    """Plot feature importance scores.
    
    Args:
        feature_names (list): List of feature names
        importance_scores (list): List of importance scores for each feature
        save_path (str, optional): Path to save the figure
    """
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_scores = [importance_scores[i] for i in sorted_idx]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(sorted_names, sorted_scores, color='skyblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                 f'{width:.4f}', ha='left', va='center')
    
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_reward_network_predictions(actual_rewards, predicted_rewards, save_path=None):
    """Plot actual vs predicted rewards from the reward network.
    
    Args:
        actual_rewards (list): List of actual rewards
        predicted_rewards (list): List of rewards predicted by the reward network
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Plot actual and predicted rewards
    plt.scatter(actual_rewards, predicted_rewards, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(min(actual_rewards), min(predicted_rewards))
    max_val = max(max(actual_rewards), max(predicted_rewards))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate and display correlation
    correlation = np.corrcoef(actual_rewards, predicted_rewards)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title('Reward Network Predictions vs Actual Rewards', fontsize=16)
    plt.xlabel('Actual Rewards', fontsize=12)
    plt.ylabel('Predicted Rewards', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_performance_dashboard(metrics, trading_history, save_dir=None):
    """Create a comprehensive performance dashboard with multiple plots.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        trading_history (dict): Dictionary containing trading history data
        save_dir (str, optional): Directory to save the figures
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot portfolio performance
    plot_portfolio_performance(
        trading_history['portfolio_values'],
        trading_history.get('benchmark_values'),
        save_path=os.path.join(save_dir, 'portfolio_performance.png') if save_dir else None
    )
    
    # Plot trading actions
    plot_trading_actions(
        trading_history['actions'],
        trading_history['prices'],
        save_path=os.path.join(save_dir, 'trading_actions.png') if save_dir else None
    )
    
    # Plot reward distribution
    plot_reward_distribution(
        trading_history['rewards'],
        save_path=os.path.join(save_dir, 'reward_distribution.png') if save_dir else None
    )
    
    # Plot action distribution
    plot_action_distribution(
        trading_history['actions'],
        save_path=os.path.join(save_dir, 'action_distribution.png') if save_dir else None
    )
    
    # Plot portfolio metrics
    plot_portfolio_metrics(
        metrics,
        save_path=os.path.join(save_dir, 'portfolio_metrics.png') if save_dir else None
    )
    
    # Plot reward components if available
    if 'reward_components' in trading_history:
        plot_reward_components(
            trading_history['reward_components'],
            save_path=os.path.join(save_dir, 'reward_components.png') if save_dir else None
        )
    
    # Plot Q-values if available
    if 'q_values' in trading_history:
        plot_q_values(
            trading_history['q_values'],
            save_path=os.path.join(save_dir, 'q_values.png') if save_dir else None
        )
    
    # Plot reward network predictions if available
    if 'actual_rewards' in trading_history and 'predicted_rewards' in trading_history:
        plot_reward_network_predictions(
            trading_history['actual_rewards'],
            trading_history['predicted_rewards'],
            save_path=os.path.join(save_dir, 'reward_predictions.png') if save_dir else None
        )