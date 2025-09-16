import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_metrics(results_dir, model_name):
    """Load evaluation metrics for a model."""
    metrics_path = os.path.join(results_dir, model_name, 'evaluation', 'evaluation_metrics.csv')
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    else:
        print(f"Warning: Metrics file not found for {model_name} at {metrics_path}")
        return None

def load_trading_history(results_dir, model_name):
    """Load trading history for a model."""
    history_path = os.path.join(results_dir, model_name, 'evaluation', 'trading_history.csv')
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    else:
        print(f"Warning: Trading history file not found for {model_name} at {history_path}")
        return None

def compare_models(results_dir, model_names):
    """Compare multiple models based on their evaluation metrics."""
    # Create directory for comparison results
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    

def plot_portfolio_values(results, save_path=None):
    """Plot portfolio values over time for multiple models."""
    plt.figure(figsize=(12, 6))
    
    for model_name, data in results.items():
        if 'portfolio_values' in data:
            plt.plot(data['portfolio_values'], label=model_name)
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_action_distributions(results, save_path=None):
    """Plot action distributions for multiple models."""
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
    
    if num_models == 1:
        axes = [axes]
    
    for i, (model_name, data) in enumerate(results.items()):
        if 'actions' in data:
            actions = data['actions']
            unique_actions = np.unique(actions)
            action_counts = [np.sum(actions == action) for action in unique_actions]
            
            axes[i].bar(unique_actions, action_counts)
            axes[i].set_title(f'{model_name} Action Distribution')
            axes[i].set_xlabel('Action')
            axes[i].set_ylabel('Count')
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    # Load metrics for all models
    all_metrics = {}
    for model_name in model_names:
        metrics = load_evaluation_metrics(results_dir, model_name)
        if metrics is not None:
            all_metrics[model_name] = metrics
    
    if not all_metrics:
        print("No metrics found for any model.")
        return
    
    # Combine metrics into a single DataFrame
    combined_metrics = pd.DataFrame()
    for model_name, metrics in all_metrics.items():
        metrics['model'] = model_name
        combined_metrics = pd.concat([combined_metrics, metrics], ignore_index=True)
    
    # Save combined metrics
    combined_metrics.to_csv(os.path.join(comparison_dir, 'combined_metrics.csv'), index=False)
    
    # Create comparison visualizations
    create_comparison_plots(combined_metrics, comparison_dir)
    
    # Compare portfolio values over time
    compare_portfolio_values(results_dir, model_names, comparison_dir)
    
    # Compare action distributions
    compare_action_distributions(results_dir, model_names, comparison_dir)
    
    print(f"Comparison results saved to {comparison_dir}")

def create_comparison_plots(combined_metrics, comparison_dir):
    """Create comparison plots for multiple models."""
    # Set style
    sns.set(style="whitegrid")
    
    # Define metrics to compare
    metrics_to_compare = [
        'final_portfolio_value',
        'return_percentage',
        'sharpe_ratio',
        'max_drawdown',
        'cumulative_return',
        'annualized_return',
        'total_reward'
    ]
    
    # Create bar plots for each metric
    for metric in metrics_to_compare:
        if metric in combined_metrics.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model', y=metric, data=combined_metrics)
            plt.title(f'Comparison of {metric.replace("_", " ").title()}')
            plt.xlabel('Model')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'comparison_{metric}.png'))
            plt.close()
    
    # Create a summary table plot
    plt.figure(figsize=(12, 8))
    metrics_subset = combined_metrics[['model'] + [m for m in metrics_to_compare if m in combined_metrics.columns]]
    
    # Normalize metrics for radar chart
    normalized_metrics = metrics_subset.copy()
    for col in metrics_subset.columns:
        if col != 'model' and col in metrics_subset.columns:
            if col == 'max_drawdown':  # Lower is better for max_drawdown
                normalized_metrics[col] = 1 - (metrics_subset[col] - metrics_subset[col].min()) / (metrics_subset[col].max() - metrics_subset[col].min() + 1e-10)
            else:  # Higher is better for other metrics
                normalized_metrics[col] = (metrics_subset[col] - metrics_subset[col].min()) / (metrics_subset[col].max() - metrics_subset[col].min() + 1e-10)
    
    # Create radar chart
    create_radar_chart(normalized_metrics, comparison_dir)

def create_radar_chart(normalized_metrics, comparison_dir):
    """Create a radar chart comparing models across multiple metrics."""
    # Get metrics and models
    metrics = [col for col in normalized_metrics.columns if col != 'model']
    models = normalized_metrics['model'].unique()
    
    # Number of variables
    N = len(metrics)
    
    # Create angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add metrics to the plot
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        model_data = normalized_metrics[normalized_metrics['model'] == model]
        values = model_data[metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison Across Metrics', size=15, y=1.1)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'radar_chart.png'))
    plt.close()

def compare_portfolio_values(results_dir, model_names, comparison_dir):
    """Compare portfolio values over time for multiple models."""
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        history = load_trading_history(results_dir, model_name)
        if history is not None:
            plt.plot(history['time_step'], history['portfolio_value'], label=model_name)
    
    plt.title('Portfolio Value Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'portfolio_value_comparison.png'))
    plt.close()

def compare_action_distributions(results_dir, model_names, comparison_dir):
    """Compare action distributions for multiple models."""
    action_counts = {}
    action_labels = ['Sell', 'Hold', 'Buy']
    
    for model_name in model_names:
        history = load_trading_history(results_dir, model_name)
        if history is not None:
            counts = []
            for label in action_labels:
                counts.append((history['action'] == label).sum())
            action_counts[model_name] = counts
    
    if not action_counts:
        return
    
    # Create DataFrame for plotting
    action_df = pd.DataFrame(action_counts).T
    action_df.columns = action_labels
    
    # Calculate percentages
    action_percentages = action_df.div(action_df.sum(axis=1), axis=0) * 100
    
    # Plot raw counts
    plt.figure(figsize=(12, 8))
    action_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Action Distribution Comparison (Counts)')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.legend(title='Action')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'action_distribution_counts.png'))
    plt.close()
    
    # Plot percentages
    plt.figure(figsize=(12, 8))
    action_percentages.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Action Distribution Comparison (Percentages)')
    plt.xlabel('Model')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Action')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'action_distribution_percentages.png'))
    plt.close()

def compare_reward_networks(results_dir, model_names, comparison_dir):
    """Compare reward networks for multiple models."""
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        history = load_trading_history(results_dir, model_name)
        if history is not None and 'reward' in history.columns:
            plt.plot(history['time_step'], history['reward'], label=model_name)
    
    plt.title('Reward Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'reward_comparison.png'))
    plt.close()

def calculate_comparative_statistics(results_dir, model_names, baseline_model=None):
    """Calculate comparative statistics between models."""
    # Create directory for comparison results
    comparison_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Load metrics for all models
    all_metrics = {}
    for model_name in model_names:
        metrics = load_evaluation_metrics(results_dir, model_name)
        if metrics is not None:
            all_metrics[model_name] = metrics
    
    if not all_metrics:
        print("No metrics found for any model.")
        return
    
    # Calculate comparative statistics
    comparative_stats = pd.DataFrame(index=model_names)
    
    # If baseline model is provided, calculate relative performance
    if baseline_model and baseline_model in all_metrics:
        baseline_metrics = all_metrics[baseline_model]
        
        for model_name, metrics in all_metrics.items():
            if model_name != baseline_model:
                # Calculate relative performance
                comparative_stats.loc[model_name, 'portfolio_value_vs_baseline'] = \
                    (metrics['final_portfolio_value'].values[0] / baseline_metrics['final_portfolio_value'].values[0] - 1) * 100
                comparative_stats.loc[model_name, 'sharpe_ratio_vs_baseline'] = \
                    metrics['sharpe_ratio'].values[0] / baseline_metrics['sharpe_ratio'].values[0]
                comparative_stats.loc[model_name, 'max_drawdown_improvement'] = \
                    baseline_metrics['max_drawdown'].values[0] - metrics['max_drawdown'].values[0]
    
    # Save comparative statistics
    comparative_stats.to_csv(os.path.join(comparison_dir, 'comparative_statistics.csv'))
    
    return comparative_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple trading models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing results')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model names to compare')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline model for relative comparison')
    
    args = parser.parse_args()
    
    # Compare models
    compare_models(args.results_dir, args.models)
    
    # Calculate comparative statistics
    if args.baseline:
        calculate_comparative_statistics(args.results_dir, args.models, args.baseline)