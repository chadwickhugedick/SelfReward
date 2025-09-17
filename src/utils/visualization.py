import matplotlib.pyplot as plt
from typing import List, Dict


def plot_portfolio_performance(portfolio_values: List[float], benchmark_values: List[float], save_path: str):
	"""Plot portfolio vs benchmark and save the figure to `save_path`."""
	fig, ax = plt.subplots(figsize=(6, 3))
	ax.plot(portfolio_values, label='Portfolio')
	if benchmark_values is not None:
		ax.plot(benchmark_values, label='Benchmark')
	ax.legend()
	ax.set_title('Portfolio Performance')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)


def plot_trading_actions(actions_history: List[int], prices: List[float], save_path: str):
	"""Plot trading actions over price series and save the figure."""
	fig, ax = plt.subplots(figsize=(6, 3))
	ax.plot(prices, label='Price')
	# Mark buy (1) and sell (2) actions
	buys = [i for i, a in enumerate(actions_history) if a == 1]
	sells = [i for i, a in enumerate(actions_history) if a == 2]
	if buys:
		ax.scatter(buys, [prices[i] for i in buys], marker='^', color='g', label='Buy')
	if sells:
		ax.scatter(sells, [prices[i] for i in sells], marker='v', color='r', label='Sell')
	ax.legend()
	ax.set_title('Trading Actions')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)


def plot_reward_distribution(rewards_history: List[float], save_path: str):
	"""Plot a histogram of rewards and save the figure."""
	fig, ax = plt.subplots(figsize=(4, 3))
	ax.hist(rewards_history, bins=20)
	ax.set_title('Reward Distribution')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)


def plot_training_metrics(training_metrics: Dict[str, List[float]], save_path: str):
	"""Plot training metrics dictionary where values are lists and save the figure."""
	# Simple multi-line plot: plot each metric on its own subplot when possible
	keys = list(training_metrics.keys())
	n = len(keys)
	fig, axes = plt.subplots(nrows=max(1, n), ncols=1, figsize=(6, 2 * n))
	if n == 1:
		axes = [axes]
	for ax, k in zip(axes, keys):
		try:
			ax.plot(training_metrics[k])
			ax.set_title(k)
		except Exception:
			ax.text(0.5, 0.5, f'No data for {k}', ha='center')
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close(fig)


__all__ = [
	'plot_portfolio_performance',
	'plot_trading_actions',
	'plot_reward_distribution',
	'plot_training_metrics'
]
