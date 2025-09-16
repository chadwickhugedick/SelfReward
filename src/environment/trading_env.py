import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    Implements the OpenAI Gym interface for RL environments.
    """
    
    def __init__(self, data, window_size, initial_capital=500000, transaction_cost=0.003):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV and feature data
            window_size (int): Size of the observation window
            initial_capital (float): Initial capital for trading
            transaction_cost (float): Transaction cost as a percentage
        """
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: window_size * features + position info
        self.feature_dim = data.shape[1]  # Number of features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.feature_dim + 1),  # +1 for position info
            dtype=np.float32
        )

        # Initialize Sharpe ratio cache
        self.cached_sharpe = None
        self.last_returns_len = 0

        # Initialize state
        self.reset()
    
    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.array: Initial observation
        """
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.shares_held = 0
        self.current_position = 0  # 0: no position, 1: long position
        self.returns = []
        self.portfolio_values = [self.initial_capital]
        self.done = False

        # Reset Sharpe cache
        self.cached_sharpe = None
        self.last_returns_len = 0

        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation (state).
        
        Returns:
            np.array: Current observation
        """
        # Get the window of data
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step].values
        
        # Add position information to each time step in the window
        position_info = np.full((self.window_size, 1), self.current_position)
        observation = np.hstack((window_data, position_info))
        
        return observation
    
    def _calculate_reward(self, action):
        """
        Calculate the reward for the current action.
        This is the expert-defined reward function that will be used
        alongside the self-rewarding network.
        
        Args:
            action (int): Action taken (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            float: Reward value
        """
        # Get current price and previous price
        current_price = self.data.iloc[self.current_step]['Close']
        prev_price = self.data.iloc[self.current_step - 1]['Close']

        # Calculate price change percentage
        if prev_price == 0:
            price_change = 0.0
        else:
            price_change = (current_price - prev_price) / prev_price
        
        # Calculate portfolio value
        portfolio_value = self.capital + self.shares_held * current_price
        prev_portfolio_value = self.portfolio_values[-1]
        portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Update portfolio values list
        self.portfolio_values.append(portfolio_value)
        
        # Calculate returns for Sharpe ratio
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns.append(daily_return)

        # Calculate different reward components

        # 1. Min-Max Reward: Scaled between -1 and 1 based on portfolio change
        min_max_reward = np.clip(portfolio_change * 100, -1, 1)

        # 2. Return Reward: Direct portfolio return
        return_reward = portfolio_change

        # 3. Sharpe Ratio Reward: Risk-adjusted return (with caching)
        if len(self.returns) != self.last_returns_len and len(self.returns) > 1:
            self.cached_sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-6) * np.sqrt(252)
            self.last_returns_len = len(self.returns)
        sharpe_reward = self.cached_sharpe if self.cached_sharpe is not None else 0
        
        # The expert reward will be the Min-Max reward by default
        # The self-rewarding network will learn to predict better rewards
        expert_reward = min_max_reward
        
        # Store all reward types for the self-rewarding network to learn from
        reward_dict = {
            'Min-Max': min_max_reward,
            'Return': return_reward,
            'Sharpe': sharpe_reward
        }
        
        return expert_reward, reward_dict
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): Action to take (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute the action
        if action == 1:  # Buy
            if self.current_position == 0:  # Only buy if not already in a position
                # Calculate maximum shares that can be bought
                max_shares = self.capital // (current_price * (1 + self.transaction_cost))
                self.shares_held = max_shares
                self.capital -= self.shares_held * current_price * (1 + self.transaction_cost)
                self.current_position = 1
        
        elif action == 2:  # Sell
            if self.current_position == 1:  # Only sell if in a position
                # Sell all shares
                self.capital += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
                self.current_position = 0
        
        # Calculate reward
        expert_reward, reward_dict = self._calculate_reward(action)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Get the new observation
        observation = self._get_observation()
        
        # Calculate portfolio value for info
        portfolio_value = self.capital + self.shares_held * current_price
        
        # Create info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'reward_dict': reward_dict
        }
        
        return observation, expert_reward, self.done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]['Close']
            portfolio_value = self.capital + self.shares_held * current_price
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Capital: {self.capital:.2f}")
            print(f"Portfolio value: {portfolio_value:.2f}")
            print(f"Position: {'Long' if self.current_position == 1 else 'None'}")
            print("-" * 50)
    
    def get_portfolio_stats(self):
        """
        Calculate portfolio statistics.
        
        Returns:
            dict: Dictionary with portfolio statistics
        """
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        # Calculate maximum drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdowns)
        
        return {
            'CR': cumulative_return,
            'AR': annualized_return,
            'SR': sharpe_ratio,
            'MDD': max_drawdown
        }