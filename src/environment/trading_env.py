import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class IncrementalStats:
    """
    Helper class for computing incremental statistics efficiently.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # sum of squared differences
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, value):
        """Update statistics with a new value."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def get_variance(self):
        """Get population variance."""
        if self.n < 2:
            return 0.0
        return self.m2 / self.n

    def get_std(self):
        """Get standard deviation."""
        return np.sqrt(self.get_variance())

    def get_sharpe(self, risk_free_rate=0.0, periods_per_year=252):
        """Get annualized Sharpe ratio."""
        if self.get_std() == 0:
            return 0.0
        return (self.mean - risk_free_rate) * np.sqrt(periods_per_year) / self.get_std()


class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    Implements the OpenAI Gym/Gymnasium interface for RL environments.
    """

    def __init__(self, data, window_size, initial_capital=500000, transaction_cost=0.003, data_frequency=None, expert_selection=None):
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
        self.expert_selection = expert_selection
        self.data_frequency = data_frequency or self._infer_frequency()

        # Action space: 0 = Hold, 1 = Buy (open/close long), 2 = Sell (close long / open short depending on config)
        self.action_space = spaces.Discrete(3)

        # Observation space: window_size x (features + 1 position flag)
        self.feature_dim = data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.feature_dim + 1),
            dtype=np.float32
        )

        # Initialize stateful tracking
        self.cached_sharpe = None
        self.last_returns_len = 0
        self.return_stats = IncrementalStats()
        self.portfolio_peak = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.stats_cache = {}
        self.cache_dirty = True

        # Initialize runtime state
        self.reset()

    def _infer_frequency(self):
        """Infer approximate bar frequency (minutes) from index if possible."""
        if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 10:
            diffs = self.data.index.to_series().diff().dropna().dt.total_seconds()
            median_sec = diffs.median() if not diffs.empty else 60.0
            if median_sec <= 61:
                return '1min'
            elif median_sec <= 305:
                return '5min'
            elif median_sec <= 905:
                return '15min'
            elif median_sec <= 3600 + 30:
                return '1h'
            elif median_sec <= 24 * 3600:
                return '1d'
        return '1d'

    def _annualization_factor(self):
        freq = self.data_frequency
        if freq == '1min':
            return 365 * 24 * 60
        if freq == '5min':
            return 365 * 24 * 12
        if freq == '15min':
            return 365 * 24 * 4
        if freq == '1h':
            return 365 * 24
        if freq == '1d':
            return 252
        return 252

    def reset(self, seed=None, options=None, **kwargs):
        """
        Reset the environment to the initial state.
        Returns (observation, info) to follow Gymnasium API.
        """
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.shares_held = 0
        self.current_position = 0  # -1 short, 0 flat, 1 long
        self.returns = []
        self.portfolio_values = [self.initial_capital]
        self.done = False

        self.return_stats = IncrementalStats()
        self.portfolio_peak = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        self.cached_sharpe = None
        self.last_returns_len = 0
        self.stats_cache = {}
        self.cache_dirty = True

        # Recompute observation space in case data columns changed
        try:
            self.feature_dim = self.data.shape[1]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, self.feature_dim + 1),
                dtype=np.float32
            )
        except Exception:
            pass

        return self._get_observation(), {}

    def _get_observation(self):
        """Return window of raw features augmented with position info."""
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step].values
        position_info = np.full((self.window_size, 1), self.current_position)
        observation = np.hstack((window_data, position_info))
        return observation

    def _calculate_reward(self, action):
        idx = self.current_step
        # Prefer precomputed expert labels if present
        if self.expert_selection == 'best':
            col_best_k = f'expert_best_{self.window_size}'
            col_best_plain = 'expert_best'
            best_val = None
            try:
                if col_best_k in self.data.columns:
                    best_val = float(self.data.iloc[idx][col_best_k])
                elif col_best_plain in self.data.columns:
                    best_val = float(self.data.iloc[idx][col_best_plain])
            except Exception:
                best_val = None

            if best_val is not None:
                current_price = self.data.iloc[self.current_step]['Close']
                safe_price = max(current_price, 1e-8)
                portfolio_value = self.capital + self.shares_held * safe_price
                prev_portfolio_value = self.portfolio_values[-1]
                if prev_portfolio_value != 0:
                    portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                    self.return_stats.update(portfolio_change)
                    self.returns.append(portfolio_change)
                self.portfolio_values.append(portfolio_value)

                reward_dict = {'best': best_val}
                expert_reward = best_val
                self.cache_dirty = True
                return expert_reward, reward_dict

        if f'expert_Min-Max' in self.data.columns:
            try:
                min_max_reward = float(self.data.iloc[idx][f'expert_Min-Max'])
                return_reward = float(self.data.iloc[idx][f'expert_Return']) if f'expert_Return' in self.data.columns else 0.0
                sharpe_reward = float(self.data.iloc[idx][f'expert_Sharpe']) if f'expert_Sharpe' in self.data.columns else 0.0
            except Exception:
                min_max_reward = 0.0
                return_reward = 0.0
                sharpe_reward = 0.0

            current_price = self.data.iloc[self.current_step]['Close']
            safe_price = max(current_price, 1e-8)
            portfolio_value = self.capital + self.shares_held * safe_price
            prev_portfolio_value = self.portfolio_values[-1]
            if prev_portfolio_value != 0:
                portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                self.return_stats.update(portfolio_change)
                self.returns.append(portfolio_change)
            self.portfolio_values.append(portfolio_value)

            reward_dict = {
                'Min-Max': min_max_reward,
                'Return': return_reward,
                'Sharpe': sharpe_reward
            }
            expert_reward = min_max_reward
            self.cache_dirty = True

            return expert_reward, reward_dict

        # Fallback calculation
        current_price = self.data.iloc[self.current_step]['Close']
        prev_price = self.data.iloc[self.current_step - 1]['Close']
        if prev_price == 0:
            price_change = 0.0
        else:
            price_change = (current_price - prev_price) / prev_price

        safe_price = max(current_price, 1e-8)
        portfolio_value = self.capital + self.shares_held * safe_price
        prev_portfolio_value = self.portfolio_values[-1]
        self.portfolio_values.append(portfolio_value)

        if prev_portfolio_value != 0:
            portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            daily_return = portfolio_change
            self.return_stats.update(daily_return)
            self.returns.append(daily_return)
            if portfolio_value > self.portfolio_peak:
                self.portfolio_peak = portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            portfolio_change = 0.0
            daily_return = 0.0

        self.cache_dirty = True

        min_max_reward = np.tanh(portfolio_change * 2)
        return_reward = portfolio_change
        if self.return_stats.n > 1:
            periods_per_year = self._annualization_factor()
            std = self.return_stats.get_std()
            sharpe_reward = 0.0 if std == 0 else (self.return_stats.mean) * np.sqrt(periods_per_year) / std
        else:
            sharpe_reward = 0.0

        expert_reward = min_max_reward
        reward_dict = {
            'Min-Max': min_max_reward,
            'Return': return_reward,
            'Sharpe': sharpe_reward
        }

        return expert_reward, reward_dict

    def step(self, action):
        """Take a step in the environment returning (obs, reward, terminated, truncated, info)."""
        if self.current_step >= len(self.data):
            self.done = True
            terminated = True
            truncated = False
            obs = self._get_observation()
            info = {'portfolio_value': self.capital, 'capital': self.capital, 'shares_held': self.shares_held, 'current_price': None, 'reward_dict': {}}
            return obs, 0.0, terminated, truncated, info

        current_price = self.data.iloc[self.current_step]['Close']

        # Action semantics: 1 = buy/open long if flat, 2 = sell/close long
        if action == 1:
            if self.current_position == 0:
                safe_price = max(current_price, 1e-8)
                max_shares = int(self.capital // (safe_price * (1 + self.transaction_cost)))
                if max_shares > 0:
                    self.shares_held = max_shares
                    cost = self.shares_held * safe_price * (1 + self.transaction_cost)
                    self.capital = max(0.0, self.capital - cost)
                    self.current_position = 1
        elif action == 2:
            if self.current_position == 1:
                safe_price = max(current_price, 1e-8)
                proceeds = self.shares_held * safe_price * (1 - self.transaction_cost)
                self.capital = max(0.0, self.capital + proceeds)
                self.shares_held = 0
                self.current_position = 0

        expert_reward, reward_dict = self._calculate_reward(action)

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        terminated = bool(self.done)
        truncated = False

        observation = self._get_observation()
        safe_price = max(current_price, 1e-8)
        portfolio_value = self.capital + self.shares_held * safe_price
        if np.isnan(portfolio_value) or np.isinf(portfolio_value):
            portfolio_value = self.initial_capital

        info = {
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'reward_dict': reward_dict
        }

        return observation, expert_reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]['Close']
            portfolio_value = self.capital + self.shares_held * current_price
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Capital: {self.capital:.2f}")
            print(f"Portfolio value: {portfolio_value:.2f}")
            pos_str = 'Long' if self.current_position == 1 else ('Short' if self.current_position == -1 else 'None')
            print(f"Position: {pos_str}")
            print("-" * 50)

    def get_portfolio_stats(self):
        if not self.cache_dirty and self.stats_cache:
            return self.stats_cache

        if len(self.portfolio_values) < 2:
            return {
                'CR': 0.0,
                'AR': 0.0,
                'SR': 0.0,
                'MDD': 0.0
            }

        portfolio_values = np.array(self.portfolio_values)
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        num_days = len(portfolio_values) - 1
        if num_days > 0:
            annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1
        else:
            annualized_return = 0.0

        if self.return_stats.n > 1:
            periods_per_year = self._annualization_factor()
            std = self.return_stats.get_std()
            sharpe_ratio = 0.0 if std == 0 else (self.return_stats.mean) * np.sqrt(periods_per_year) / std
        else:
            sharpe_ratio = 0.0
        max_drawdown = self.max_drawdown

        stats = {
            'CR': cumulative_return,
            'AR': annualized_return,
            'SR': sharpe_ratio,
            'MDD': max_drawdown
        }

        self.stats_cache = stats
        self.cache_dirty = False
        return stats

    # --- Backwards-compatibility wrappers ---
    def reset_legacy(self):
        """Backward-compatible reset returning only the observation (legacy API).

        New code should call `reset()` which returns `(obs, info)`. This helper
        is provided for older call-sites that expect a single return value.
        """
        obs, _ = self.reset()
        return obs

    def step_legacy(self, action):
        """Backward-compatible step returning (obs, reward, done, info).

        New code should call `step()` which returns `(obs, reward, terminated, truncated, info)`.
        This helper merges `terminated` and `truncated` into a single `done` boolean
        to match older Gym APIs.
        """
        obs, reward, terminated, truncated, info = self.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info
    
    def _calculate_reward(self, action):
        """
        Calculate the reward for the current action with optimized caching.
        This is the expert-defined reward function that will be used
        alongside the self-rewarding network.
        
        Args:
            action (int): Action taken (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            tuple: (expert_reward, reward_dict)
        """
        # Use precomputed expert labels if available to avoid per-step lookahead
        idx = self.current_step
        # If configured to prefer the best expert metric, try to use precomputed best_{k}
        if self.expert_selection == 'best':
            col_best_k = f'expert_best_{self.window_size}'
            col_best_plain = 'expert_best'
            best_val = None
            try:
                if col_best_k in self.data.columns:
                    best_val = float(self.data.iloc[idx][col_best_k])
                elif col_best_plain in self.data.columns:
                    best_val = float(self.data.iloc[idx][col_best_plain])
            except Exception:
                best_val = None

            if best_val is not None:
                # Update portfolio values with simple estimate (keep existing mechanism)
                current_price = self.data.iloc[self.current_step]['Close']
                safe_price = max(current_price, 1e-8)
                portfolio_value = self.capital + self.shares_held * safe_price
                prev_portfolio_value = self.portfolio_values[-1]
                if prev_portfolio_value != 0:
                    portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                    self.return_stats.update(portfolio_change)
                    self.returns.append(portfolio_change)
                self.portfolio_values.append(portfolio_value)

                reward_dict = {'best': best_val}
                expert_reward = best_val
                self.cache_dirty = True
                return expert_reward, reward_dict

        # Default path: use precomputed Min-Max / Return / Sharpe if present
        if f'expert_Min-Max' in self.data.columns:
            try:
                min_max_reward = float(self.data.iloc[idx][f'expert_Min-Max'])
                return_reward = float(self.data.iloc[idx][f'expert_Return']) if f'expert_Return' in self.data.columns else 0.0
                sharpe_reward = float(self.data.iloc[idx][f'expert_Sharpe']) if f'expert_Sharpe' in self.data.columns else 0.0
            except Exception:
                min_max_reward = 0.0
                return_reward = 0.0
                sharpe_reward = 0.0

            # Update portfolio values with simple estimate (keep existing mechanism)
            current_price = self.data.iloc[self.current_step]['Close']
            safe_price = max(current_price, 1e-8)
            portfolio_value = self.capital + self.shares_held * safe_price
            prev_portfolio_value = self.portfolio_values[-1]
            if prev_portfolio_value != 0:
                portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                self.return_stats.update(portfolio_change)
                self.returns.append(portfolio_change)
            self.portfolio_values.append(portfolio_value)

            reward_dict = {
                'Min-Max': min_max_reward,
                'Return': return_reward,
                'Sharpe': sharpe_reward
            }
            expert_reward = min_max_reward
            # Mark cache as dirty (stats changed)
            self.cache_dirty = True

            return expert_reward, reward_dict

        # Fallback to original online computation if precomputed labels are not present
        # Get current price and previous price
        current_price = self.data.iloc[self.current_step]['Close']
        prev_price = self.data.iloc[self.current_step - 1]['Close']

        # Calculate price change percentage
        if prev_price == 0:
            price_change = 0.0
        else:
            price_change = (current_price - prev_price) / prev_price
        # Calculate portfolio value using current capital and held shares
        safe_price = max(current_price, 1e-8)
        portfolio_value = self.capital + self.shares_held * safe_price
        prev_portfolio_value = self.portfolio_values[-1]

        # Update portfolio values list
        self.portfolio_values.append(portfolio_value)

        # Calculate portfolio change and update incremental stats
        if prev_portfolio_value != 0:
            portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            daily_return = portfolio_change
            
            # Update incremental statistics
            self.return_stats.update(daily_return)
            self.returns.append(daily_return)
            
            # Update drawdown calculation
            if portfolio_value > self.portfolio_peak:
                self.portfolio_peak = portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            portfolio_change = 0.0
            daily_return = 0.0

        # Mark cache as dirty when stats change
        self.cache_dirty = True

        # Calculate different reward components

        # 1. Min-Max Reward: Scaled between -1 and 1 based on portfolio change
        # Fix: portfolio_change is already a percentage, so we use tanh for smooth scaling
        # This provides better gradient for RL while maintaining bounded rewards
        min_max_reward = np.tanh(portfolio_change * 2)  # Scale factor of 2 provides good sensitivity

        # 2. Return Reward: Direct portfolio return
        return_reward = portfolio_change

        # 3. Sharpe Ratio Reward: Use incremental calculation
        if self.return_stats.n > 1:
            periods_per_year = self._annualization_factor()
            # Recompute Sharpe with dynamic periods
            std = self.return_stats.get_std()
            sharpe_reward = 0.0 if std == 0 else (self.return_stats.mean) * np.sqrt(periods_per_year) / std
        else:
            sharpe_reward = 0.0

        # The expert reward will be the Min-Max reward by default
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
            tuple: (observation, reward, terminated, truncated, info)
        """
        # If current_step is out of range, mark done and return terminal observation
        if self.current_step >= len(self.data):
            self.done = True
            terminated = True
            truncated = False
            obs = self._get_observation()
            info = {'portfolio_value': self.capital, 'capital': self.capital, 'shares_held': self.shares_held, 'current_price': None, 'reward_dict': {}}
            return obs, 0.0, terminated, truncated, info

        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute the action
        # Action semantics updated to support short positions
        #  - action 1 (Buy): open long if flat, close short if currently short, otherwise ignore
        #  - action 2 (Sell): open short if flat, close long if currently long, otherwise ignore
        if action == 1:  # Buy / open long (no short support)
            if self.current_position == 0:
                # Open long position
                safe_price = max(current_price, 1e-8)
                max_shares = int(self.capital // (safe_price * (1 + self.transaction_cost)))
                if max_shares > 0:
                    self.shares_held = max_shares
                    cost = self.shares_held * safe_price * (1 + self.transaction_cost)
                    self.capital = max(0.0, self.capital - cost)
                    self.current_position = 1

        elif action == 2:  # Sell / close long (do not open short when flat)
            if self.current_position == 1:
                # Close long position
                safe_price = max(current_price, 1e-8)
                proceeds = self.shares_held * safe_price * (1 - self.transaction_cost)
                self.capital = max(0.0, self.capital + proceeds)
                self.shares_held = 0
                self.current_position = 0
        
        # Calculate reward
        expert_reward, reward_dict = self._calculate_reward(action)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if episode is done (terminated)
        if self.current_step >= len(self.data) - 1:
            self.done = True

        terminated = bool(self.done)
        truncated = False
        
        # Get the new observation
        observation = self._get_observation()
        
        # Calculate portfolio value for info
        safe_price = max(current_price, 1e-8)
        portfolio_value = self.capital + self.shares_held * safe_price
        
        # Ensure portfolio value is valid
        if np.isnan(portfolio_value) or np.isinf(portfolio_value):
            portfolio_value = self.initial_capital
        
        # Create info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'reward_dict': reward_dict
        }

        return observation, expert_reward, terminated, truncated, info
    
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
            pos_str = 'Long' if self.current_position == 1 else ('Short' if self.current_position == -1 else 'None')
            print(f"Position: {pos_str}")
            print("-" * 50)
    
    def get_portfolio_stats(self):
        """
        Calculate portfolio statistics with caching for efficiency.
        
        Returns:
            dict: Dictionary with portfolio statistics
        """
        if not self.cache_dirty and self.stats_cache:
            return self.stats_cache
        
        if len(self.portfolio_values) < 2:
            return {
                'CR': 0.0,
                'AR': 0.0,
                'SR': 0.0,
                'MDD': 0.0
            }
        
        # Use incremental statistics for better performance
        portfolio_values = np.array(self.portfolio_values)
        
        # Calculate metrics
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Annualized return
        num_days = len(portfolio_values) - 1
        if num_days > 0:
            annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1
        else:
            annualized_return = 0.0
        
        # Dynamic Sharpe using inferred frequency
        if self.return_stats.n > 1:
            periods_per_year = self._annualization_factor()
            std = self.return_stats.get_std()
            sharpe_ratio = 0.0 if std == 0 else (self.return_stats.mean) * np.sqrt(periods_per_year) / std
        else:
            sharpe_ratio = 0.0
        max_drawdown = self.max_drawdown
        
        stats = {
            'CR': cumulative_return,
            'AR': annualized_return,
            'SR': sharpe_ratio,
            'MDD': max_drawdown
        }
        
        # Cache the results
        self.stats_cache = stats
        self.cache_dirty = False
        
        return stats