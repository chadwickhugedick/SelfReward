import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.trading_env import TradingEnvironment

class TestTradingEnvironmentExtended(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing with predictable price patterns
        dates = pd.date_range('2022-01-01', periods=100)
        
        # Create price data with a clear trend for predictable rewards
        close_prices = np.linspace(100, 200, 100)  # Linear increase from 100 to 200
        
        self.data = pd.DataFrame({
            'Open': close_prices - 1,
            'High': close_prices + 2,
            'Low': close_prices - 2,
            'Close': close_prices,
            'Volume': np.random.rand(100) * 1000000,
            'SMA_10': np.random.rand(100) * 10 + 100,
            'RSI': np.random.rand(100) * 100
        }, index=dates)
        
        # Initialize the environment
        self.window_size = 20
        self.env = TradingEnvironment(
            data=self.data,
            window_size=self.window_size,
            initial_capital=10000,
            transaction_cost=0.001
        )
        # Keep expert selection unset in unit tests
        self.env.expert_selection = None
    
    def test_observation_structure(self):
        # Test that the observation has the correct structure
        observation, info = self.env.reset()
        
        # Check observation shape and type
        self.assertEqual(observation.shape, (self.window_size, self.data.shape[1] + 1))
        # Note: The actual dtype might be float64 instead of float32 depending on implementation
        self.assertIn(observation.dtype, [np.float32, np.float64])
        
        # Check that position info is correctly added
        self.assertEqual(observation[0, -1], 0)  # Initial position is 0
    
    def test_reward_calculation(self):
        # Test the reward calculation
        observation, info = self.env.reset()

        # Take a buy action when prices are rising
        observation, reward, terminated, truncated, info = self.env.step(1)  # Buy
        done = bool(terminated or truncated)
        
        # Check that reward is calculated and included in info
        self.assertIsInstance(reward, float)
        self.assertIn('reward_dict', info)
        self.assertIn('Min-Max', info['reward_dict'])
        self.assertIn('Return', info['reward_dict'])
        self.assertIn('Sharpe', info['reward_dict'])
    
    def test_portfolio_value_tracking(self):
        # Test that portfolio values are tracked correctly
        observation, info = self.env.reset()
        initial_value = self.env.portfolio_values[0]
        
        # Take a buy action
        self.env.step(1)  # Buy
        
        # Check that portfolio values list is updated
        self.assertEqual(len(self.env.portfolio_values), 2)
        
        # Take a sell action
        self.env.step(2)  # Sell
        
        # Check that portfolio values list is updated again
        self.assertEqual(len(self.env.portfolio_values), 3)
    
    def test_episode_completion(self):
        # Test that episode completes when reaching the end of data
        observation, info = self.env.reset()
        
        # Set current step to near the end
        self.env.current_step = len(self.data) - 2
        
        # Take one more step
        observation, reward, terminated, truncated, info = self.env.step(0)  # Hold
        done = bool(terminated or truncated)
        
        # Check that episode is done
        self.assertTrue(done)
    
    def test_transaction_costs(self):
        # Test that transaction costs are applied correctly
        observation, info = self.env.reset()
        initial_capital = self.env.capital
        
        # Get current price
        current_price = self.data.iloc[self.env.current_step]['Close']
        
        # Calculate expected capital after buying with transaction costs
        max_shares = initial_capital // (current_price * (1 + self.env.transaction_cost))
        expected_capital = initial_capital - max_shares * current_price * (1 + self.env.transaction_cost)
        
        # Take buy action
        self.env.step(1)  # Buy
        
        # Check that capital is reduced by the correct amount
        self.assertAlmostEqual(self.env.capital, expected_capital, delta=0.01)
        
        # Store current capital and shares before selling
        capital_before_sell = self.env.capital
        shares_before_sell = self.env.shares_held
        
        # Take sell action
        self.env.step(2)  # Sell
        
        # Check that capital increased and shares are zero
        self.assertTrue(self.env.capital > capital_before_sell)
        self.assertEqual(self.env.shares_held, 0)
    
    def test_portfolio_stats_calculation(self):
        # Test portfolio statistics calculation
        observation, info = self.env.reset()
        
        # Simulate a trading strategy: buy and hold
        self.env.step(1)  # Buy
        
        # Move forward several steps
        for _ in range(20):
            self.env.step(0)  # Hold
            if self.env.done:
                break
        
        # Get portfolio stats
        stats = self.env.get_portfolio_stats()
        
        # Check that all required stats are present
        self.assertIn('CR', stats)  # Cumulative return
        self.assertIn('AR', stats)  # Annualized return
        self.assertIn('SR', stats)  # Sharpe ratio
        self.assertIn('MDD', stats)  # Maximum drawdown
        
        # Check that stats are reasonable
        self.assertGreater(stats['CR'], 0)  # Positive return due to rising prices
        self.assertGreaterEqual(stats['MDD'], 0)  # Non-negative drawdown
    
    def test_render_function(self):
        # Test that render function works without errors
        observation, info = self.env.reset()
        try:
            self.env.render()
            render_success = True
        except Exception as e:
            render_success = False
        
        self.assertTrue(render_success)
    
    def test_invalid_actions(self):
        # Test handling of invalid actions
        self.env.reset()
        
        # Buy twice in a row (second buy should have no effect)
        self.env.step(1)  # Buy
        initial_shares = self.env.shares_held
        initial_capital = self.env.capital
        
        self.env.step(1)  # Buy again
        
        # Check that shares and capital remain unchanged
        self.assertEqual(self.env.shares_held, initial_shares)
        self.assertEqual(self.env.capital, initial_capital)
        
        # Sell when no position (should have no effect)
        self.env.reset()
        initial_capital = self.env.capital
        
        self.env.step(2)  # Sell
        
        # Check that capital remains unchanged
        self.assertEqual(self.env.capital, initial_capital)

if __name__ == '__main__':
    unittest.main()