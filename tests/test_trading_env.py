import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.trading_env import TradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe for testing
        dates = pd.date_range('2022-01-01', periods=100)
        self.data = pd.DataFrame({
            'Open': np.random.rand(100) * 10 + 100,
            'High': np.random.rand(100) * 10 + 105,
            'Low': np.random.rand(100) * 10 + 95,
            'Close': np.random.rand(100) * 10 + 100,
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
    
    def test_initialization(self):
        # Test environment initialization
        self.assertEqual(self.env.initial_capital, 10000)
        self.assertEqual(self.env.transaction_cost, 0.001)
        self.assertEqual(self.env.window_size, 20)
        self.assertEqual(self.env.action_space.n, 3)  # Hold, Buy, Sell
        
        # Check observation space
        expected_shape = (self.window_size, self.data.shape[1] + 1)  # +1 for position info
        self.assertEqual(self.env.observation_space.shape, expected_shape)
    
    def test_reset(self):
        # Test environment reset
        observation = self.env.reset()
        
        # Check observation shape
        expected_shape = (self.window_size, self.data.shape[1] + 1)  # +1 for position info
        self.assertEqual(observation.shape, expected_shape)
        
        # Check initial state
        self.assertEqual(self.env.current_step, self.window_size)
        self.assertEqual(self.env.capital, 10000)
        self.assertEqual(self.env.shares_held, 0)
        self.assertEqual(self.env.current_position, 0)
        self.assertEqual(len(self.env.portfolio_values), 1)
        self.assertEqual(self.env.portfolio_values[0], 10000)
    
    def test_step_hold(self):
        # Test hold action
        self.env.reset()
        initial_capital = self.env.capital
        initial_shares = self.env.shares_held
        
        # Take hold action
        observation, reward, done, info = self.env.step(0)
        
        # Check that capital and shares remain unchanged
        self.assertEqual(self.env.capital, initial_capital)
        self.assertEqual(self.env.shares_held, initial_shares)
        self.assertEqual(self.env.current_step, self.window_size + 1)
    
    def test_step_buy(self):
        # Test buy action
        self.env.reset()
        initial_capital = self.env.capital
        
        # Take buy action
        observation, reward, done, info = self.env.step(1)
        
        # Check that shares were bought and capital decreased
        self.assertTrue(self.env.shares_held > 0)
        self.assertTrue(self.env.capital < initial_capital)
        self.assertEqual(self.env.current_position, 1)
    
    def test_step_sell_after_buy(self):
        # Test sell action after buying
        self.env.reset()
        
        # First buy
        self.env.step(1)
        shares_after_buy = self.env.shares_held
        capital_after_buy = self.env.capital
        
        # Then sell
        observation, reward, done, info = self.env.step(2)
        
        # Check that shares were sold and capital increased
        self.assertEqual(self.env.shares_held, 0)
        self.assertTrue(self.env.capital > capital_after_buy)
        self.assertEqual(self.env.current_position, 0)
    
    def test_portfolio_stats(self):
        # Test portfolio statistics calculation
        self.env.reset()
        
        # Simulate a few steps
        for _ in range(20):
            action = np.random.randint(0, 3)  # Random action
            self.env.step(action)
            if self.env.done:
                break
        
        # Get portfolio stats
        stats = self.env.get_portfolio_stats()
        
        # Check that stats are calculated
        self.assertIn('CR', stats)  # Cumulative return
        self.assertIn('AR', stats)  # Annualized return
        self.assertIn('SR', stats)  # Sharpe ratio
        self.assertIn('MDD', stats)  # Maximum drawdown

if __name__ == '__main__':
    unittest.main()