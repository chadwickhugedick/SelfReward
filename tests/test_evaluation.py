import unittest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the visualization module to test
from src.utils.visualization import (
    plot_portfolio_performance, plot_trading_actions, plot_reward_distribution,
    plot_training_metrics
)



class TestVisualization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data for testing
        self.portfolio_values = [10000, 10200, 10500, 10300, 10800, 11200, 12000]
        self.benchmark_values = [10000, 10100, 10300, 10200, 10400, 10600, 10800]
        self.actions_history = [1, 2, 1, 0, 2, 1]  # Hold, Buy, Hold, Sell, Buy, Hold
        self.prices = [100, 102, 105, 103, 108, 112, 120]
        self.rewards_history = [0.01, 0.02, -0.01, 0.03, 0.02, 0.04]
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [10, 15, 20, 25, 30],
            'portfolio_values': [10000, 10500, 11000, 11500, 12000],
            'dqn_losses': [0.5, 0.4, 0.3, 0.2, 0.1],
            'reward_net_losses': [0.3, 0.25, 0.2, 0.15, 0.1]
        }
        
    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_portfolio_performance(self, mock_savefig):
        # Test plotting portfolio performance
        save_path = os.path.join(self.test_dir, 'portfolio_performance.png')
        plot_portfolio_performance(self.portfolio_values, self.benchmark_values, save_path)
        
        # Check if savefig was called with the correct path
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        self.assertEqual(args[0], save_path)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_trading_actions(self, mock_savefig):
        # Test plotting trading actions
        save_path = os.path.join(self.test_dir, 'trading_actions.png')
        plot_trading_actions(self.actions_history, self.prices, save_path)
        
        # Check if savefig was called with the correct path
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        self.assertEqual(args[0], save_path)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_reward_distribution(self, mock_savefig):
        # Test plotting reward distribution
        save_path = os.path.join(self.test_dir, 'reward_distribution.png')
        plot_reward_distribution(self.rewards_history, save_path)
        
        # Check if savefig was called with the correct path
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        self.assertEqual(args[0], save_path)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_metrics(self, mock_savefig):
        # Test plotting training metrics
        save_path = os.path.join(self.test_dir, 'training_metrics.png')
        plot_training_metrics(self.training_metrics, save_path)
        
        # Check if savefig was called with the correct path
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        self.assertEqual(args[0], save_path)
    


if __name__ == '__main__':
    unittest.main()