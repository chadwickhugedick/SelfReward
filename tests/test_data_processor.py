import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        # Create a simple config for testing
        self.config = {
            'data': {
                'train_start_date': '2020-01-01',
                'train_end_date': '2021-01-01',
                'test_start_date': '2021-01-02',
                'test_end_date': '2021-06-01'
            },
            'environment': {
                'window_size': 20
            }
        }
        self.data_processor = DataProcessor(self.config)
    
    def test_download_data(self):
        # Test downloading data for a known ticker
        data = self.data_processor.download_data('AAPL', '2022-01-01', '2022-01-31')
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data) > 0)
        self.assertTrue('Close' in data.columns)
    
    def test_add_technical_indicators(self):
        # Create a sample dataframe
        dates = pd.date_range('2022-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.rand(30) * 100 + 100,
            'High': np.random.rand(30) * 100 + 110,
            'Low': np.random.rand(30) * 100 + 90,
            'Close': np.random.rand(30) * 100 + 100,
            'Volume': np.random.rand(30) * 1000000
        }, index=dates)
        
        # Add technical indicators
        processed_data = self.data_processor.add_technical_indicators(data)
        
        # Check if indicators were added
        self.assertTrue('SMA_10' in processed_data.columns)
        self.assertTrue('RSI' in processed_data.columns)
        self.assertTrue('MACD' in processed_data.columns)
        self.assertTrue('ATR' in processed_data.columns)
        self.assertTrue('OBV' in processed_data.columns)
    
    def test_normalize_data(self):
        # Create a sample dataframe with fixed values
        dates = pd.date_range('2022-01-01', periods=5)
        data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)

        # Normalize the data
        normalized_data = self.data_processor.normalize_data(data)

        # Check if data is normalized (values between 0 and 1)
        self.assertTrue(normalized_data.min().min() >= 0)
        self.assertTrue(normalized_data.max().max() <= 1 + 1e-10)  # Allow for floating point precision
    
    def test_create_sequences(self):
        # Create a sample dataframe
        dates = pd.date_range('2022-01-01', periods=30)
        data = pd.DataFrame({
            'Open': np.random.rand(30) * 100 + 100,
            'High': np.random.rand(30) * 100 + 110,
            'Low': np.random.rand(30) * 100 + 90,
            'Close': np.random.rand(30) * 100 + 100,
            'Volume': np.random.rand(30) * 1000000
        }, index=dates)
        
        # Create sequences
        X, y = self.data_processor.create_sequences(data, window_size=5)
        
        # Check shapes
        self.assertEqual(X.shape[0], 25)  # 30 - 5 = 25 sequences
        self.assertEqual(X.shape[1], 5)   # window_size = 5
        self.assertEqual(X.shape[2], 5)   # 5 features
        self.assertEqual(y.shape[0], 25)  # 25 targets
    
    def test_split_data(self):
        # Create a sample dataframe
        dates = pd.date_range('2022-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.rand(100) * 100 + 100
        }, index=dates)

        # Split the data
        train_data, val_data, test_data = self.data_processor.split_data(data, train_ratio=0.7, val_ratio=0.15)

        # Check sizes
        self.assertEqual(len(train_data), 70)
        self.assertEqual(len(val_data), 15)
        self.assertEqual(len(test_data), 15)

if __name__ == '__main__':
    unittest.main()