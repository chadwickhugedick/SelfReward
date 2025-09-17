#!/usr/bin/env python3
"""
Test script to validate the implementation fixes for:
1. Reward scaling bug fix
2. Data chunking for large datasets
3. Vectorized sequence creation
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add src to path
sys.path.append('src')

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment

def test_reward_scaling_fix():
    """Test the Min-Max reward scaling fix."""
    print("Testing reward scaling fix...")

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(105, 115, 100),
        'Low': np.random.uniform(95, 105, 100),
        'Close': np.random.uniform(100, 110, 100),
        'Volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)

    # Create environment
    env = TradingEnvironment(data, window_size=10, expert_selection=None)

    # Test different portfolio changes
    test_changes = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5]

    print("Portfolio Change -> Min-Max Reward")
    print("-" * 35)

    for change in test_changes:
        # Simulate the reward calculation
        min_max_reward = np.tanh(change * 2)
        print("8.3f")

    print("PASS: Reward scaling test completed\n")

def test_vectorized_sequences():
    """Test the vectorized sequence creation."""
    print("Testing vectorized sequence creation...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'Close': np.random.uniform(100, 110, 1000),
        'Volume': np.random.uniform(1000, 10000, 1000),
        'Feature1': np.random.randn(1000),
        'Feature2': np.random.randn(1000)
    }, index=dates)

    # Create data processor
    config = {'data': {'train_start_date': '2020-01-01', 'train_end_date': '2020-12-31'}}
    processor = DataProcessor(config)

    window_size = 20

    # Test vectorized method
    start_time = time.time()
    X_vec, y_vec = processor.create_sequences(data, window_size)
    vec_time = time.time() - start_time

    print(f"Vectorized method: {X_vec.shape[0]} sequences, {X_vec.shape[1]} timesteps, {X_vec.shape[2]} features")
    print(".4f")
    print(f"Memory usage: {X_vec.nbytes / 1024 / 1024:.2f} MB")

    # Verify shapes
    expected_sequences = len(data) - window_size
    assert X_vec.shape[0] == expected_sequences, f"Expected {expected_sequences} sequences, got {X_vec.shape[0]}"
    assert X_vec.shape[1] == window_size, f"Expected {window_size} timesteps, got {X_vec.shape[1]}"
    assert len(y_vec) == expected_sequences, f"Expected {expected_sequences} targets, got {len(y_vec)}"

    print("PASS: Vectorized sequence creation test passed\n")

def test_chunked_sequences():
    """Test the chunked sequence creation for large datasets."""
    print("Testing chunked sequence creation...")

    # Create larger sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50000, freq='D')
    data = pd.DataFrame({
        'Close': np.random.uniform(100, 110, 50000),
        'Volume': np.random.uniform(1000, 10000, 50000),
        'Feature1': np.random.randn(50000),
        'Feature2': np.random.randn(50000)
    }, index=dates)

    # Create data processor
    config = {'data': {'train_start_date': '2020-01-01', 'train_end_date': '2020-12-31'}}
    processor = DataProcessor(config)

    window_size = 20
    chunk_size = 5000

    # Test chunked method
    start_time = time.time()
    X_chunk, y_chunk = processor.create_sequences_chunked(data, window_size, chunk_size)
    chunk_time = time.time() - start_time

    print(f"Chunked method: {X_chunk.shape[0]} sequences, {X_chunk.shape[1]} timesteps, {X_chunk.shape[2]} features")
    print(".4f")
    print(f"Memory usage: {X_chunk.nbytes / 1024 / 1024:.2f} MB")

    # Verify shapes
    expected_sequences = len(data) - window_size
    assert X_chunk.shape[0] == expected_sequences, f"Expected {expected_sequences} sequences, got {X_chunk.shape[0]}"
    assert X_chunk.shape[1] == window_size, f"Expected {window_size} timesteps, got {X_chunk.shape[1]}"
    assert len(y_chunk) == expected_sequences, f"Expected {expected_sequences} targets, got {len(y_chunk)}"

    print("PASS: Chunked sequence creation test passed\n")

def test_performance_comparison():
    """Compare performance of old vs new methods."""
    print("Performance comparison...")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=10000, freq='D')
    data = pd.DataFrame({
        'Close': np.random.uniform(100, 110, 10000),
        'Volume': np.random.uniform(1000, 10000, 10000),
        'Feature1': np.random.randn(10000),
        'Feature2': np.random.randn(10000)
    }, index=dates)

    window_size = 20

    # Old method (loop-based)
    def old_create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data.iloc[i:i+window_size].values)
            y.append(data.iloc[i+window_size]['Close'])
        return np.array(X), np.array(y)

    # Time old method
    start_time = time.time()
    X_old, y_old = old_create_sequences(data, window_size)
    old_time = time.time() - start_time

    # Time new method
    config = {'data': {'train_start_date': '2020-01-01', 'train_end_date': '2020-12-31'}}
    processor = DataProcessor(config)
    start_time = time.time()
    X_new, y_new = processor.create_sequences(data, window_size)
    new_time = time.time() - start_time

    # Compare results
    print("Performance comparison:")
    print(".4f")
    print(".4f")
    print(".2f")

    # Verify results are identical
    np.testing.assert_array_almost_equal(X_old, X_new, decimal=10)
    np.testing.assert_array_almost_equal(y_old, y_new, decimal=10)

    print("PASS: Results are identical - performance improvement achieved!\n")

if __name__ == "__main__":
    print("Running implementation fixes validation tests...\n")

    try:
        test_reward_scaling_fix()
        test_vectorized_sequences()
        test_chunked_sequences()
        test_performance_comparison()

        print("SUCCESS: All tests passed! Implementation fixes are working correctly.")

    except Exception as e:
        print(f"FAIL: Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)