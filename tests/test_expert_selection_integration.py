import os
import sys
from pathlib import Path

# Ensure workspace root is on sys.path so `src` is importable during tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment


def make_synthetic_ohlcv(n=500):
    rng = pd.date_range('2020-01-01', periods=n, freq='min')
    price = np.cumsum(np.random.randn(n)) + 100.0
    df = pd.DataFrame({
        'Open': price + np.random.randn(n) * 0.01,
        'High': price + np.abs(np.random.randn(n) * 0.05),
        'Low': price - np.abs(np.random.randn(n) * 0.05),
        'Close': price + np.random.randn(n) * 0.01,
        'Volume': np.abs(np.random.randn(n) * 10 + 100)
    }, index=rng)
    return df


def test_env_uses_expert_best_for_reward(tmp_path):
    p = tmp_path / "synthetic.parquet"
    df = make_synthetic_ohlcv(600)
    df.to_parquet(p)

    window_size = 3
    lookaheads = [3, 5, 10]

    config = {
        'environment': {'window_size': window_size},
        'training': {'lookaheads': lookaheads, 'reward_labels': ['Min-Max', 'Return', 'Sharpe']},
        'data': {'train_ratio': 0.6, 'val_ratio': 0.2, 'multi_horizon': {'enabled': False}}
    }

    dp = DataProcessor(config)
    train_df, val_df, test_df = dp.prepare_data_from_parquet(str(p), resample_timeframe=None)

    # Prefer train_df for environment
    df_norm = train_df
    assert df_norm is not None and len(df_norm) > 0

    best_col = f"expert_best_{window_size}"
    assert best_col in df_norm.columns, f"Missing {best_col} in prepared data"

    env = TradingEnvironment(df_norm, window_size=window_size, expert_selection='best')

    obs, info = env.reset()
    pre_step_idx = env.current_step

    action = 0
    next_obs, reward, terminated, truncated, info = env.step(action)

    expected = df_norm.iloc[pre_step_idx][best_col]

    assert np.isfinite(reward), "Returned reward must be finite"
    assert np.isclose(reward, expected, atol=1e-6), f"Reward {reward} != expected {expected}"
