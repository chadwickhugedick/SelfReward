import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure workspace root is on sys.path so `src` is importable during tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_processor import DataProcessor


def test_expert_label_columns(tmp_path):
    # Create minimal config
    config = {
        'environment': {'window_size': 5},
        'training': {'lookaheads': [3, 5], 'reward_labels': ['Min-Max', 'Return', 'Sharpe']},
        'data': {'train_ratio': 0.6, 'val_ratio': 0.2}
    }

    # Create small synthetic OHLCV DataFrame and write to parquet
    # Use more rows so technical indicators (SMA/EMA/ATR) don't drop all rows
    idx = pd.date_range('2024-01-01', periods=200, freq='min')
    df = pd.DataFrame({
        'Open': np.linspace(100, 110, 200),
        'High': np.linspace(101, 111, 200),
        'Low': np.linspace(99, 109, 200),
        'Close': np.linspace(100, 110, 200),
        'Volume': np.random.randint(1, 100, 200)
    }, index=idx)

    p = tmp_path / "test.parquet"
    df.to_parquet(p)

    dp = DataProcessor(config)
    train_df, val_df, test_df = dp.prepare_data_from_parquet(str(p), resample_timeframe=None)

    # window_size is 5, lookaheads include 3 and 5
    # We expect columns like 'expert_Min-Max_3' and 'expert_best_5' to be present
    expected_best_3 = 'expert_best_3'
    expected_best_5 = 'expert_best_5'
    expected_plain_minmax = 'expert_Min-Max'

    assert expected_best_3 in train_df.columns, f"Missing {expected_best_3} in train_df"
    assert expected_best_5 in train_df.columns, f"Missing {expected_best_5} in train_df"
    assert expected_plain_minmax in train_df.columns, f"Missing plain {expected_plain_minmax} in train_df"
