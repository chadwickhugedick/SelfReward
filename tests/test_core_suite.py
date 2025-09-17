import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment

try:
    from src.utils.replay_buffer import ReplayBuffer
except Exception:
    ReplayBuffer = None


def make_synthetic(n=200):
    idx = pd.date_range('2024-01-01', periods=n, freq='min')
    price = np.linspace(100, 120, n)
    df = pd.DataFrame({'Open': price, 'High': price + 0.1, 'Low': price - 0.1, 'Close': price, 'Volume': np.ones(n)}, index=idx)
    return df


def test_data_processor_expert_columns(tmp_path):
    p = tmp_path / 'data.parquet'
    df = make_synthetic(300)
    df.to_parquet(p)

    config = {'environment': {'window_size': 5}, 'training': {'lookaheads': [3,5], 'reward_labels': ['Min-Max','Return','Sharpe']}, 'data': {'train_ratio':0.6,'val_ratio':0.2,'multi_horizon':{'enabled':False}}}
    dp = DataProcessor(config)
    train, val, test = dp.prepare_data_from_parquet(str(p), resample_timeframe=None)

    assert train is not None
    assert any(c.startswith('expert_best_') for c in train.columns)
    assert 'expert_Min-Max' in train.columns


def test_create_sequences_shape():
    df = make_synthetic(100)
    config = {'data': {}}
    dp = DataProcessor(config)
    window_size = 10
    X, y = dp.create_sequences(df, window_size)
    assert X.ndim == 3
    assert X.shape[1] == window_size
    assert X.shape[2] == df.shape[1]
    assert y.shape[0] == X.shape[0]


def test_env_expert_best_selection(tmp_path):
    p = tmp_path / 'data2.parquet'
    df = make_synthetic(200)
    df.to_parquet(p)

    config = {'environment': {'window_size': 3}, 'training': {'lookaheads':[3]}, 'data': {'train_ratio':0.6,'val_ratio':0.2,'multi_horizon':{'enabled':False}}}
    dp = DataProcessor(config)
    train, val, test = dp.prepare_data_from_parquet(str(p), resample_timeframe=None)
    df_norm = train
    env = TradingEnvironment(df_norm, window_size=3, expert_selection='best')
    obs, info = env.reset()
    # step once
    _, reward, *_ = env.step(0)
    # there should be an expert_best column and reward should be finite
    assert any(c.startswith('expert_best_') for c in df_norm.columns)
    assert np.isfinite(reward)


def test_replay_buffer_basic():
    if ReplayBuffer is None:
        return
    buf = ReplayBuffer(100)
    s = np.zeros((4,))
    buf.add(s, 1, 0.5, s, False)
    assert len(buf) == 1
    sample = buf.sample(1)
    assert sample is not None
