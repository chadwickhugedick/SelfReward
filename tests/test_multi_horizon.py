import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment


def make_sample_df(n=100, freq='T'):
    idx = pd.date_range('2024-01-01', periods=n, freq=freq)
    df = pd.DataFrame({'Open':np.linspace(100,110,n),'High':np.linspace(101,111,n),'Low':np.linspace(99,109,n),'Close':np.linspace(100,110,n),'Volume':np.random.randint(1,100,n)}, index=idx)
    return df


def test_compute_multi_horizon_defaults():
    cfg = {'data': {}}
    dp = DataProcessor(cfg)
    df = make_sample_df(50)
    mh = dp.compute_multi_horizon(df)
    # Should produce columns with dh_ and wh_ prefixes
    assert any(c.startswith('dh_') for c in mh.columns)
    assert any(c.startswith('wh_') for c in mh.columns)
    # Reindexed to original length
    assert len(mh) == len(df)


def test_prepare_data_with_mh_enabled():
    cfg = {'data': {'multi_horizon': {'enabled': True, 'horizons': {'dh':'1D','wh':'1W'}}}, 'environment': {'window_size':5}}
    dp = DataProcessor(cfg)
    df = make_sample_df(60)
    # call prepare_data_from_parquet-like flow by using compute + merge
    mh = dp.compute_multi_horizon(df, horizons=cfg['data']['multi_horizon']['horizons'])
    merged = pd.concat([df, mh], axis=1)
    env = TradingEnvironment(data=merged, window_size=5)
    obs, info = env.reset()
    assert obs.shape[0] == 5
    # feature dimension should be original + mh columns + 1
    expected_feat = merged.shape[1] + 1
    assert obs.shape[1] == expected_feat
