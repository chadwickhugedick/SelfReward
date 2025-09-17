from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment

# Minimal config
config = {'environment': {'window_size': 5}, 'data': {}}

# Create tiny data
idx = pd.date_range('2024-01-01', periods=50, freq='T')
df = pd.DataFrame({'Open':np.linspace(100,110,50),'High':np.linspace(101,111,50),'Low':np.linspace(99,109,50),'Close':np.linspace(100,110,50),'Volume':np.random.randint(1,100,50)}, index=idx)

print('Original df shape:', df.shape)

dp = DataProcessor(config)
# Compute multi-horizon features (daily/weekly defaults)
horizon = dp.compute_multi_horizon(df)
print('Horizon df shape:', horizon.shape)
print('Horizon columns sample:', list(horizon.columns)[:10])

# Merge horizons into base df
merged = pd.concat([df, horizon], axis=1)
print('Merged df shape:', merged.shape)
print('Merged columns count:', len(merged.columns))
print('Sample columns:', merged.columns[:12])

# Instantiate environment
env = TradingEnvironment(data=merged, window_size=5, expert_selection=None)
obs, info = env.reset()
print('Reset observation shape:', obs.shape)

# Step once
obs2, reward, terminated, truncated, info = env.step(0)
print('After step observation shape:', obs2.shape)
print('Reward:', reward)
print('Info keys:', list(info.keys()))

print('Done smoke test')
