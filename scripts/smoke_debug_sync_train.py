import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yaml

from src.environment.vectorized_env import SyncVectorEnv
from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.train import train_srddqn

# Minimal config for a tiny training run
config = {
    'seed': 42,
    'training': {
        'num_episodes': 1,
        'batch_size': 4,
        'num_envs': 2
    },
    'environment': {
        'window_size': 5,
        'initial_capital': 10000,
        'transaction_cost': 0.001
    },
    'model': {
        'dqn': {'hidden_size': 64, 'learning_rate': 1e-4, 'gamma': 0.99, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.995, 'target_update': 200, 'tau': 0.005},
        'reward_net': {'learning_rate': 1e-4, 'model_type': 'TimesNet'}
    }
}

# Create tiny data
idx = pd.date_range('2024-01-01', periods=50, freq='T')
df = pd.DataFrame({'Open':np.random.rand(50)+100,'High':np.random.rand(50)+101,'Low':np.random.rand(50)+99,'Close':np.random.rand(50)+100,'Volume':np.random.randint(1,100,50)}, index=idx)

# Factories
from functools import partial
factories = [partial(TradingEnvironment, data=df, window_size=5) for _ in range(config['training']['num_envs'])]

vec = SyncVectorEnv(factories)

# Create agent
obs_shape = vec.reset()[0].shape[1:]
seq_len = 5
feature_dim = obs_shape[1]
state_dim = seq_len * feature_dim
agent = SRDDQNAgent(state_dim=state_dim, action_dim=3, seq_len=seq_len, num_envs=config['training']['num_envs'])

metrics = train_srddqn(agent, vec, None, config, 'results/smoke_debug_model')
print('Done. Metrics keys:', list(metrics.keys()))
