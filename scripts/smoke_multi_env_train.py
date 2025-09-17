import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
from gymnasium.vector import AsyncVectorEnv

from src.environment.trading_env import TradingEnvironment
from src.models.agents.srddqn import SRDDQNAgent
from src.train import train_srddqn


# Top-level factory to ensure picklability on Windows spawn
def create_trading_env(data, window_size):
    return TradingEnvironment(data=data, window_size=window_size, expert_selection=None)


def main():
    # Minimal config
    config = {
        'seed': 42,
        'training': {
            'num_episodes': 1,
            'batch_size': 4,
        },
        'mixed_precision': False
    }

    # Create tiny data
    idx = pd.date_range('2024-01-01', periods=50, freq='T')
    df = pd.DataFrame({'Open':np.random.rand(50)+100,'High':np.random.rand(50)+101,'Low':np.random.rand(50)+99,'Close':np.random.rand(50)+100,'Volume':np.random.randint(1,100,50)}, index=idx)

    # Build picklable factories using functools.partial
    factories = [partial(create_trading_env, df, 5), partial(create_trading_env, df, 5)]

    vec = AsyncVectorEnv(factories)

    # Create dummy agent with matching dims
    obs_shape = vec.reset()[0].shape[1:]
    seq_len = 5
    feature_dim = obs_shape[1]
    state_dim = seq_len * feature_dim
    agent = SRDDQNAgent(state_dim=state_dim, action_dim=3, seq_len=seq_len, num_envs=2)

    # Run short training (uses train_srddqn signature: agent, train_env, val_env, config, model_path)
    metrics = train_srddqn(agent, vec, None, config, 'results/smoke_model')
    print('Smoke training metrics keys:', list(metrics.keys()))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
