import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.trading_env import TradingEnvironment
from gymnasium.vector import AsyncVectorEnv
import pandas as pd
import numpy as np

def main():
    # Create a tiny synthetic DataFrame
    idx = pd.date_range('2024-01-01', periods=50, freq='T')
    df = pd.DataFrame({'Open':np.random.rand(50)+100,'High':np.random.rand(50)+101,'Low':np.random.rand(50)+99,'Close':np.random.rand(50)+100,'Volume':np.random.randint(1,100,50)}, index=idx)

    def make(data):
        return lambda: TradingEnvironment(data=data, window_size=5, expert_selection=None)

    vec = AsyncVectorEnv([make(df), make(df)])
    obs, infos = vec.reset()
    print('reset obs shape:', obs.shape)
    actions = [0,1]
    obs2, rewards, dones, truncs, infos = vec.step(actions)
    print('step obs shape:', obs2.shape, 'rewards', rewards, 'dones', dones, 'truncs', truncs)
    # `infos` from VectorEnv may be a list of dicts or a dict mapping env idx -> info
    sample_info = None
    if isinstance(infos, dict):
        # try index 0, else take first value
        if 0 in infos:
            sample_info = infos[0]
        else:
            # take first available entry
            try:
                sample_info = next(iter(infos.values()))
            except StopIteration:
                sample_info = {}
    elif isinstance(infos, (list, tuple, np.ndarray)):
        sample_info = infos[0] if len(infos) > 0 else {}
    else:
        sample_info = {}

    if isinstance(sample_info, dict):
        print('info sample keys:', list(sample_info.keys()))
    else:
        print('info sample (non-dict):', sample_info)


if __name__ == '__main__':
    main()
