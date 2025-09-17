import numpy as np


class SyncVectorEnv:
    """
    A minimal synchronous vectorized environment wrapper.
    Wraps multiple Gym-like env instances and exposes a simple
    batched `reset()` and `step(actions)` API.
    """

    def __init__(self, env_fns):
        """Create a vectorized env from a list of callables that return envs."""
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.is_vector_env = True

    def reset(self):
        """Reset all sub-environments and return a list/array of observations."""
        obs = []
        infos = []
        for env in self.envs:
            o, info = env.reset()
            obs.append(o)
            infos.append(info if isinstance(info, dict) else {})
        return np.stack(obs), infos

    def step(self, actions):
        """Step each environment with the corresponding action.

        Args:
            actions: iterable of actions with length == num_envs

        Returns:
            obs_batch, reward_batch, done_batch, info_list
        """
        obs_batch = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for env, a in zip(self.envs, actions):
            obs, rew, terminated, truncated, info = env.step(int(a))
            obs_batch.append(obs)
            rewards.append(rew)
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))
            infos.append(info if isinstance(info, dict) else {})

        return np.stack(obs_batch), np.array(rewards), np.array(terminations), np.array(truncations), infos

    def render(self, mode='human'):
        for env in self.envs:
            env.render(mode=mode)

    def __len__(self):
        return self.num_envs
