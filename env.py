import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class NetworkEnv(gym.Env):
    """
    Custom Gym environment for network intrusion detection.
    Each sample is treated as one timestep.
    - observation: normalized feature vector
    - action: Discrete(2): 0=normal, 1=attack
    - reward: +1 for correct prediction, -1 for incorrect
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, X, y, episode_length=256, shuffle=True):
        super().__init__()
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        self.X = X.astype(np.float32)
        self.y = y.astype(int)
        self.n_features = self.X.shape[1]

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float32)

        self.episode_length = int(episode_length)
        self.shuffle = bool(shuffle)
        self._max_steps = min(self.episode_length, len(self.y))
        self._rng = np.random.RandomState()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        self.indices = np.arange(len(self.y))
        if self.shuffle:
            self._rng.shuffle(self.indices)

        self.pos = 0
        self.steps = 0
        self.current_index = int(self.indices[self.pos])
        obs = self.X[self.current_index]

        # Return observation and an empty info dictionary for compatibility
        return obs, {}

    def step(self, action):
        true_label = int(self.y[self.current_index])
        reward = 1.0 if int(action) == true_label else -1.0

        self.pos += 1
        self.steps += 1
        terminated = self.steps >= self._max_steps
        truncated = self.pos >= len(self.indices)

        if not (terminated or truncated):
            self.current_index = int(self.indices[self.pos])
            obs = self.X[self.current_index]
        else:
            obs = np.zeros(self.n_features, dtype=np.float32)

        info = {'true_label': true_label}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
