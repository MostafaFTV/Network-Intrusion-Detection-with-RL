import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

class NetworkEnv(gym.Env):
    """
    Custom Gym environment for network intrusion detection.

    One FULL episode = one full pass over the dataset.
    - observation: normalized feature vector
    - action: Discrete(2): 0=normal, 1=attack
    - reward: +1 correct, -1 incorrect
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, X, y, shuffle=True):
        super().__init__()
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

        self.X = X.astype(np.float32)
        self.y = y.astype(int)
        self.n_features = self.X.shape[1]

        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.n_features,),
            dtype=np.float32
        )

        self.shuffle = bool(shuffle)
        self.dataset_size = len(self.y)
        self._rng = np.random.RandomState()

        self.indices = None
        self.pos = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)

        self.indices = np.arange(self.dataset_size)
        if self.shuffle:
            self._rng.shuffle(self.indices)

        self.pos = 0
        idx = self.indices[self.pos]
        obs = self.X[idx]

        return obs, {}

    def step(self, action):
        idx = self.indices[self.pos]
        true_label = int(self.y[idx])

        reward = 1.0 if int(action) == true_label else -1.0

        self.pos += 1
        terminated = self.pos >= self.dataset_size
        truncated = False

        if not terminated:
            next_idx = self.indices[self.pos]
            obs = self.X[next_idx]
        else:
            obs = np.zeros(self.n_features, dtype=np.float32)

        info = {
            "true_label": true_label,
            "dataset_pos": self.pos
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
