
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class EnvLearnerEnv(gym.Env):
    def __init__(self, env_in):
        self.data = []
        act_dim = env_in.action_space.shape[0]
        state_dim = env_in.observation_space.shape[0]
        self.state_mul_const = env_in.observation_space.high
        self.act_mul_const = env_in.action_space.high
        self.i = 0
        self.num_envs = 1

        # state_dim
        act_ones = np.ones(shape=env_in.observation_space.shape)
        self.action_space = spaces.Box(high=act_ones, low=-act_ones)

        obs_ones = np.ones(shape=(act_dim+state_dim,))
        self.observation_space = spaces.Box(high=obs_ones, low=-obs_ones)

    def load(self, data):
        self.data = data

    def reset(self):
        (obs_in, action_in, _, new_obs_in, done, episode_step) = self.data[self.i]
        obs = np.array([np.concatenate([obs_in/self.state_mul_const,
                                        action_in/self.act_mul_const])]).flatten()
        return obs

    def step(self, action):
        (obs_in, action_in, _, new_obs_in, done, episode_step) = self.data[self.i]
        self.i += 1
        # obs = np.array([np.concatenate([obs_in/self.state_mul_const,
        #                                 action_in/self.act_mul_const])]).flatten()
        r = -np.linalg.norm(new_obs_in / self.state_mul_const - action) / action.shape[0]
        if not done and self.i < len(self.data):
            new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                                            self.data[self.i][1] / self.act_mul_const])]).flatten()
        else:
            new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                                            np.zeros_like(action_in)])]).flatten()
        return new_obs, r, done, {}