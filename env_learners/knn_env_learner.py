import numpy as np
from sklearn import neighbors

from env_learners.env_learner import EnvLearner


class KNNEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        self.knn = neighbors.KNeighborsRegressor(5, weights='distance')

    def train(self, train, total_steps, valid=None,log_interval=10, early_stopping=-1, saver=None, save_str=None):
        X = []
        Y = []
        for (obs, action, r, new_obs, done, episode_step) in train:
            obs = obs / self.state_mul_const
            action = action / self.act_mul_const
            new_obs = new_obs / self.state_mul_const
            X.append(np.array([np.concatenate([obs, action])]).flatten())
            Y.append(new_obs)
        self.knn.fit(X, Y)

    def get_loss(self, data):
        return 0,0,0

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        obs_in = obs_in/self.state_mul_const
        action_in = action_in/self.act_mul_const
        new_obs = self.knn.predict([np.array([np.concatenate([obs_in, action_in])]).flatten()])[0]
        return new_obs