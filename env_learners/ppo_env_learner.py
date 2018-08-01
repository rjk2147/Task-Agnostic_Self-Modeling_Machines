import numpy as np
import tensorflow as tf
from baselines.ppo1 import pposgd_simple

from misc import models
from env_learners.env_learner import EnvLearner
from envs.env_learner_env import EnvLearnerEnv
from collections import deque

class PPOEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        self.buff_len = 10
        self.env = EnvLearnerEnv(env_in, self.buff_len)
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train(self, train, total_steps, valid=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        G, yS, yR, yD, X, S, A = self.__prep_data__(train, batch_size=0)
        self.env.load(train, G[0], yS[0], yR[0], yD[0], X[0], S[0], A[0])
        def policy_fn(name, ob_space, ac_space):
            return models.GenPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        self.ppo_model = pposgd_simple.learn(self.env, policy_fn,
                                             max_timesteps=len(train) - 100,
                                             timesteps_per_actorbatch=1000,
                                             clip_param=0.2, entcoeff=0.0,
                                             optim_epochs=total_steps, optim_stepsize=3e-4, optim_batchsize=64,
                                             gamma=0.99, lam=0.95, schedule='linear',
                                             )

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        obs = np.array([np.concatenate([obs_in / self.state_mul_const,
                                        action_in / self.act_mul_const])]).flatten()
        action, v = self.ppo_model.act(False, obs)
        new_obs = action
        return new_obs