import time

import numpy as np
import tensorflow as tf
from collections import deque

from misc import models
from env_learners.env_learner import EnvLearner
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

class DDPGEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # from baselines.ddpg.models import Actor, Critic
        # Parse noise_type
        action_noise = None
        param_noise = None
        noise_type = 'adaptive-param_0.2'
        layer_norm = True
        nb_actions = self.state_dim
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                            sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        # Configure components.

        self.buff_len = 10
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        obs_space = (self.buff_init[0].size * self.buff_len,)
        self.memory = Memory(limit=int(1e6), action_shape=env_in.observation_space.shape, observation_shape=obs_space)
        self.critic = models.Critic(layer_norm=layer_norm)
        self.actor = models.Actor(nb_actions, layer_norm=layer_norm)

        self.agent = DDPG(self.actor, self.critic, self.memory, obs_space, env_in.observation_space.shape,
                          gamma=0.99, tau=0.01, normalize_returns=False,
                          normalize_observations=True,
                          batch_size=64, action_noise=action_noise, param_noise=param_noise,
                          critic_l2_reg=1e-2,
                          actor_lr=1e-5, critic_lr=1e-5, enable_popart=False, clip_norm=None,
                          reward_scale=1.)


    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())
        self.agent.initialize(self.sess)

    def train(self, train, total_steps, valid=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        G, yS, yR, yD, X, S, A = self.__prep_data__(train, batch_size=0)
        X = X[0]
        S = S[0]
        self.agent.reset()
        # max_action = self.env.action_space.high
        batch_size = 64
        t = 0
        episode_reward = 0
        episode_step = 0
        episodes = 0
        epoch_episodes = 0
        epoch_episode_rewards = []
        nb_epoch_cycles = 10
        nb_rollout_steps = 100
        nb_epochs = int(len(train)/(nb_epoch_cycles*nb_rollout_steps))

        nb_train_steps = total_steps
        param_noise_adaption_interval = 50
        i = 0

        for epoch in range(nb_epochs):
            start_time = time.time()
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.

                    # (obs_in, action_in, _, new_obs_in, done, episode_step) = train[i]

                    # obs = np.array([np.concatenate([obs_in/self.state_mul_const,
                    #                                 action_in/self.act_mul_const])]).flatten()
                    obs = X[i]
                    done = train[i][4]
                    action, q = self.agent.pi(obs, apply_noise=True, compute_Q=True)
                    r = -np.linalg.norm(S[i]/self.state_mul_const-action)/action.shape[0]

                    # if not done and i < len(train):
                    #     new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                    #                                     train[i][1] / self.act_mul_const])]).flatten()
                    # else:
                    #     new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                    #                                     np.zeros_like(action_in)])]).flatten()
                    if i < len(train):
                        new_obs = X[i+1]
                    else:
                        new_obs = np.zeros_like(X[i])
                    t += 1
                    i += 1
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    self.agent.store_transition(obs, action, r, new_obs, done)

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_reward = 0.
                        epoch_episodes += 1
                        episodes += 1

                        self.agent.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if self.memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = self.agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = self.agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    self.agent.update_target_net()
            print('Epoch '+str(epoch)+'/'+str(nb_epochs)+' with avg rew of: '+
                  str(sum(epoch_episode_rewards)/len(epoch_episode_rewards))+' in '+str(time.time()-start_time)+'s')
            if epoch%log_interval == 0 and epoch > 0:
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    print("Model saved in path: %s" % save_path)
        if saver is not None and save_str is not None:
            save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
            print("Model saved in path: %s" % save_path)


    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        obs = obs_in/self.state_mul_const
        action = action_in/self.act_mul_const
        if save:
            if episode_step == 0:
                self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
            self.buffer.append(np.array([np.concatenate([obs, action])]).flatten())
        else:
            if buff is None:
                buff = self.buffer.copy()
            if episode_step == 0:
                buff = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
            buff.append(np.array([np.concatenate([obs, action])]).flatten())

        if buff is not None:
            x = np.array([np.concatenate(buff).flatten()])[0]
        else:
            x = np.array([np.concatenate(self.buffer).flatten()])[0]
        new_obs, _ = self.agent.pi(x, apply_noise=True, compute_Q=True)
        return new_obs