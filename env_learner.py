import tensorflow as tf
import numpy as np
import math
from collections import deque
import logger
import self_models
import losses
import time
from env_learner_env import EnvLearnerEnv
from baselines.ppo1 import mlp_policy, pposgd_simple

from sklearn import neighbors


class EnvLearner:
    def __init__(self, env_in, model='gan'):
        self.state_mul_const = env_in.observation_space.high
        self.act_mul_const = env_in.action_space.high
        self.act_dim = env_in.action_space.shape[0]
        self.state_dim = env_in.observation_space.shape[0]

        if model == 'knn':
            self.model_type = model
            self.knn = neighbors.KNeighborsRegressor(5, weights='distance')
        elif model == 'rand_search':
            self.model_type = model
            self.buff_init = [np.zeros(state_dim+act_dim)]
            self.buff_len = 5
            self.seq_len = 1
            self.max_seq_len = 10

            self.min_loss = 1000000000
            self.min_weights = []

            self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.buff_init[0].size * self.buff_len]))
            self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, state_dim * self.max_seq_len]))
            self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, act_dim * self.max_seq_len]))
            a_seq_split = tf.split(self.a_seq, self.max_seq_len, 1)
            y_seq_split = tf.split(self.y_seq, self.max_seq_len, 1)

            input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)
            self.out_state_raw = self_models.generator_model(input_tmp_seq, state_dim, drop_rate=0.0)
            self.out_state = self.out_state_raw*self.state_mul_const
            self.loss_state = losses.loss_p(self.out_state_raw, y_seq_split[0])
        elif model == 'ddpg':
            # from baselines.ddpg.models import Actor, Critic
            from baselines.ddpg.ddpg import DDPG
            from baselines.ddpg.memory import Memory
            from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
            self.model_type = model
            # Parse noise_type
            action_noise = None
            param_noise = None
            noise_type = 'adaptive-param_0.2'
            layer_norm = True
            nb_actions = state_dim
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
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

            # Configure components.

            obs_space = (act_dim+state_dim,)
            self.memory = Memory(limit=int(1e6), action_shape=env_in.observation_space.shape, observation_shape=obs_space)
            self.critic = self_models.Critic(layer_norm=layer_norm)
            self.actor = self_models.Actor(nb_actions, layer_norm=layer_norm)

            self.agent = DDPG(self.actor, self.critic, self.memory, obs_space, env_in.observation_space.shape,
                         gamma=0.99, tau=0.01, normalize_returns=False,
                         normalize_observations=True,
                         batch_size=64, action_noise=action_noise, param_noise=param_noise,
                         critic_l2_reg=1e-2,
                         actor_lr=1e-5, critic_lr=1e-5, enable_popart=False, clip_norm=None,
                         reward_scale=1.)
        elif model == 'ppo':
            self.model_type = 'ppo'
            self.env = EnvLearnerEnv(env_in)
        else:
            self.model_type = 'gan'
            # Initialization
            self.buff_len = 5
            self.seq_len = 1
            self.max_seq_len = 10
            self.last_r = np.array([0.0]).flatten()
            self.buff_init = [np.zeros(self.state_dim+self.act_dim)]
            self.seq_init = [np.zeros(self.act_dim)]
            self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
            dropout_rate = 0.5
            lr_disc = 1e-5
            lr_gen = 1e-5
            logger.info('General Stats: ')
            logger.info('Drop Rate: ' + str(dropout_rate))
            logger.info('Buffer Len: ' + str(self.buff_len))
            logger.info('Start Sequence Len: ' + str(self.seq_len))
            logger.info('End Sequence Len: ' + str(self.max_seq_len))
            logger.info('gan_model:')
            logger.info('Learning Rate: ' + str(lr_disc))
            logger.info('Learning Rate: ' + str(lr_gen))

            """ State Prediction """
            self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.buff_init[0].size * self.buff_len]))
            self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim * self.max_seq_len]))
            self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim * self.max_seq_len]))

            a_seq_split = tf.split(self.a_seq, self.max_seq_len, 1)
            y_seq_split = tf.split(self.y_seq, self.max_seq_len, 1)

            input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)
            self.out_state_raw = self_models.generator_model(input_tmp_seq, self.state_dim, drop_rate=dropout_rate)

            self.out_state = self.out_state_raw*self.state_mul_const
            self.loss_seq = 0.0
            self.loss_last = 0.0
            out_states = []
            out_states.append(self.out_state_raw)
            self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[0])
            self.loss_last += losses.loss_p(out_states[-1], tf.slice(input_tmp_seq[-1], [0, 0], [-1, self.state_dim]))
            for i in range(1, self.seq_len):
                state_tmp = tf.slice(self.x_seq[:],
                                       [0, self.buff_init[0].size],
                                       [-1, -1]
                                       )
                state_tmp = tf.concat([state_tmp, out_states[-1]], axis=1)
                input_tmp = tf.concat([state_tmp, a_seq_split[i]], axis=1)

                input_tmp_seq = tf.split(input_tmp, self.buff_len, 1)
                out_state_raw_tmp = self_models.generator_model(input_tmp_seq, self.state_dim, drop_rate=dropout_rate)
                out_states.append(out_state_raw_tmp)
                self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[i])
                self.loss_last += losses.loss_p(out_states[-1], out_states[-2])

            self.out_state_seq = tf.concat(out_states, axis=1)

            self.loss_state = self.loss_seq

            self.train_step_state = tf.train.AdamOptimizer(lr_gen).minimize(self.loss_state)

            """ GAN Stuff """
            x_seq = []
            g_seq = []
            out_seq_split = tf.split(self.out_state_seq, self.seq_len, 1)
            for i in range(self.seq_len):
                x_seq.append(tf.concat([y_seq_split[i], a_seq_split[i]], axis=1))
                g_seq.append(tf.concat([out_seq_split[i], a_seq_split[i]], axis=1))

            x_in = x_seq
            g_in = g_seq
            self.Dx = self_models.discriminator_model(x_in, drop_rate=dropout_rate)
            self.Dg = self_models.discriminator_model(g_in, drop_rate=dropout_rate)
            var_d = tf.trainable_variables('discriminator')
            var_g = tf.trainable_variables('generator')
            g_lambda = 1.0
            p_lambda = 10.0
            t_lambda = 0.0

            """ Vanilla GAN """
            # self.n_d = 1
            # self.disc_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1-self.Dg))
            # self.g_loss = -tf.reduce_mean(tf.log(self.Dg))
            # self.gen_loss =  g_lambda*self.g_loss + p_lambda * self.loss_seq
            # self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
            # self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

            """ WGAN-GP """
            self.n_d = 1
            epsilon = 0.01
            gp_lambda = 10

            self.disc_loss = tf.reduce_mean(self.Dg) - tf.reduce_mean(self.Dx)
            self.g_loss = -tf.reduce_mean(self.Dg)
            self.gen_loss =  g_lambda*self.g_loss + \
                             p_lambda * self.loss_seq + \
                             t_lambda * self.loss_last
            x_hat = epsilon*self.Dx + (1-epsilon)*self.Dg
            grad_list = tf.gradients(x_hat, var_d)[2:]
            GP = 0.0
            for layer in grad_list:
                GP += gp_lambda * (tf.sqrt(tf.reduce_sum(tf.square(layer))) - 1) ** 2
            self.disc_loss += GP
            self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
            self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

    def init_gan_losses(self):

        a_seq_split = tf.split(self.a_seq, self.max_seq_len, 1)
        y_seq_split = tf.split(self.y_seq, self.max_seq_len, 1)
        input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)
        self.loss_seq = 0.0
        self.loss_last = 0.0
        out_states = []
        out_states.append(self.out_state_raw)
        self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[0])
        self.loss_last += losses.loss_p(out_states[-1], tf.slice(input_tmp_seq[-1], [0, 0], [-1, self.state_dim]))
        for i in range(1, self.seq_len):
            state_tmp = tf.slice(self.x_seq[:],
                                 [0, self.buff_init[0].size],
                                 [-1, -1]
                                 )
            state_tmp = tf.concat([state_tmp, out_states[-1]], axis=1)
            input_tmp = tf.concat([state_tmp, a_seq_split[i]], axis=1)

            input_tmp_seq = tf.split(input_tmp, self.buff_len, 1)
            out_state_raw_tmp = self_models.generator_model(input_tmp_seq, self.state_dim, drop_rate=0.5)
            out_states.append(out_state_raw_tmp)
            self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[i])
            self.loss_last += losses.loss_p(out_states[-1], out_states[-2])

        self.out_state_seq = tf.concat(out_states, axis=1)

        self.loss_state = self.loss_seq

        """ GAN Stuff """
        x_seq = []
        g_seq = []
        out_seq_split = tf.split(self.out_state_seq, self.seq_len, 1)
        for i in range(self.seq_len):
            x_seq.append(tf.concat([y_seq_split[i], a_seq_split[i]], axis=1))
            g_seq.append(tf.concat([out_seq_split[i], a_seq_split[i]], axis=1))

        self.x_in = x_seq
        self.g_in = g_seq
        var_d = tf.trainable_variables('discriminator')
        var_g = tf.trainable_variables('generator')
        g_lambda = 1.0
        p_lambda = 10.0
        t_lambda = 0.0

        """ Vanilla GAN """
        # self.n_d = 1
        # self.disc_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1-self.Dg))
        # self.g_loss = -tf.reduce_mean(tf.log(self.Dg))
        # self.gen_loss =  g_lambda*self.g_loss + p_lambda * self.loss_seq
        # self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
        # self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

        """ WGAN-GP """
        self.n_d = 1
        epsilon = 0.01
        gp_lambda = 10

        self.disc_loss = tf.reduce_mean(self.Dg) - tf.reduce_mean(self.Dx)
        self.g_loss = -tf.reduce_mean(self.Dg)
        self.gen_loss = g_lambda * self.g_loss + \
                        p_lambda * self.loss_seq + \
                        t_lambda * self.loss_last
        x_hat = epsilon * self.Dx + (1 - epsilon) * self.Dg
        grad_list = tf.gradients(x_hat, var_d)[2:]
        GP = 0.0
        for layer in grad_list:
            GP += gp_lambda * (tf.sqrt(tf.reduce_sum(tf.square(layer))) - 1) ** 2
        self.disc_loss += GP
        # self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
        # self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())
        if self.model_type == 'ddpg':
            self.agent.initialize(self.sess)

    def __batch__(self, data, batch_size):
        batches = []
        while len(data) >= batch_size:
            batches.append(data[:batch_size])
            data = data[batch_size:]
        return batches

    def __prep_data__(self, data, batch_size=32):
        G = []
        X = []
        A = []
        yS = []
        yR = []
        yD = []
        S = []
        is_new = True

        t = 0
        for (obs, action, r, new_obs, done, episode_step) in data:
            # Normalizing to [-1, 1]
            obs = obs/self.state_mul_const
            action = action/self.act_mul_const
            new_obs = new_obs/self.state_mul_const
            zD = np.array([int(done)])

            if is_new:
                x_episode = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
                x_episode.append(np.array([np.concatenate([obs, action])]).flatten())
                is_new = False
                t = 0

                if len(A) > 0:
                    for i in range(self.max_seq_len-1):
                        a_seq.append(np.concatenate([np.zeros(action.size)]))
                        y_seq.append(yS[-1])
                        A.append(np.concatenate(a_seq).flatten())
                        S.append(np.concatenate(y_seq).flatten())
                a_seq = deque([np.zeros(action.size)] * self.max_seq_len, maxlen=self.max_seq_len)
                y_seq = deque([np.zeros(new_obs.size)] * self.max_seq_len, maxlen=self.max_seq_len)
            else:
                x_episode.append(np.array([np.concatenate([obs, action])]).flatten())
            t += 1
            a_seq.append(action)
            y_seq.append(new_obs)

            if t > self.max_seq_len-1:
                A.append(np.concatenate(a_seq).flatten())
                S.append(np.concatenate(y_seq).flatten())
            yD.append(zD.flatten())
            yS.append(np.array([new_obs]).flatten())
            yR.append(np.array([r]))
            X.append(np.concatenate(x_episode).flatten())
            if done:
                is_new = True
        if len(A) > 0:
            for i in range(self.max_seq_len-1):
                a_seq.append(np.concatenate([np.zeros(action.size)]))
                y_seq.append(yS[-1])
                A.append(np.concatenate(a_seq).flatten())
                S.append(np.concatenate(y_seq).flatten())

        assert len(X) == len(yS) == len(yR) == len(yD) == len(S) == len(A)

        p = np.random.permutation(len(X))
        X = np.array(X)[p]
        yS = np.array(yS)[p]
        yR = np.array(yR)[p]
        yD = np.array(yD)[p]
        S = np.array(S)[p]
        A = np.array(A)[p]


        if batch_size > 0:
            X = self.__batch__(X, batch_size)
            yS = self.__batch__(yS, batch_size)
            yR = self.__batch__(yR, batch_size)
            yD = self.__batch__(yD, batch_size)
            S = self.__batch__(S, batch_size)
            A = self.__batch__(A, batch_size)
        else:
            G = [G]
            X = [X]
            yS = [yS]
            yR = [yR]
            yD = [yD]
            S = [S]
            A = [A]
        return G, yS, yR, yD, X, S, A

    def train_epoch(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, self.buff_len)
        n_d = self.n_d
        lGen = 0.0
        lDisc = 0.0
        lC = 0.0
        for i in range(len(X)):
            dg, dx, dLoss, _ = self.sess.run([self.Dg, self.Dx, self.disc_loss, self.train_step_disc],
                                             feed_dict={self.x_seq: X[i],
                                                        self.y_seq: S[i],
                                                        self.a_seq: A[i]
                                                        })  # Update the discriminator
            lDisc += dLoss

        for i in range(len(X)):
            if n_d <= 1 or i % n_d == n_d - 1:
                _, gLoss, ls = self.sess.run([self.train_step_gen, self.gen_loss, self.loss_seq], feed_dict={self.x_seq: X[i],
                                                                                          self.y_seq: S[i],
                                                                                          self.a_seq: A[i]
                                                                                          })  # Update the generator
                lGen += gLoss
                lC += ls
        return n_d * lGen / len(X), lDisc / len(X), n_d * lC / len(X)

    def train(self, train, total_steps, valid=None, logger=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        if self.model_type == 'knn':
            X = []
            Y = []
            for (obs, action, r, new_obs, done, episode_step) in train:
                obs = obs/self.state_mul_const
                action = action/self.act_mul_const
                new_obs = new_obs/self.state_mul_const
                X.append(np.array([np.concatenate([obs, action])]).flatten())
                Y.append(new_obs)
            self.knn.fit(X, Y)
        elif self.model_type == 'rand':
            epoch_min_loss = 100000000
            G, yS, yR, yD, X, S, A = self.__prep_data__(data, 10000)
            # vars = [v for v in tf.trainable_variables()]
            # for var in vars:
            #     tf.assign(var, tf.random_normal(shape=var.get_shape().as_list(), mean=0.0, stddev=0.1))
            for i in range(50):
                self.sess.run(tf.global_variables_initializer())
                loss = 0
                for i in range(len(X)):
                    batch_loss = self.sess.run([self.loss_state], feed_dict={self.x_seq: X[i],
                                                                                self.y_seq: S[i],
                                                                                self.a_seq: A[i]
                                                                                })[0]
                    loss += batch_loss
                if loss < self.min_loss:
                    epoch_min_loss = loss
                    vars = [v for v in tf.trainable_variables()]
                    self.min_weights = []
                    for var in vars:
                        self.min_weights.append(self.sess.run(var))
            if epoch_min_loss == 100000000:
                vars = [v for v in tf.trainable_variables()]
                for i in range(len(vars)):
                    tf.assign(vars[i], self.min_weights[i])
                loss = 0
                for i in range(len(X)):
                    batch_loss = self.sess.run([self.loss_state], feed_dict={self.x_seq: X[i],
                                                                                self.y_seq: S[i],
                                                                                self.a_seq: A[i]
                                                                                })[0]
                    loss += batch_loss
            return epoch_min_loss, 0
        elif self.model_type == 'ddpg':
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

                        (obs_in, action_in, _, new_obs_in, done, episode_step) = train[i]
                        i += 1

                        obs = np.array([np.concatenate([obs_in/self.state_mul_const,
                                                        action_in/self.act_mul_const])]).flatten()

                        action, q = self.agent.pi(obs, apply_noise=True, compute_Q=True)
                        # assert action.shape == self.env.action_space.shape
                        # assert max_action.shape == action.shape

                        r = -np.linalg.norm(new_obs_in/self.state_mul_const-action)/action.shape[0]

                        if not done and i < len(train):
                            new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                                                            train[i][1] / self.act_mul_const])]).flatten()
                        else:
                            new_obs = np.array([np.concatenate([new_obs_in / self.state_mul_const,
                                                            np.zeros_like(action_in)])]).flatten()
                        # new_obs, r, done, info = self.env.step(max_action * action)
                        t += 1
                        episode_reward += r
                        episode_step += 1

                        # Book-keeping.
                        # epoch_actions.append(action)
                        # epoch_qs.append(q)
                        self.agent.store_transition(obs, action, r, new_obs, done)
                        # obs = new_obs

                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            # episode_rewards_history.append(episode_reward)
                            # epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
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
        elif self.model_type == 'ppo':
            self.env.load(train)
            def policy_fn(name, ob_space, ac_space):
                return self_models.GenPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
            self.ppo_model = pposgd_simple.learn(self.env, policy_fn,
                    max_timesteps=len(train)-100,
                    timesteps_per_actorbatch=1000,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=total_steps, optim_stepsize=3e-4, optim_batchsize=64,
                    gamma=0.99, lam=0.95, schedule='linear',
                )
        elif self.model_type == 'gan':
            min_loss = 10000000000
            stop_count = 0

            seq_i = 0
            seq_idx= [1]*(self.max_seq_len-self.seq_len+1)
            for j in range(1, self.max_seq_len-self.seq_len+1):
                seq_tmp = self.max_seq_len-j
                seq_idx[j] = (seq_tmp+1)*seq_idx[j-1]/seq_tmp
            seq_idx.reverse()
            mul_const = total_steps/sum(seq_idx)
            for j in range(len(seq_idx)):
                seq_idx[j]=round(mul_const*seq_idx[j])
                if j > 0:
                    seq_idx[j] += seq_idx[j-1]
            for i in range(total_steps):
                # if i > 0 and i % (
                #     total_steps / self.max_seq_len) == 0 and self.seq_len < self.max_seq_len:
                if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                    self.seq_len += 1
                    seq_i += 1
                    self.init_gan_losses()
                    print('Sequence Length: ' + str(self.seq_len))

                if i % log_interval == 0 and i > 0 and logger is not None and valid is not None:
                    (vGen, vDisc, vC) = self.get_loss(valid)
                    logger.info('Epoch: ' + str(i) + '/' + str(total_steps))
                    logger.info('Valid Loss')
                    logger.info('Gen:  ' + str(vGen))
                    logger.info('Disc: ' + str(vDisc))
                    logger.info('Close: ' + str(vC))
                    logger.info()
                    if saver is not None and save_str is not None:
                        save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                        logger.info("Model saved in path: %s" % save_path)
                start = time.time()
                tlGen, tlDisc, tlC = self.train_epoch(train)
                duration = time.time() - start
                if tlGen < min_loss:
                    min_loss = tlGen
                    stop_count = 0
                else:
                    stop_count += 1
                if stop_count > early_stopping and early_stopping > 0:
                    break
                if i % log_interval != 0 and i > 0 and logger is not None:
                    logger.info('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                    logger.info('Train Loss')
                    logger.info('Gen:  ' + str(tlGen))
                    logger.info('Disc: ' + str(tlDisc))
                    logger.info('Close: ' + str(tlC))
                    logger.info()
            if logger is not None and valid is not None:
                (vGen, vDisc, vC) = self.get_loss(valid)
                logger.info('Final Epoch: ')
                logger.info('Valid Loss')
                logger.info('Gen:  ' + str(vGen))
                logger.info('Disc: ' + str(vDisc))
                logger.info('Close: ' + str(vC))
                logger.info()
            if saver is not None and save_str is not None:
                save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                logger.info("Final Model saved in path: %s" % save_path)

    def get_loss(self, data):
        if self.model_type == 'knn':
            return 0,0,0
        elif self.model_type == 'rand':
            G, yS, yR, yD, X, S, A = self.__prep_data__(data, 0)
            loss = 0
            for i in range(len(X)):
                batch_loss = self.sess.run([self.loss_state], feed_dict={self.x_seq: X[i],
                                                                            self.y_seq: S[i],
                                                                            self.a_seq: A[i]
                                                                            })[0]
                loss += batch_loss
            return 0,0,loss
        elif self.model_type == 'gan':
            G, yS, yR, yD, X, S, A = self.__prep_data__(data, self.buff_len)
            lC = 0.0
            lGen = 0.0
            lDisc = 0.0
            for i in range(len(X)):

                (lD, lG, ls) = self.sess.run([self.disc_loss, self.gen_loss, self.loss_seq], feed_dict={self.x_seq: X[i],
                                                                  self.y_seq: S[i],
                                                                  self.a_seq: A[i]
                                                                 })

                # lDisc += self.sess.run(self.disc_loss, feed_dict={self.x_seq: X[i],
                #                                                   self.y_seq: S[i],
                #                                                   self.a_seq: A[i]
                #                                                  })
                # lGen += self.sess.run(self.g_loss, feed_dict={self.x_seq: X[i],
                #                                                 self.y_seq: S[i],
                #                                                 self.a_seq: A[i]
                #                                                 })
                #
                # osr, ls, os = self.sess.run([self.out_state_raw, self.loss_state, self.out_state], feed_dict={self.x_seq: X[i],
                #                                                                               self.y_seq: S[i],
                #                                                                               self.a_seq: A[i]
                #                                                                               })
                lC += ls
                lGen += lG
                lDisc += lD
            return lGen / len(X), lDisc / len(X), lC / len(X)

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        if self.model_type == 'knn':
            obs_in = obs_in/self.state_mul_const
            action_in = action_in/self.act_mul_const
            new_obs = self.knn.predict([np.array([np.concatenate([obs_in, action_in])]).flatten()])[0]
        elif self.model_type == 'ddpg':
            obs = np.array([np.concatenate([obs_in / self.state_mul_const,
                                            action_in / self.act_mul_const])]).flatten()
            new_obs, _ = self.agent.pi(obs, apply_noise=True, compute_Q=True)
        elif self.model_type == 'ppo':
            obs = np.array([np.concatenate([obs_in / self.state_mul_const,
                                            action_in / self.act_mul_const])]).flatten()
            action, v = self.ppo_model.act(False, obs)
            new_obs = action
        elif self.model_type == 'gan':
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
                x = np.array([np.concatenate(buff).flatten()])
            else:
                x = np.array([np.concatenate(self.buffer).flatten()])
            new_obs = self.sess.run(self.out_state, feed_dict={self.x_seq: x}).flatten()
        return new_obs
