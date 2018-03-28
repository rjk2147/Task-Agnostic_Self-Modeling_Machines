import tensorflow as tf
import numpy as np
import math
from collections import deque
import logger
import self_models
import losses


class EnvLearner:
    def __init__(self, env_in):
        # Initialization
        self.buff_len = 5
        self.seq_len = 1
        self.max_seq_len = 10
        self.last_r = np.array([0.0]).flatten()
        act_dim = env_in.action_space.shape[0]
        state_dim = env_in.observation_space.shape[0]
        self.buff_init = [np.zeros(state_dim+act_dim)]
        self.seq_init = [np.zeros(act_dim)]
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


        self.state_mul_const = env_in.observation_space.high
        self.act_mul_const = env_in.action_space.high

        """ State Prediction """
        self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.buff_init[0].size * self.buff_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, state_dim * self.max_seq_len]))
        self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, act_dim * self.max_seq_len]))

        a_seq_split = tf.split(self.a_seq, self.max_seq_len, 1)
        y_seq_split = tf.split(self.y_seq, self.max_seq_len, 1)

        input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)
        self.out_state_raw = self_models.generator_model(input_tmp_seq, state_dim, drop_rate=dropout_rate)

        self.out_state = self.out_state_raw*self.state_mul_const
        self.loss_seq = 0.0
        self.loss_last = 0.0
        out_states = []
        out_states.append(self.out_state_raw)
        self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[0])
        self.loss_last += losses.loss_p(out_states[-1], tf.slice(input_tmp_seq[-1], [0, 0], [-1, state_dim]))
        for i in range(1, self.seq_len):
            state_tmp = tf.slice(self.x_seq[:],
                                   [0, self.buff_init[0].size],
                                   [-1, -1]
                                   )
            state_tmp = tf.concat([state_tmp, out_states[-1]], axis=1)
            input_tmp = tf.concat([state_tmp, a_seq_split[i]], axis=1)

            input_tmp_seq = tf.split(input_tmp, self.buff_len, 1)
            out_state_raw_tmp = self_models.generator_model(input_tmp_seq, state_dim, drop_rate=dropout_rate)
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

    def initialize(self, session):
        self.sess = session
        self.sess.run(tf.global_variables_initializer())

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

    def train_adv(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, self.buff_len)
        n_d = self.n_d
        lGen = 0.0
        lDisc = 0.0
        for i in range(len(X)):
            dg, dx, dLoss, _ = self.sess.run([self.Dg, self.Dx, self.disc_loss, self.train_step_disc],
                                             feed_dict={self.x_seq: X[i],
                                                        self.y_seq: S[i],
                                                        self.a_seq: A[i]
                                                        })  # Update the discriminator
            lDisc += dLoss

        for i in range(len(X)):
            if n_d <= 1 or i % n_d == n_d - 1:
                _, gLoss = self.sess.run([self.train_step_gen, self.gen_loss], feed_dict={self.x_seq: X[i],
                                                                                          self.y_seq: S[i],
                                                                                          self.a_seq: A[i]
                                                                                          })  # Update the generator
                lGen += gLoss
        return n_d * lGen / len(X), lDisc / len(X)

    def get_loss(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, self.buff_len)
        lC = 0.0
        lGen = 0.0
        lDisc = 0.0
        for i in range(len(X)):
            lDisc += self.sess.run(self.disc_loss, feed_dict={self.x_seq: X[i],
                                                              self.y_seq: S[i],
                                                              self.a_seq: A[i]
                                                             })
            lGen += self.sess.run(self.g_loss, feed_dict={self.x_seq: X[i],
                                                            self.y_seq: S[i],
                                                            self.a_seq: A[i]
                                                            })

            osr, ls, os = self.sess.run([self.out_state_raw, self.loss_state, self.out_state], feed_dict={self.x_seq: X[i],
                                                                                          self.y_seq: S[i],
                                                                                          self.a_seq: A[i]
                                                                                          })
            lC += ls
        return lGen / len(X), lDisc / len(X), lC / len(X)

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
            x = np.array([np.concatenate(buff).flatten()])
        else:
            x = np.array([np.concatenate(self.buffer).flatten()])
        new_obs = self.sess.run(self.out_state, feed_dict={self.x_seq: x}).flatten()
        return new_obs
