import time
from collections import deque

import numpy as np
import tensorflow as tf

from misc import models
from env_learners.env_learner import EnvLearner
from misc import losses

class GANEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # Initialization
        self.buff_len = 10
        self.seq_len = 1
        self.max_seq_len = 5
        self.last_r = np.array([0.0]).flatten()
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        dropout_rate = 0.5
        self.lr_disc = 1e-5
        self.lr_gen = 1e-5
        print('General Stats: ')
        print('Drop Rate: ' + str(dropout_rate))
        print('Buffer Len: ' + str(self.buff_len))
        print('Start Sequence Len: ' + str(self.seq_len))
        print('End Sequence Len: ' + str(self.max_seq_len))
        print('gan_model:')
        print('Learning Rate: ' + str(self.lr_disc))
        print('Learning Rate: ' + str(self.lr_gen))

        """ State Prediction """
        self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.buff_init[0].size * self.buff_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim * self.max_seq_len]))
        self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim * self.max_seq_len]))

        a_seq_split = tf.split(self.a_seq, self.max_seq_len, 1)
        y_seq_split = tf.split(self.y_seq, self.max_seq_len, 1)

        input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)
        self.out_state_raw = models.generator_model(input_tmp_seq, self.state_dim, drop_rate=dropout_rate)

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
            out_state_raw_tmp = models.generator_model(input_tmp_seq, self.state_dim, drop_rate=dropout_rate)
            out_states.append(out_state_raw_tmp)
            self.loss_seq += losses.loss_p(out_states[-1], y_seq_split[i])
            self.loss_last += losses.loss_p(out_states[-1], out_states[-2])

        self.out_state_seq = tf.concat(out_states, axis=1)

        self.loss_state = self.loss_seq

        self.train_step_state = tf.train.AdamOptimizer(self.lr_gen).minimize(self.loss_state)

        """ GAN Stuff """
        x_seq = []
        g_seq = []
        out_seq_split = tf.split(self.out_state_seq, self.seq_len, 1)
        for i in range(self.seq_len):
            x_seq.append(tf.concat([y_seq_split[i], a_seq_split[i]], axis=1))
            g_seq.append(tf.concat([out_seq_split[i], a_seq_split[i]], axis=1))

        x_in = x_seq
        g_in = g_seq
        self.Dx = models.discriminator_model(x_in, drop_rate=dropout_rate)
        self.Dg = models.discriminator_model(g_in, drop_rate=dropout_rate)
        var_d = tf.trainable_variables('discriminator')
        var_g = tf.trainable_variables('generator')
        self.g_lambda = 1.0
        self.p_lambda = 10.0
        self.t_lambda = 0.0

        """ Vanilla GAN """
        # self.n_d = 1
        # self.disc_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1-self.Dg))
        # self.g_loss = -tf.reduce_mean(tf.log(self.Dg))
        # self.gen_loss =  g_lambda*self.g_loss + p_lambda * self.loss_seq
        # self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
        # self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

        """ WGAN-GP """
        self.n_d = 5
        self.epsilon = 0.01
        self.gp_lambda = 10

        self.disc_loss = tf.reduce_mean(self.Dg) - tf.reduce_mean(self.Dx)
        self.g_loss = -tf.reduce_mean(self.Dg)
        self.gen_loss =  self.g_lambda*self.g_loss + \
                         self.p_lambda * self.loss_seq + \
                         self.t_lambda * self.loss_last
        x_hat = self.epsilon*self.Dx + (1-self.epsilon)*self.Dg
        grad_list = tf.gradients(x_hat, var_d)[2:]
        GP = 0.0
        for layer in grad_list:
            GP += self.gp_lambda * (tf.sqrt(tf.reduce_sum(tf.square(layer))) - 1) ** 2
        self.disc_loss += GP
        self.train_step_disc = tf.train.AdamOptimizer(self.lr_disc, beta1=0, beta2=0.9).minimize(self.disc_loss, var_list=var_d)
        self.train_step_gen = tf.train.AdamOptimizer(self.lr_gen, beta1=0, beta2=0.9).minimize(self.gen_loss, var_list=var_g)

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

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
            out_state_raw_tmp = models.generator_model(input_tmp_seq, self.state_dim, drop_rate=0.5)
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

        """ Vanilla GAN """
        # self.n_d = 1
        # self.disc_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1-self.Dg))
        # self.g_loss = -tf.reduce_mean(tf.log(self.Dg))
        # self.gen_loss =  g_lambda*self.g_loss + p_lambda * self.loss_seq
        # self.train_step_disc = tf.train.AdamOptimizer(lr_disc).minimize(self.disc_loss, var_list=var_d)
        # self.train_step_gen = tf.train.AdamOptimizer(lr_gen).minimize(self.gen_loss, var_list=var_g)

        """ WGAN-GP """
        self.disc_loss = tf.reduce_mean(self.Dg) - tf.reduce_mean(self.Dx)
        self.g_loss = -tf.reduce_mean(self.Dg)
        self.gen_loss = self.g_lambda * self.g_loss + \
                        self.p_lambda * self.loss_seq + \
                        self.t_lambda * self.loss_last
        x_hat = self.epsilon * self.Dx + (1 - self.epsilon) * self.Dg
        grad_list = tf.gradients(x_hat, var_d)[2:]
        GP = 0.0
        for layer in grad_list:
            GP += self.gp_lambda * (tf.sqrt(tf.reduce_sum(tf.square(layer))) - 1) ** 2
        self.disc_loss += GP
        # self.train_step_disc = tf.train.AdamOptimizer(self.lr_disc).minimize(self.disc_loss, var_list=var_d)
        # self.train_step_gen = tf.train.AdamOptimizer(self.lr_gen).minimize(self.gen_loss, var_list=var_g)

    def train_epoch(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=32)
        n_d = self.n_d
        lGen = 0.0
        lDisc = 0.0
        lC = 0.0
        for i in range(len(X)):
            test = self.sess.run([self.y_seq], feed_dict={self.y_seq: S[i]})
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

    def train(self, train, total_steps, valid=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        min_loss = 10000000000
        stop_count = 0

        seq_i = 0
        seq_idx = [1] * (self.max_seq_len - self.seq_len + 1)
        for j in range(1, self.max_seq_len - self.seq_len + 1):
            seq_tmp = self.max_seq_len - j
            seq_idx[j] = (seq_tmp + 1) * seq_idx[j - 1] / seq_tmp
        seq_idx.reverse()
        mul_const = total_steps / sum(seq_idx)
        for j in range(len(seq_idx)):
            seq_idx[j] = round(mul_const * seq_idx[j])
            if j > 0:
                seq_idx[j] += seq_idx[j - 1]
        for i in range(total_steps):
            # if i > 0 and i % (
            #     total_steps / self.max_seq_len) == 0 and self.seq_len < self.max_seq_len:
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1
                self.init_gan_losses()
                print('Sequence Length: ' + str(self.seq_len))

            start = time.time()
            tlGen, tlDisc, tlC = self.train_epoch(train)
            duration = time.time() - start
            if i % log_interval == 0 and i > 0:
                if valid is not None:
                    (vGen, vDisc, vC) = self.get_loss(valid)
                    print('Epoch: ' + str(i) + '/' + str(total_steps))
                    print('Valid Loss')
                    print('Gen:  ' + str(vGen))
                    print('Disc: ' + str(vDisc))
                    print('Close: ' + str(vC))
                    print('')
                else:
                    print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                    print('Train Loss')
                    print('Gen:  ' + str(tlGen))
                    print('Disc: ' + str(tlDisc))
                    print('Close: ' + str(tlC))
                    print('')
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    print("Model saved in path: %s" % save_path)
            if tlGen < min_loss:
                min_loss = tlGen
                stop_count = 0
            else:
                stop_count += 1
            if stop_count > early_stopping and early_stopping > 0:
                break
            if i % log_interval != 0 and i > 0:
                print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                print('Train Loss')
                print('Gen:  ' + str(tlGen))
                print('Disc: ' + str(tlDisc))
                print('Close: ' + str(tlC))
                print('')
        if valid is not None:
            (vGen, vDisc, vC) = self.get_loss(valid)
            print('Final Epoch: ')
            print('Valid Loss')
            print('Gen:  ' + str(vGen))
            print('Disc: ' + str(vDisc))
            print('Close: ' + str(vC))
            print('')
        if saver is not None and save_str is not None:
            save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
            print("Final Model saved in path: %s" % save_path)

    def get_loss(self, data):
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
        import copy
        obs = obs_in/self.state_mul_const
        action = action_in/self.act_mul_const
        if save:
            if episode_step == 0:
                self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
            self.buffer.append(np.array([np.concatenate([obs, action])]).flatten())
        else:
            if buff is None:
                buff = copy.copy(self.buffer)
            if episode_step == 0:
                buff = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
            buff.append(np.array([np.concatenate([obs, action])]).flatten())

        if buff is not None:
            x = np.array([np.concatenate(buff).flatten()])
        else:
            x = np.array([np.concatenate(self.buffer).flatten()])
        new_obs = self.sess.run(self.out_state, feed_dict={self.x_seq: x}).flatten()
        return new_obs