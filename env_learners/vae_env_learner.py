import time
from collections import deque

import numpy as np
import tensorflow as tf

from env_learners.env_learner import EnvLearner
from misc import losses, logger


def encoder_mean(x, latent_size, drop_rate=0.5):
    x_seq = []
    for x_tmp in x:
        x_tmp = tf.layers.batch_normalization(x_tmp)
        x_seq.append(x_tmp)

    # x_seq = tf.split(x, buff_len, 1)
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(1024)
    rnn_cell = tf.contrib.rnn.GRUCell(1024, name='rnn_mean')
    outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
    x = outputs[-1]

    x_new = []
    # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    for x in outputs:
        x = tf.expand_dims(x, -1)
        x_new.append(x)
    x = tf.concat(x_new, axis=2)
    x = tf.layers.conv1d(x, 64, 3)
    x = tf.layers.conv1d(x, 32, 1)
    x = tf.layers.flatten(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, latent_size)
    return x

def encoder_var(x, latent_size, drop_rate=0.5):
    x_seq = []
    for x_tmp in x:
        x_tmp = tf.layers.batch_normalization(x_tmp)
        x_seq.append(x_tmp)

    # x_seq = tf.split(x, buff_len, 1)
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(1024)
    rnn_cell = tf.contrib.rnn.GRUCell(1024, name='rnn_var')
    outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
    x = outputs[-1]

    x_new = []
    # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    for x in outputs:
        x = tf.expand_dims(x, -1)
        x_new.append(x)
    x = tf.concat(x_new, axis=2)
    x = tf.layers.conv1d(x, 64, 3)
    x = tf.layers.conv1d(x, 32, 1)
    x = tf.layers.flatten(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, latent_size)
    return x

def decode_this(x, out_dim, drop_rate=0.5):
    # # x_seq = tf.split(x, buff_len, 1)
    # x_seq = []
    # for x_tmp in x:
    #     x_tmp = tf.layers.batch_normalization(x_tmp)
    #     x_seq.append(x_tmp)
    #
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
    # # rnn_cell = tf.contrib.rnn.GRUCell(512)
    # outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # x = outputs[-1]
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)

    # x_new = []
    # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    # for x in x_seq:
    #     x = tf.expand_dims(x, -1)
    #     x_new.append(x)
    # x = tf.concat(x_new, axis=2)
    # x = tf.layers.conv1d(x, 64, 3)
    # x = tf.layers.conv1d(x, 32, 1)
    # x = tf.layers.flatten(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)
    return x

def decode_next(x, out_dim, drop_rate=0.5):
    # x_seq = tf.split(x, buff_len, 1)
    # x_seq = []
    # for x_tmp in x:
    #     x_tmp = tf.layers.batch_normalization(x_tmp)
    #     x_seq.append(x_tmp)
    #
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
    # # rnn_cell = tf.contrib.rnn.GRUCell(512)
    # outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # x = outputs[-1]
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)

    # x_new = []
    # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    # for x in x_seq:
    #     x = tf.expand_dims(x, -1)
    #     x_new.append(x)
    # x = tf.concat(x_new, axis=2)
    # x = tf.layers.conv1d(x, 64, 3)
    # x = tf.layers.conv1d(x, 32, 1)
    # x = tf.layers.flatten(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)
    return x

def KLD_loss(latent_var, latent_mean):
    return -0.5 * tf.reduce_mean(1.0 + latent_var - tf.pow(latent_mean, 2) - tf.exp(latent_var))



## This needs some testing

class VAEEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # Initialization

        self.latent_size = 1024

        self.buff_len = 10
        self.last_r = np.array([0.0]).flatten()
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        dropout_rate = 0.5
        lr = 1e-5
        logger.info('General Stats: ')
        logger.info('Drop Rate: ' + str(dropout_rate))
        logger.info('Buffer Len: ' + str(self.buff_len))
        logger.info('vae_model:')
        logger.info('Learning Rate: ' + str(lr))

        """ State Prediction """
        self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.buff_init[0].size * self.buff_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim * self.max_seq_len]))
        # self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim * self.max_seq_len]))


        input_tmp_seq = tf.split(self.x_seq, self.buff_len, 1)


        self.latent_mean = encoder_mean(input_tmp_seq, self.latent_size)
        self.latent_var = encoder_var(input_tmp_seq, self.latent_size)


        train_latent = tf.random_uniform(shape=([self.latent_size]))
        train_latent = train_latent*tf.exp(0.5*self.latent_var)+self.latent_mean
        test_latent = self.latent_mean


        self.decoded_next_train = decode_next(train_latent, self.state_dim)
        self.decoded_next = decode_next(test_latent, self.state_dim)

        # Maybe scale this up for increasing sequences as well?
        # unsure if that's been done with VAEs so leaving for now

        self.loss_next = losses.loss_p(self.decoded_next_train, self.y_seq) + KLD_loss(self.latent_var, self.latent_mean)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_next)


    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data)
        Loss = 0.0
        for i in range(len(X)):
            loss, _ = self.sess.run([self.loss_next, self.train_step],
                                             feed_dict={self.x_seq: X[i],
                                                        self.y_seq: S[i]
                                                        })
            Loss += loss

        return Loss / len(X), 0,0


    def train(self, train, total_steps, valid=None, logger=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
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
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1
                print('Sequence Length: ' + str(self.seq_len))

            if i % log_interval == 0 and i > 0 and logger is not None and valid is not None:
                (loss, _,_) = self.get_loss(valid)
                logger.info('Epoch: ' + str(i) + '/' + str(total_steps))
                logger.info('Train Loss: ' + str(loss))
                logger.info()
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    logger.info("Model saved in path: %s" % save_path)
            start = time.time()
            (loss, _, _) = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            if i % log_interval != 0 and i > 0 and logger is not None:
                logger.info('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                logger.info('Valid Loss: ' + str(loss))
                logger.info()
        if logger is not None and valid is not None:
                (loss, _,_) = self.get_loss(valid)
                logger.info('Final Epoch')
                logger.info('Valid Loss:' + str(loss))
                logger.info()
        if saver is not None and save_str is not None:
            save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
            logger.info("Final Model saved in path: %s" % save_path)

    def get_loss(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, self.buff_len)
        Loss = 0.0
        for i in range(len(X)):
            loss = self.sess.run(self.loss_next,
                                             feed_dict={self.x_seq: X[i],
                                                        self.y_seq: S[i]
                                                        })
            self.training = False
            Loss += loss
        return Loss / len(X), 0,0

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
        new_obs = self.sess.run(self.decoded_next, feed_dict={self.x_seq: x}).flatten()
        return new_obs