import numpy as np
import tensorflow as tf

from env_learners.env_learner import EnvLearner


## Incomplete but not a priority to complete

class RandEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        self.buff_init = [np.zeros(self.state_dim+self.act_dim)]

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train(self, train, total_steps, valid=None, logger=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        epoch_min_loss = 100000000
        G, yS, yR, yD, X, S, A = self.__prep_data__(train, 10000)
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

    def get_loss(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, 0)
        loss = 0
        for i in range(len(X)):
            batch_loss = self.sess.run([self.loss_state], feed_dict={self.x_seq: X[i],
                                                                        self.y_seq: S[i],
                                                                        self.a_seq: A[i]
                                                                        })[0]
            loss += batch_loss
        return 0,0,loss

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        pass