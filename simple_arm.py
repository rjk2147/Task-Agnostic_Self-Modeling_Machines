import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import math

class SimpleArm(gym.Env):
    def __init__(self, load=False):
        self.r = np.array([1, 1])
        self.max_iter = 100
        self.load = load
        if self.load:
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            try:
                loader = tf.train.import_meta_graph('models/2-joint-arm.meta')
                loader.restore(self.sess, tf.train.latest_checkpoint('models/'))
            except Exception as e:
                print(e)
                self.load = False

        self.action_space = spaces.Box(-math.pi/16.0, math.pi/16.0, shape=(2,))
#        self.observation_space = spaces.Box(
#                                            low=np.array([-np.sum(self.r), -np.sum(self.r)]),
#                                            high=np.array([np.sum(self.r), np.sum(self.r)])
#                                           )

        self.observation_space = spaces.Box(
                                            low=np.array([-math.pi, -math.pi, -2, -2]),
                                            high=np.array([math.pi, math.pi, 2, 2])
                                           )

        self.observation = self.reset()
    def __get_obs__(self):
        return np.concatenate([self.x, self.y], axis=0)
        #return self.y

    def __get_pos__(self, x, load=False):
        y = np.zeros(2)
        last_deg = 0.0
        for j in range(self.r.size):
            y[0] += float(self.r[j]*math.cos(last_deg+x[j]))
            y[1] += float(self.r[j]*math.sin(last_deg+x[j]))
            last_deg += x[j]
        return y

    def __clip_x__(self):
        # for i in range(self.x.size):
        #     if np.abs(self.x[i]) > 7*math.pi/8:
        #         self.x[i] = np.sign(self.x[i])*(7*math.pi/8)
        return self.x

    def reset(self):
        np.random.seed()
        self.x = np.random.uniform(-math.pi, math.pi, 2)
        # self.x = np.array([0.0,0.0])
        self.y = self.__get_pos__(self.x, load=self.load)
        np.random.seed()
        tmp = np.random.uniform(-math.pi, math.pi, 2)
        self.target = self.__get_pos__(tmp, load=self.load)
        # print(self.target)
        self.iteration = 0
        self.d = np.linalg.norm(self.y - self.target)
        self.state = self.__get_obs__()
        return self.state

    def step(self, action, save=True):
        if save:
            self.x += action
            self.x += np.random.normal(0, math.pi/90.0, size=self.x.size)
            self.__clip_x__()
            self.y = self.__get_pos__(self.x)
            self.iteration += 1
            self.done = (self.iteration >= self.max_iter)
            new_d = float(np.linalg.norm(self.y - self.target))
            #if self.iteration == 1:
            #    self.reward = -new_d
            #else:
            self.reward = -new_d
            # self.reward = np.sign(self.d-new_d)
            if new_d == 0:
                print('Success')
            #    self.reward = 1
                self.done = True
            self.d = new_d
            return self.__get_obs__(), self.reward, self.done, {}
        else:
            x_new = self.x+action
            for i in range(self.x.size):
                while x_new[i] > math.pi: x_new[i] -= 2*math.pi
                while x_new[i] < -math.pi: x_new[i] += 2*math.pi
            y_new = self.__get_pos__(x_new)
            d_new = float(np.linalg.norm(self.y - self.target))
            return np.concatenate([x_new, y_new], axis=0), -d_new, False, {}
