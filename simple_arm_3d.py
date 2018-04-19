import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import math

class SimpleArm(gym.Env):
    def __init__(self, train=True):
        self.r = np.array([1, 1])
        self.max_iter = 100
        self.train = train

        self.action_space = spaces.Box(-math.pi/16.0, math.pi/16.0, shape=(2*self.r.size,))
#        self.observation_space = spaces.Box(
#                                            low=np.array([-np.sum(self.r), -np.sum(self.r)]),
#                                            high=np.array([np.sum(self.r), np.sum(self.r)])
#                                           )
        # joint angles
        h = [math.pi]*2*self.r.size

        # elbows
        # sum = 0.0
        # for i in range(self.r.size-1):
        #     sum += np.sum(self.r[:i+1])
        #     h.extend([sum]*3)

        # end effectors
        h.extend([np.sum(self.r)]*3)
        l = [-i for i in h]
        if not self.train:
            l.append(0)
            h.append(2*np.sum(self.r))
        self.observation_space = spaces.Box(
                                            low=np.array(l),
                                            high=np.array(h)
                                           )

        self.observation = self.reset()

    def __get_obs__(self):
        elbows = []
        last_ver = 0.0
        last_hor = 0.0
        # elbow = np.zeros(3)
        # for j in range(self.r.size-1):
        #     elbow[0] += float(self.r[j]*math.cos(last_hor+self.x[2*j])*math.sin(math.pi/2-last_ver-self.x[2*j+1]))
        #     elbow[1] += float(self.r[j]*math.sin(last_hor+self.x[2*j])*math.sin(math.pi/2-last_ver-self.x[2*j+1]))
        #     elbow[2] += float(self.r[j]*math.cos(math.pi/2-last_ver-self.x[2*j+1]))
        #     elbows.append(elbow)
        # elbows = np.concatenate(elbows)

        if not self.train:
            return np.concatenate([self.x, self.y, np.array([self.d])], axis=0)
        else:
            return np.concatenate([self.x, self.y], axis=0)
        #return self.y

    def __get_pos__(self, x):
        y = np.zeros(3)
        last_ver = 0.0
        last_hor = 0.0
        for j in range(self.r.size):
            y[0] += float(self.r[j]*math.cos(last_hor+x[2*j])*math.sin(math.pi/2-last_ver-x[2*j+1]))
            y[1] += float(self.r[j]*math.sin(last_hor+x[2*j])*math.sin(math.pi/2-last_ver-x[2*j+1]))
            y[2] += float(self.r[j]*math.cos(math.pi/2-last_ver-x[2*j+1]))
            last_hor += x[2*j]
            last_ver += x[2*j+1]
        return y

    def __clip_x__(self):
        for i in range(self.x.size):
            if np.abs(self.x[i]) > 7*math.pi/8:
                self.x[i] = np.sign(self.x[i])*(7*math.pi/8)
        return self.x

    def reset(self):
        np.random.seed()
        self.x = np.random.uniform(-math.pi, math.pi, 2*self.r.size)
        self.y = self.__get_pos__(self.x)
        np.random.seed()
        tmp = np.random.uniform(-math.pi, math.pi, 2*self.r.size)
        self.target = self.__get_pos__(tmp)
        # print(self.target)
        self.iteration = 0
        self.d = np.linalg.norm(self.y - self.target)
        self.state = self.__get_obs__()
        return self.state

    def step(self, action, save=True):
        if save:
            self.x += action

            # Comment out for baseline
            # self.x += np.random.normal(0, math.pi/90.0, size=self.x.size)
            # self.__clip_x__()

            self.y = self.__get_pos__(self.x)
            self.iteration += 1
            self.done = (self.iteration >= self.max_iter)
            new_d = float(np.linalg.norm(self.y - self.target))
            if self.iteration == 1:
               self.reward = -new_d
            else:
                self.reward = self.d-new_d
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
            if not self.train:
                return np.concatenate([x_new, y_new, np.array([self.d])], axis=0), -d_new, False, {}
            else:
                return np.concatenate([x_new, y_new], axis=0), -d_new, False, {}
