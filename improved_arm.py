import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import math
from math import sin,cos

def translate(p, obj):
    t = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [p[0], p[1], p[2], 1]]
                )
    return np.dot(t, obj)

def rotate(theta, obj):
    r = [math.radians(theta[0]), math.radians(theta[1]), math.radians(theta[2])]
    Rx = np.array([[1, 0, 0, 0],
                  [0, cos(r[0]), -sin(r[0]), 0],
                  [0, sin(r[0]), cos(r[0]), 0],
                  [0, 0, 0, 1]]
                )
    Ry = np.array([[cos(r[1]), 0, sin(r[1]), 0],
                  [0, 1, 0, 0],
                  [-sin(r[1]), 0, cos(r[1]), 0],
                  [0, 0, 0, 1]]
                )
    Rz = np.array([[cos(r[2]), -sin(r[2]), 0, 0],
                  [sin(r[2]), cos(r[2]), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
                )
    R = np.dot(Rx, Ry)
    R = np.dot(R, Rz)
    return np.dot(R, obj)
def simulate_arm(l, thetas):
    p = [0,0,0]
    arm_r = l[0]/2
    obj = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
                    )
    positions = []
    # Joint 1
    obj = translate(p, translate([0,0,5.5*arm_r/7.0], obj))
    positions.append(obj[3][:3])
    #print(obj[3][:3])
    i = 0
    while i+2 < len(thetas):
        # Joint i+2
        arm_r /= 1.8
        obj = translate([0,0,l[int(i/2)]+5.5*arm_r/7.0], rotate([thetas[i],0,thetas[i+1]], obj))
        positions.append(obj[3][:3])
        # print(obj[3][:3])
        i += 2
    # End Effector
    obj = translate([0,0,l[2]], rotate([thetas[4],0, thetas[5]], obj))
    positions.append(obj[3][:3])
    # print(obj[3][:3])
    return positions

class ImprovedArm(gym.Env):
    def __init__(self, train=True):
        self.r = np.array([1, 1, 1])
        self.max_iter = 100
        self.train = train

        self.action_space = spaces.Box(-math.pi/16.0, math.pi/16.0, shape=(2*self.r.size,))
        h = [math.pi]*2*self.r.size
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
        if not self.train:
            return np.concatenate([self.x, self.y, np.array([self.d])], axis=0)
        else:
            return np.concatenate([self.x, self.y], axis=0)

    def __get_pos__(self, x):
        positions = simulate_arm(self.r, np.degrees(x))
        return positions[-1]

    # Prevents self-collision
    def __clip_x__(self):
        for i in range(self.x.size):
            if i%2 == 0 and np.abs(self.x[i]) > math.pi/2:
                self.x[i] = np.sign(self.x[i])*math.pi/2
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

            # Enables Noisy
            # self.x += np.random.normal(0, math.pi/90.0, size=self.x.size)
            # Disables self-collision
            self.__clip_x__()

            self.y = self.__get_pos__(self.x)
            self.iteration += 1
            self.done = (self.iteration >= self.max_iter)
            new_d = float(np.linalg.norm(self.y - self.target))
            if self.iteration == 1:
               self.reward = -new_d
            else:
                self.reward = self.d-new_d
            if new_d == 0:
                print('Success')
                self.done = True
            self.d = new_d
            return self.__get_obs__(), self.reward, self.done, {}
        else:
            x_new = self.x+action
            self.__clip_x__()
            for i in range(self.x.size):
                while x_new[i] > math.pi: x_new[i] -= 2*math.pi
                while x_new[i] < -math.pi: x_new[i] += 2*math.pi
            y_new = self.__get_pos__(x_new)
            d_new = float(np.linalg.norm(self.y - self.target))
            if not self.train:
                return np.concatenate([x_new, y_new, np.array([self.d])], axis=0), -d_new, False, {}
            else:
                return np.concatenate([x_new, y_new], axis=0), -d_new, False, {}

l = np.array([1, 1, 1])
pos = simulate_arm(l, [51.86890375930864,-162.20510810582564, -44.4826441264804,-125.42934219996172, 98.85694621965835,3.0689309732702528])
print(pos[-1])