import gym
from gym import spaces
import numpy as np
import math
from spring_mass import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SpringMass(gym.Env):
    def __init__(self):
        self.reset()
    def reset(self):
        pass
    def step(self, action):
        pass

class SpringMassCrawler(gym.Env):
    def __init__(self, render=False):
        self.dt = 0.0001
        self.env_dt = 0.01
        self.fric = 0.5
        self.episode_time = 1 # len of 1 episode in seconds
        self.Fext = np.array([0, 0, -9.8])
        self.x = np.zeros(2)
        self.pos = np.zeros(9)
        self.Ys = []
        self.action_space = spaces.Box(-0.01, 0.01, shape=(2,))
#        self.observation_space = spaces.Box(
#                                            low=np.array([-np.sum(self.r), -np.sum(self.r)]),
#                                            high=np.array([np.sum(self.r), np.sum(self.r)])
#                                           )

        self.observation_space = spaces.Box(
                                            low=np.array([-0.1]*9),
                                            high=np.array([0.1]*9)
                                           )
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # plt.figure()
        # plt.subplot(111, aspect='equal')
        # # plt.show()
        # plt.ion()

    def reset(self):
        self.masses, self.springs = make_cube()
        hip = self.masses[0]
        knee = Mass(p=[hip.p[0], hip.p[1]-0.2*math.cos(math.pi/4), hip.p[2]+0.2*math.sin(math.pi/4)])
        foot = Mass(p=[0,knee.p[1], 0])
        self.masses.append(knee)
        self.masses.append(foot)

        self.springs.append(self.masses[2].connect(knee))
        self.springs.append(self.masses[3].connect(knee))
        self.springs.append(hip.connect(knee))
        self.springs.append(knee.connect(foot))
        # self.springs.append(knee.connect(self.masses[4]))
        # self.springs.append(foot.connect(self.masses[4]))
        # self.springs[-2].min=1
        # self.springs[-1].min=1
        # self.springs.append(knee.connect(self.masses[2]))
        # self.springs.append(knee.connect(self.masses[3]))
        # self.springs.append(foot.connect(self.masses[2]))
        # self.springs.append(foot.connect(self.masses[3]))

        self.muscles = []
        self.muscles.append(knee.connect(self.masses[1]))
        self.muscles.append(foot.connect(hip))
        # self.muscles[0].k = 1000
        # self.muscles[1].k = 500
        self.springs.extend(self.muscles)

        cube_avg = np.zeros(3)
        for i in range(len(self.masses)-2):
            cube_avg += self.masses[0].p
        cube_avg /= float(len(self.masses)-2)
        self.pos = np.concatenate([cube_avg, self.masses[-2].p, self.masses[-1].p])

        self.last_center = 0.0
        self.i = 0
        state = np.zeros_like(self.pos)
        return state


    def __step__(self, action):
        self.muscles[0].l0 = max(min(self.muscles[0].l0+action[0], 0.2+0.2),0)
        self.muscles[0].update()
        self.muscles[1].l0 = max(min(self.muscles[1].l0+action[1], 0.2+0.3414213562373095),0)
        self.muscles[1].update()
        update_all(masses=self.masses, springs=self.springs, dt=self.dt, Fext=self.Fext, u=self.fric)
        cube_avg = np.zeros(3)
        for i in range(len(self.masses)-2):
            cube_avg += self.masses[i].p
        cube_avg /= float(len(self.masses)-2)
        new_pos = np.concatenate([cube_avg, self.masses[-2].p, self.masses[-1].p])
        state = new_pos-self.pos
        self.pos = new_pos

        rew = cube_avg[1]-self.last_center
        done = self.i > int(self.episode_time/self.dt)
        # done = False
        self.last_center = cube_avg[1]
        self.i += 1
        return state, rew, done, {}

    def step(self, action):
        time_slice = self.env_dt/self.dt
        state = np.zeros(9)
        rew = 0
        is_done = False
        for i in range(int(time_slice)):
            state_slice, rew_slice, done, _ = self.__step__(action/time_slice)
            state += state_slice
            rew += rew_slice
            if is_done or done: is_done=True
            # if done:
            #     return state, rew, done, {}
        return state, rew, is_done, {}

    def render_(self, mode='human'):
        last_deg = 0.0
        plt.clf()
        for spring in self.springs:
            x = [spring.ends[0].p[1], spring.ends[1].p[1]]
            y = [spring.ends[0].p[2], spring.ends[1].p[2]]
            if spring not in self.muscles:
                plt.plot(x, y, 'ro', linestyle='-', color='black')
        for spring in self.muscles:
            x = [spring.ends[0].p[1], spring.ends[1].p[1]]
            y = [spring.ends[0].p[2], spring.ends[1].p[2]]
            plt.plot(x, y, 'ro', linestyle='-', color='red')

        # minX = -1
        # maxX = 4
        minX = -0.5
        maxX = 0.5
        plt.axis([minX, maxX, 0.0, 3*(maxX-minX)/4])
        # plt.pause(0.0001)
        plt.draw()
        # plt.savefig(str(self.i)+'.png')

    def render(self, mode='human'):
        # plt.close('all')
        # if self.fig is None:
        #     self.fig = plt.figure()
        # else:
        #     plt.cla()
        # self.fig = plt.figure()
        plt.cla()

        last_deg = 0.0
        for spring in self.springs:
            x = [spring.ends[0].p[0], spring.ends[1].p[0]]
            y = [spring.ends[0].p[1], spring.ends[1].p[1]]
            z = [spring.ends[0].p[2], spring.ends[1].p[2]]
            if spring not in self.muscles:
                self.ax.plot(x, y, z, marker=None, linestyle='-', color='black', alpha=1.0, linewidth=1, zorder = 20)
                self.ax.scatter(x, y, z, marker='o', linestyle='-', color='black', alpha=1.0, linewidth=1, zorder= 30)
                # plt.plot(x, y, 'ro', linestyle='-', color='black')
        for spring in self.muscles:
            x = [spring.ends[0].p[0], spring.ends[1].p[0]]
            y = [spring.ends[0].p[1], spring.ends[1].p[1]]
            z = [spring.ends[0].p[2], spring.ends[1].p[2]]
            self.ax.plot(x, y, z, marker=None, linestyle='-', color='red', alpha=1.0, linewidth=1, zorder = 0)
            self.ax.scatter(x, y, z, marker='o', linestyle='-', color='red', alpha=1.0, linewidth=1, zorder= 10)
            # plt.plot(x, y, 'ro', linestyle='-', color='red')

        minX = -0.5
        maxX = 0.5
        self.ax.set_xlim(minX, maxX)
        self.ax.set_ylim(minX, maxX)
        self.ax.set_zlim(0, maxX)
        plt.draw()
        plt.pause(0.5)
        import time
        # time.sleep(100)
        # plt.savefig(str(self.i)+'.png')