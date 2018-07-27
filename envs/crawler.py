import gym
from gym import spaces
import numpy as np
import math

class SimpleCrawler(gym.Env):
    def __init__(self, load=False):

        # coef of friction = 0.5
        self.grav = 9.8
        self.u = 0.5
        self.T = 10
        self.dt = 0.1
        self.m = [3,1,1]
        self.pos = np.zeros(2)
        self.last_pos = np.zeros(2)
        # torque on motors = 10 Nm
        # Mass of center  = 2kg
        # Mass of elbow = 1kg
        # Mass of end = 1kg
        # length of arms 1m each

        self.r = np.array([1, 1])
        self.F = 0.0
        self.max_iter = 100

        self.action_space = spaces.Box(-math.pi/16.0, math.pi/16.0, shape=(2,))
#        self.observation_space = spaces.Box(
#                                            low=np.array([-np.sum(self.r), -np.sum(self.r)]),
#                                            high=np.array([np.sum(self.r), np.sum(self.r)])
#                                           )

        self.observation_space = spaces.Box(
                                            low=np.array([-math.pi, -math.pi, -1, -1, -2, -2, -1, -1]),
                                            high=np.array([math.pi, math.pi, 1, 1, 2, 2, 1, 1])
                                           )

        self.observation = self.reset()
    def __get_obs__(self):
        y = np.concatenate(self.Ys, axis=0)
        return np.concatenate([self.x, y[2:], (self.last_pos-self.pos)], axis=0)
        #return self.y

    def __get_pos__(self, x, load=False):
        self.Ys = []
        y = np.zeros(2)
        self.Ys.append(np.zeros(2))
        last_deg = 0.0
        for j in range(self.r.size):
            y[0] += float(self.r[j]*math.cos(last_deg+x[j]))
            y[1] += float(self.r[j]*math.sin(last_deg+x[j]))
            self.Ys.append(np.zeros(2)+y)
            if self.Ys[-1][1] < 0:
                new_deg = 0.0
                while self.Ys[-1][1] < 0:
                    self.Ys[-1][1] = self.Ys[-2][1] + float(self.r[j] * math.sin(last_deg + self.x[j] + new_deg))
                    new_deg += 0.01
                # print(new_deg)
                # y = np.zeros(2)+self.Ys[-1]

                dTheta = self.last_x[j] - self.x[j]
                ## Not sure if this is correct....
                # F = env.T/math.sin(new_deg)
                F = self.T

                hor = abs(F * math.cos(last_deg + self.x[j] + new_deg))
                N = abs(F * math.sin(last_deg + self.x[j] + new_deg))
                # N += env.grav*env.m[j]
                fric = N * self.u
                if abs(fric) > abs(hor):
                    hor = 0
                else:
                    hor = (hor - fric) * np.sign(dTheta)

                acc = hor / sum(self.m)
                v = acc * self.dt
                p = v * self.dt
                # print(p)
                self.pos[0] += p

            last_deg += x[j]
        return y

    def __calc_ext_force(self):
        pass

    def __clip_x__(self):
        for i in range(self.x.size):
            if np.abs(self.x[i]) > 7*math.pi/8:
                self.x[i] = np.sign(self.x[i])*(7*math.pi/8)
        return self.x

    def reset(self):
        np.random.seed()
        self.x = np.random.uniform(0, math.pi, 2)
        self.last_pos = np.zeros(2)
        self.pos = np.zeros(2)
        self.last_x = np.zeros(2)+self.x
        # self.last_x = np.array([3*math.pi/4,math.pi/4])
        # self.x = np.array([3*math.pi/4,math.pi/4])
        self.y = self.__get_pos__(self.x)
        np.random.seed()
        tmp = np.random.uniform(-math.pi, math.pi, 2)
        self.target = self.__get_pos__(tmp)
        # print(self.target)
        self.iteration = 0
        self.d = np.linalg.norm(self.y - self.target)
        self.state = self.__get_obs__()
        return self.state

    def step(self, action, save=True):
        if save:
            self.last_pos = np.zeros(2)+self.pos
            self.last_x = np.zeros(2)+self.x
            self.x += action
            # self.x += np.random.normal(0, math.pi/90.0, size=self.x.size)
            # self.__clip_x__()
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def draw_robot(env, self_model, loop):
        x = [0]
        y = [0]
        last_deg = 0.0
        for coord in env.Ys:
            x.append(coord[0])
            y.append(coord[1])
        # for j in range(env.r.size):
        #     x.append(x[-1] + float(env.r[j] * math.cos(last_deg + env.x[j])))
        #     y.append(y[-1] + float(env.r[j] * math.sin(last_deg + env.x[j])))
        #     last_deg += env.x[j]
        #     # masses.append(y)
        # plt.clf()
        plt.clf()
        x = [p + env.pos[0] for p in x]
        y = [p + env.pos[1] for p in y]
        plt.plot(x, y, 'ro', linestyle='-', color='black')

        minX = -2
        maxX = 5
        plt.axis([minX, maxX, 0.0, 2.0])
        plt.pause(0.001)
        plt.draw()

        # d = math.sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2)
        # print(d)


    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    env = SimpleCrawler()
    self_model = None
    loop = None

    plt.figure()
    # timer = fig.canvas.new_timer(interval=300)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    plt.subplot(111, aspect='equal')
    # plt.show()
    plt.ion()
    obs = env.reset()

    # Inch forward
    draw_robot(env, self_model, loop)

    # Resetting
    target = np.array([3 * math.pi / 4, math.pi / 4])
    while sum(np.abs(target-env.x)) > 0:
        action = target - env.x
        for i in range(len(action)):
            if np.abs(action[i]) > math.pi / 16: action[i] = np.sign(action[i]) * math.pi / 16
        env.step(action)
        print(action)
        print(env.x)
        draw_robot(env, self_model, loop)
    i = 0
    while env.pos[0] < 5:
        y = env.__get_pos__(env.x)
        while y[1] > 0:
            # env.last_x = np.zeros(2)+env.x
            # env.x[0] += math.pi/64
            action = np.array([math.pi / 64, 0])
            env.step(action)
            draw_robot(env, self_model, loop)
            y = env.__get_pos__(env.x)
        # print(y[1])
        while y[0] > -1.99:
            # env.last_x = np.zeros(2)+env.x
            # env.x[0] += math.pi/64
            # env.x[1] -= math.pi/32
            action = np.array([math.pi / 64, -math.pi / 32])
            env.step(action)
            draw_robot(env, self_model, loop)
            y = env.__get_pos__(env.x)
        print(env.pos)
        while env.x[0] > 7 * math.pi / 8 or env.x[1] < math.pi / 6:
            # env.last_x = np.zeros(2)+env.x
            action = np.zeros(2)
            if env.x[0] > 7 * math.pi / 8:
                # env.x[0] -= math.pi / 64
                action[0] -= math.pi / 64
            if env.x[1] < math.pi / 6:
                # env.x[1] += math.pi / 64
                action[1] += math.pi / 64
            env.step(action)
            draw_robot(env, self_model, loop)
        i += 1
    print(i)

    # Inch backward
    # env.last_x = np.array([math.pi/4,-math.pi/4])
    # env.x = np.array([math.pi/4,-math.pi/4])
    # draw_robot(env)
    # while True:
    #     y = env.__get_pos__(env.x)
    #     while y[1] > 0:
    #         env.last_x = np.zeros(2)+env.x
    #         env.x[0] -= math.pi/64
    #         draw_robot(env)
    #         y = env.__get_pos__(env.x)
    #     while y[0] < 1.99:
    #         env.last_x = np.zeros(2)+env.x
    #         env.x[0] -= math.pi/64
    #         env.x[1] += math.pi/32
    #         draw_robot(env)
    #         y = env.__get_pos__(env.x)
    #     while env.x[0] < math.pi/8 or env.x[1] > -math.pi/6:
    #         env.last_x = np.zeros(2)+env.x
    #         if env.x[0] < math.pi/8:
    #             env.x[0] += math.pi / 64
    #         if env.x[1] > -math.pi/6:
    #             env.x[1] -= math.pi / 64
    #         draw_robot(env)

    # Crawl backward
    # while True:
    #     y = env.__get_pos__(env.x)
    #     while y[1] > 0:
    #         env.last_x = np.zeros(2)+env.x
    #         env.x[0] += math.pi/64
    #         draw_robot(env)
    #         y = env.__get_pos__(env.x)
    #     while y[0] < -1.25:
    #         env.last_x = np.zeros(2)+env.x
    #         env.x[0] -= math.pi/64
    #         env.x[1] += math.pi/32
    #         draw_robot(env)
    #         y = env.__get_pos__(env.x)
    #         print(y[0])
    #     env.x[0] -= math.pi/32
    #     draw_robot(env)
    #     while env.x[0] < 7*math.pi/8 or env.x[1] > math.pi/4:
    #         env.last_x = np.zeros(2)+env.x
    #         # if env.x[1] < math.pi/2:
    #         if env.x[0] < 7*math.pi/8:
    #             env.x[0] += math.pi / 64
    #         if env.x[1] > math.pi/4:
    #             env.x[1] -= math.pi / 32
    #         draw_robot(env)

    # Crawl forwards
    # env.last_x = np.array([math.pi / 4, -math.pi / 4])
    # env.x = np.array([math.pi / 4, -math.pi / 4])
    # draw_robot(env, self_model, loop)
    # i = 0
    # while env.pos[0] < 5:
    #     y = env.__get_pos__(env.x)
    #     while y[1] > 0:
    #         env.last_x = np.zeros(2) + env.x
    #         env.x[0] -= math.pi / 64
    #         draw_robot(env, self_model, loop)
    #         y = env.__get_pos__(env.x)
    #     while y[0] > 1.25:
    #         env.last_x = np.zeros(2) + env.x
    #         env.x[0] += math.pi / 64
    #         env.x[1] -= math.pi / 32
    #         draw_robot(env, self_model, loop)
    #         y = env.__get_pos__(env.x)
    #         print(y[0])
    #     env.x[0] += math.pi / 32
    #     draw_robot(env)
    #     while env.x[0] > math.pi / 8 or env.x[1] < -math.pi / 4:
    #         env.last_x = np.zeros(2) + env.x
    #         # if env.x[1] < math.pi/2:
    #         if env.x[0] > math.pi / 8:
    #             env.x[0] -= math.pi / 64
    #         if env.x[1] < -math.pi / 4:
    #             env.x[1] += math.pi / 32
    #         draw_robot(env, self_model, loop)
    #     i += 1
    # print(i)

    while True: pass

