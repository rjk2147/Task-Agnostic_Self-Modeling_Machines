import numpy as np
import math

class Spring:
    def __init__(self, l=1, k=10000):
        self.l0 = l
        self.k = k
        self.F = None
        self.l = None
        self.min=False
        self.ends = [None, None]

    def update(self):
        if None in self.ends: return
        self.l = dist(self.ends[0].p, self.ends[1].p)
        # Magnitude of the force
        # positive force is outward, negative force is inward
        self.F = self.k*(self.l0-self.l)
        if self.min and self.l > self.min:
            self.F = 0

class Mass:
    def __init__(self, m=0.2, p=None):
        self.springs = []
        self.m = m
        self.F = np.zeros(3)
        self.a = np.zeros(3)
        self.v = np.zeros(3)
        if p is None: self.p = np.zeros(3)
        else: self.p = np.array(p,dtype='float64')

    def connect(self, mass, spring=None):
        if mass == self: return
        if spring is None: spring = Spring()
        spring.ends = [self, mass]
        spring.l0 = dist(spring.ends[0].p, spring.ends[1].p)
        spring.l = dist(spring.ends[0].p, spring.ends[1].p)
        # Magnitude of the force
        spring.F = spring.k*(spring.l-spring.l0)

        self.springs.append(spring)
        mass.springs.append(spring)
        return spring

    def update_F(self, Fext=np.zeros(3)):
        self.F = np.zeros(3)+Fext

        for spring in self.springs:
            if spring.ends[0] == self: m = spring.ends[1]
            elif spring.ends[1] == self: m = spring.ends[0]
            else: m = None
            if m is not None:
                # positive force is outward, negative force is inward
                u = (self.p - m.p)/np.linalg.norm(self.p - m.p)
                self.F += 0.9*spring.F*u
            else: print('(E) Mass not found in spring.')
        # self.F -= 0.1*np.sign(self.F)*np.ones(3)
    def update_final(self, damp_c, dt, u):
        damp_c = (1-dt/100)
        self.a = self.F/self.m
        self.v += self.a*dt
        self.v = damp_c*self.v
        self.p += self.v*dt
        if self.p[2] <= 0:
            # Undoing forces before recalculation for friction
            self.p -= self.v*dt
            self.v -= self.a*dt

            # Friction calculations
            N = np.abs(self.F[2])
            fric = N * u
            self.F[0] -= np.sign(self.v[0])*abs(fric)
            self.F[1] -= np.sign(self.v[1])*abs(fric)

            # Redoing forces calculation after friction
            self.a = self.F / self.m
            self.v += self.a*dt
            self.v = (1-2*dt)*self.v
            self.p += self.v*dt

            # Bouncing effect with 50% force loss into the ground
            self.p[2] *= -0.5
            self.v[2] *= -0.5

def dist(a, b):
    return np.linalg.norm(a-b)

def update_all(masses, springs, dt=0.0001, Fext=np.zeros(3), u=0.0, damp_c=0.99999):
    multithreaded = False
    import time
    start = time.time()
    if multithreaded:
        import multiprocessing as mp

        def mass_update_F(Fext, queue):
            mass = queue.get()
            mass.update_F(Fext)
            queue.put(mass)
        def mass_update_final(damp_c, dt, u, queue):
            mass = queue.get()
            mass.update_final(damp_c, dt, u)
            # queue.put(mass)
        def spring_update(queue):
            spring = queue.get()
            spring.update()
            # queue.put(spring)

        mass_queue = mp.JoinableQueue()
        spring_queue = mp.JoinableQueue()
        jobs = []
        for mass in masses: mass_queue.put(mass)
        for spring in springs: spring_queue.put(spring)


        for i in range(len(masses)):
            p = mp.Process(target=mass_update_F, args=(Fext,mass_queue))
            p.start()
            jobs.append(p)
            # mass.update_F(Fext)
        for job in jobs:    job.join()
        jobs = []
        # print('Force')

        for i in range(len(masses)):
            p = mp.Process(target=mass_update_final, args=(damp_c, dt, u, mass_queue))
            p.start()
            jobs.append(p)
            # mass.update_final(damp_c, dt, u)
        for job in jobs:    job.join()
        jobs = []
        # print('Final')

        for i in range(len(springs)):
            p = mp.Process(target=spring_update, args=(spring_queue,))
            p.start()
            jobs.append(p)
            # spring.update()
        # for spring in springs:
        #     spring.update()
        for job in jobs:    job.join()
        # print('Spring')
    else:
        for mass in masses:
            mass.update_F(Fext)
        for mass in masses:
            mass.update_final(damp_c, dt, u)
        for spring in springs:
            spring.update()
    # print(time.time()-start)

def make_cube():
    # masses = [
    #     Mass(p=[0,0,0]),
    #     Mass(p=[0,0,1]),
    #     Mass(p=[0,1,0]),
    #     Mass(p=[0,1,1]),
    #     Mass(p=[1,0,0]),
    #     Mass(p=[1,0,1]),
    #     Mass(p=[1,1,0]),
    #     Mass(p=[1,1,1]),
    # ]
    masses = [
        Mass(p=[0,-0.1,0.2]),
        Mass(p=[0,0.1,0.2]),
        Mass(p=[-0.1,0,0.2]),
        Mass(p=[0.1,0,0.2]),
        Mass(p=[0,-0.1,0]),
        Mass(p=[0,0.1,0]),
        Mass(p=[-0.1,0,0]),
        Mass(p=[0.1,0,0]),
    ]
    springs = []
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            spring = masses[i].connect(masses[j])
            springs.append(spring)
    return masses, springs

def __fall_test__():
    import matplotlib.pyplot as plt
    z = []
    x = []

    dt = 0.01
    masses, springs = make_cube()
    # masses = [Mass(p=[0,0,0])]
    # springs = []
    g = np.array([0, 0, -9.8])
    for mass in masses:
        mass.p[2] += 1
    avg_p = np.zeros(3)
    for mass in masses: avg_p += mass.p
    avg_v = np.zeros(3)
    for mass in masses: avg_v += mass.v
    avg_p /= len(masses)
    z.append(avg_p[2])
    x.append(avg_p[0])
    for i in range(int(10.0/dt)):
        update_all(masses, springs, dt=dt, Fext=g)
        avg_p = np.zeros(3)
        for mass in masses: avg_p += mass.p
        avg_v = np.zeros(3)
        for mass in masses: avg_v += mass.v
        avg_p /= len(masses)
        z.append(avg_p[2])
        x.append(avg_p[0])
    plt.plot(x, z)
    plt.show()

def GCP(env, train_episodes=1000, batch_size=1):
    i = 0
    episode_duration = 10000
    max_action = env.action_space.high
    train = []
    episode_reward = 0.0
    episode_step = 0
    obs = env.reset()
    # import random

    # print('Episode 0')
    # print(obs)
    amp_std = 0.01
    off_std = 0.01
    np.random.seed(0)
    # amplitude = random.normalvariate(0, amp_std)
    # offset = random.normalvariate(0, off_std)


    # amplitude = np.array([0.00339, 0.02484])
    # offset = np.array([-0.0032,   0.00055])


    # amplitude = np.array([0.02119, 0.02453])
    # offset = np.array([ 0.01257, -0.00758])


    amplitude = np.array([0.09295, 0.04523])
    offset = np.array([0.02528, 0.06453])
    # amplitude = np.array([0., 0.])
    # offset = np.array([0.,  0.])
    best_rew = 0.0
    best_amp = amplitude
    best_off = offset

    while i < train_episodes / batch_size:
        avg_rew = 0.0
        avg_i = 0
        while avg_i < batch_size:

            # Below needed for GCP
            joints = env.action_space.shape[0]
            action = np.zeros(joints)
            for j in range(joints):
                action[j] = amplitude[j]*math.sin(2*math.pi*env.dt*episode_step+math.pi*offset[j])

            # Use uniform for training data generation
            # action = np.random.uniform(-1, 1, env.action_space.shape[0])

            # action = np.zeros(env.action_space.shape[0])
            # action = find_next_move(env, env_learner, obs, max_action, episode_step)
            new_obs, r, done, info = env.step(max_action * action)

            if episode_step%100 == 0:
                env.render()

            if episode_duration > 0:
                done = (done or (episode_step >= episode_duration))

            # train.append([obs, max_action * action, r, new_obs, done, episode_step])
            episode_step += 1
            # print(episode_step)
            obs = new_obs
            # print(obs)

            # vx = new_obs[3]
            # rew = vx
            episode_reward += r
            if done:
                # joints = env.action_space.shape[0]
                # action = np.zeros(joints)
                # for i in range(joints):
                #     action[i] = amplitude*math.sin(2*math.pi*0.0165*episode_step+math.pi*offset)
                # print(obs)
                # print('Episode '+str(i))
                avg_rew += episode_reward
                episode_step = 0.0
                obs = env.reset()
                # max_ep_rew = max(max_ep_rew, episode_reward)
                episode_reward = 0.0
                avg_i += 1
        i += 1
        avg_rew = float(avg_rew) / float(batch_size)
        if avg_rew > best_rew:
            best_rew = avg_rew
            best_amp = amplitude
            best_off = offset
            print
            print('Best Amplitude: '+str(amplitude))
            print('Best Offset: '+str(offset))
        amplitude = np.round(best_amp + np.random.normal(0, amp_std, size=2), 5)
        offset = np.round(best_off + np.random.normal(0, off_std, size=2), 5)

        print('Train Batch: '+str(i)+': '+str(avg_rew)+' Max: '+str(best_rew))
    # print('Valid Avg: '+str(avg_rew))
    print('Best Amplitude: '+str(amplitude))
    print('Best Offset: '+str(offset))


if __name__ == '__main__':
    self_model = None
    loop = None
    import spring_mass_env
    env = spring_mass_env.SpringMassCrawler()

    # plt.figure()
    # timer = fig.canvas.new_timer(interval=300)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    # plt.subplot(111, aspect='equal')
    # plt.show()
    # plt.ion()
    obs = env.reset()

    # GCP(env)
    while env.i < 10000:
        print(env.muscles[1].l0)
        print(env.masses[-2].p)
        print('Step 1')
        while env.muscles[1].l0 < 0.4:
            action = np.array([0.,0.01])
            state, rew, done, _ = env.step(action)
            print(state)
            env.render()
        # print(env.masses[-2].p)
        print('Step 2')
        # for i in range(100):
        action = np.array([-0.01,0.])
        state, rew, done, _ = env.step(action)
        print(state)
        env.render()
        # print('Step 3')
        while env.muscles[1].l0 > 0.2449:
            action = np.array([0.,0.])
            action[1] -= 0.01
            state, rew, done, _ = env.step(action)
            print(state)
            env.render()
        # print('Step 4')
        # for i in range(100):
        action = np.array([0.01,0.])
        state, rew, done, _ = env.step(action)
        print(state)
        env.render()
        # print(env.i)