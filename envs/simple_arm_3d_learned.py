import math

import gym
import numpy as np

from env_learners.env_learner import EnvLearner
from envs.simple_arm_3d import SimpleArm as SimpleArmRaw


class SimpleArm(gym.Env):
    def __init__(self):
        self.r = SimpleArmRaw().r
        self.max_iter = SimpleArmRaw().max_iter
        self.t = 0

        self.action_space = SimpleArmRaw().action_space

        self.observation_space = SimpleArmRaw().observation_space

        self.obs = self.reset()
        self.inited = False

    def init(self, sess):
        self.env_learner = EnvLearner(SimpleArmRaw(train=True))
        self.env_learner.initialize(sess)
        train, valid = self.__gen_train_data__()
        print('Data Gathered')
        self.__train_self_model__(train, valid)
        print('Model Trained')
        self.inited = True

    def __gen_train_data__(self):
        env = SimpleArmRaw(train=True)
        train_episodes = 100
        nb_valid_episodes = 50
        episode_duration = -1
        max_action = env.action_space.high
        episode_step = 0.0
        episode_reward = 0.0
        max_ep_rew = -1000

        train = []
        valid = []
        obs = env.reset()
        i = 0
        while i < train_episodes:
            action = np.random.uniform(-1, 1, env.action_space.shape[0])
            # action = find_next_move(env, self.env_learner, obs, max_action, episode_step)
            new_obs, r, done, info = env.step(max_action * action)
            if episode_duration > 0:
                done = (done or (episode_step >= episode_duration))
            train.append([obs, max_action * action, r, new_obs, done, episode_step])
            episode_step += 1
            obs = new_obs

            episode_reward += r
            if done:
                episode_step = 0.0
                obs = env.reset()
                max_ep_rew = max(max_ep_rew, episode_reward)
                episode_reward = 0.0
                i += 1

        i = 0
        while i < nb_valid_episodes:
            action = np.random.uniform(-1, 1, env.action_space.shape[0])
            # action = find_next_move(env, self.env_learner, obs, max_action, episode_step)
            new_obs, r, done, info = env.step(max_action * action)
            if episode_duration > 0:
                done = (done or (episode_step >= episode_duration))
            valid.append([obs, max_action * action, r, new_obs, done, episode_step])
            episode_step += 1
            obs = new_obs

            episode_reward += r
            if done:
                obs = env.reset()
                max_ep_rew = max(max_ep_rew, episode_reward)
                episode_reward = 0.0
                i += 1
        return train, valid

    def __train_self_model__(self, train, valid):
        total_steps = 50
        log_interval = 10
        import time
        
        min_loss = 10000000000
        stop_count = 0
        for i in range(total_steps):
            if i > 0 and i % (
                total_steps / self.env_learner.max_seq_len) == 0 and self.env_learner.seq_len < self.env_learner.max_seq_len:
                self.env_learner.seq_len += 1
                print('Sequence Length: ' + str(self.env_learner.seq_len))

            if i % log_interval == 0 and valid is not None:
                (vGen, vDisc, vC) = self.env_learner.get_loss(valid)
                print('Epoch: ' + str(i) + '/' + str(total_steps))
                print('Valid Loss')
                print('Gen:  ' + str(vGen))
                print('Disc: ' + str(vDisc))
                print('Close: ' + str(vC))
                print()
                # if saver is not None and save_str is not None:
                #     save_path = saver.save(self.env_learner.sess, 'models/' + str(save_str) + '.ckpt')
                #     print("Model saved in path: %s" % save_path)
            start = time.time()
            tlGen, tlDisc = self.env_learner.train_adv(train)
            duration = time.time() - start
            if tlGen < min_loss:
                min_loss = tlGen
                stop_count = 0
            else:
                stop_count += 1
            if i % log_interval != 0:
                print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                print('Train Loss')
                print('Gen:  ' + str(tlGen))
                print('Disc: ' + str(tlDisc))
                print()
        if valid is not None:
            (vGen, vDisc, vC) = self.env_learner.get_loss(valid)
            print('Final Epoch: ')
            print('Valid Loss')
            print('Gen:  ' + str(vGen))
            print('Disc: ' + str(vDisc))
            print('Close: ' + str(vC))
            print()
        # if saver is not None and save_str is not None:
        #     save_path = saver.save(self.env_learner.sess, 'models/' + str(save_str) + '.ckpt')
        #     print("Final Model saved in path: %s" % save_path)
    #
    # def __get_obs__(self):
    #     return np.concatenate([self.x, self.y, np.array([self.d])], axis=0)

    def __get_obs__(self):
        elbows = []
        last_ver = 0.0
        last_hor = 0.0
        elbow = np.zeros(3)
        for j in range(self.r.size-1):
            elbow[0] += float(self.r[j]*math.cos(last_hor+self.x[2*j])*math.sin(math.pi/2-last_ver-self.x[2*j+1]))
            elbow[1] += float(self.r[j]*math.sin(last_hor+self.x[2*j])*math.sin(math.pi/2-last_ver-self.x[2*j+1]))
            elbow[2] += float(self.r[j]*math.cos(math.pi/2-last_ver-self.x[2*j+1]))
            elbows.append(elbow)
        elbows = np.concatenate(elbows)
        return np.concatenate([self.x, elbows, self.y, np.array([self.d])], axis=0)
    
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

    def reset(self):
        self.t = 0
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
    
    def step(self, action):
        new_obs = self.env_learner.step(self.obs[:-1], action, self.t)
        self.t += 1
        d = np.linalg.norm(self.target - new_obs[-3:])
        if self.t == 1:
           self.rew = -d
        else:
            self.rew = self.d-d
        self.d = d
        self.obs = np.concatenate([new_obs, np.array([self.d])])
        self.done = (self.t >= self.max_iter)
        return self.obs, self.rew, self.done, {}