import numpy as np
from collections import deque

class EnvLearner:
    def __init__(self, env_in):
        self.state_mul_const = env_in.observation_space.high
        self.act_mul_const = env_in.action_space.high
        self.act_dim = env_in.action_space.shape[0]
        self.state_dim = env_in.observation_space.shape[0]

        self.action_space = env_in.action_space
        self.observation_space = env_in.observation_space

        self.buff_init = [np.zeros(self.state_dim+self.act_dim)]
        self.seq_init = [np.zeros(self.act_dim)]

        # To be changed by child classes
        self.buff_len = 1
        self.seq_len = 1
        self.max_seq_len = 1

    def initialize(self, session, load=False):
        pass

    def __batch__(self, data, batch_size):
        batches = []
        while len(data) >= batch_size:
            batches.append(data[:batch_size])
            data = data[batch_size:]
        return batches

    def __prep_data__(self, data, batch_size=32):
        G = []
        X = []
        A = []
        yS = []
        yR = []
        yD = []
        S = []
        is_new = True

        t = 0
        for (obs, action, r, new_obs, done, episode_step) in data:
            # Normalizing to [-1, 1]
            obs = obs/self.state_mul_const
            action = action/self.act_mul_const
            new_obs = new_obs/self.state_mul_const
            zD = np.array([int(done)])

            if is_new:
                x_episode = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
                x_episode.append(np.array([np.concatenate([obs, action])]).flatten())
                is_new = False
                t = 0

                if len(A) > 0:
                    for i in range(self.max_seq_len-1):
                        a_seq.append(np.concatenate([np.zeros(action.size)]))
                        y_seq.append(yS[-1])
                        A.append(np.concatenate(a_seq).flatten())
                        S.append(np.concatenate(y_seq).flatten())
                a_seq = deque([np.zeros(action.size)] * self.max_seq_len, maxlen=self.max_seq_len)
                y_seq = deque([np.zeros(new_obs.size)] * self.max_seq_len, maxlen=self.max_seq_len)
            else:
                x_episode.append(np.array([np.concatenate([obs, action])]).flatten())
            t += 1
            a_seq.append(action)
            y_seq.append(new_obs)

            if t > self.max_seq_len-1:
                A.append(np.concatenate(a_seq).flatten())
                S.append(np.concatenate(y_seq).flatten())
            yD.append(zD.flatten())
            yS.append(np.array([new_obs]).flatten())
            yR.append(np.array([r]))
            X.append(np.concatenate(x_episode).flatten())
            if done:
                is_new = True
        if len(A) > 0:
            for i in range(self.max_seq_len-1):
                a_seq.append(np.concatenate([np.zeros(action.size)]))
                y_seq.append(yS[-1])
                A.append(np.concatenate(a_seq).flatten())
                S.append(np.concatenate(y_seq).flatten())

        assert len(X) == len(yS) == len(yR) == len(yD) == len(S) == len(A)

        p = np.random.permutation(len(X))
        X = np.array(X)[p]
        yS = np.array(yS)[p]
        yR = np.array(yR)[p]
        yD = np.array(yD)[p]
        S = np.array(S)[p]
        A = np.array(A)[p]


        if batch_size > 0:
            X = self.__batch__(X, batch_size)
            yS = self.__batch__(yS, batch_size)
            yR = self.__batch__(yR, batch_size)
            yD = self.__batch__(yD, batch_size)
            S = self.__batch__(S, batch_size)
            A = self.__batch__(A, batch_size)
        else:
            G = [G]
            X = [X]
            yS = [yS]
            yR = [yR]
            yD = [yD]
            S = [S]
            A = [A]
        return G, yS, yR, yD, X, S, A

    def train(self, train, total_steps, valid=None, log_interval=10, early_stopping=-1, saver=None, save_str=None):
        pass

    def get_loss(self, data):
        pass

    def step(self, obs_in, action_in, episode_step, save=True, buff=None):
        pass
