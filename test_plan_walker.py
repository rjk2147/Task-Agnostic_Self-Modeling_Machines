import time, datetime, math
from collections import deque
import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env_learner import EnvLearner
import gym
from gym import spaces

class AntWrapper(gym.Env):
    def __init__(self, ant_env):
        self.env = ant_env
        self.action_space = self.env.action_space
        obs_ones = np.ones(shape=(self.env.observation_space.shape[0],))
        self.observation_space = spaces.Box(high=5*obs_ones, low=-5*obs_ones)
        pass

    def reset(self):
        return self.env.reset()
    def step(self, action):
        new_obs, r, done, info = self.env.step(action)
        return new_obs, r, done, info

def run_tests(test_episodes, env, env_learner, loop='open'):
    i = 0
    batch_size = 1
    train_episodes = 1000
    episode_duration = -1
    # nb_valid_episodes = 50
    episode_step = 0
    episode_reward = 0.0
    real_episode_reward = 0.0
    max_ep_rew = -10000

    nb_valid_episodes = test_episodes
    # env = env_learner

    amp_std = 0.1
    off_std = 0.1
    np.random.seed(0)

    amplitude = 1.
    offset = 0.

    best_rew = -1000.0
    real_best_rew = 0.0
    best_amp = amplitude
    best_off = offset

    max_action = env.action_space.high


    obs = env.reset()
    while i < train_episodes/batch_size:
        avg_rew = 0.0
        real_avg_rew = 0.0
        avg_i = 0
        while avg_i < batch_size:

            # Below needed for GCP
            joints = env.action_space.shape[0]
            action = np.zeros(joints)
            for j in range(joints):
                action[j] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)

            # Use uniform for training data generation
            # action = np.random.uniform(-1, 1, env.action_space.shape[0])


            # action = np.zeros(env.action_space.shape[0])
            # action = find_next_move(env, env_learner, obs, max_action, episode_step)

            new_obs = env_learner.step(obs, max_action * action, episode_step)
            real_obs, r, real_done, _ = env.step(max_action * action)

            done = real_done

            # env.render()
            if episode_duration > 0:
                done = (done or (episode_step >= episode_duration))


            # train.append([obs, max_action * action, r, new_obs, done, episode_step])
            episode_step += 1

            if loop == 'open':
                obs = new_obs
            elif loop == 'closed':
                obs = real_obs
            # print(obs)

            vx = new_obs[3]
            rew = vx
            episode_reward += rew
            real_vx = real_obs[3]
            real_rew = real_vx
            real_episode_reward += real_rew

            if done:
                # joints = env.action_space.shape[0]
                # action = np.zeros(joints)
                # for i in range(joints):
                #     action[i] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)
                # print(obs)
                # print('Episode '+str(i))
                avg_rew += episode_reward
                real_avg_rew += real_episode_reward
                episode_step = 0.0
                obs = env.reset()
                max_ep_rew = max(max_ep_rew, episode_reward)
                episode_reward = 0.0
                real_episode_reward = 0.0
                avg_i += 1
        i += 1
        avg_rew = float(avg_rew)/float(batch_size)
        real_avg_rew = float(real_avg_rew)/float(batch_size)
        if avg_rew > best_rew:
            best_rew = avg_rew
            real_best_rew = real_avg_rew

            best_amp = amplitude
            best_off = offset
        amplitude = best_amp + np.random.normal(0, amp_std)
        offset = best_off + np.random.normal(0, off_std)
        print('Train Batch: '+str(i)+': '+str(avg_rew)+'\tReal: '+str(real_avg_rew)+'  \tMax: '+str(best_rew)+'  \tReal Max: '+str(real_best_rew))

    i = 0
    amplitude = best_amp
    offset = best_off
    avg_rew = 0.0
    episode_reward = 0.0
    obs = env.reset()
    while i < nb_valid_episodes:

        # Below needed for GCP
        joints = env.action_space.shape[0]
        action = np.zeros(joints)
        for j in range(joints):
            action[j] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)

        # Uniform for random validation
        # action = np.random.uniform(-1, 1, env.action_space.shape[0])
        # action = find_next_move(env, env_learner, obs, max_action, episode_step)
        new_obs, r, done, info = env.step(max_action * action)
        if episode_duration > 0:
            done = (done or (episode_step >= episode_duration))
        # valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs

        vx = new_obs[3]
        rew = vx
        episode_reward += rew
        if done:
            avg_rew += episode_reward
            print('Valid Episode '+str(i)+': '+str(episode_reward))
            episode_step = 0.0
            obs = env.reset()
            max_ep_rew = max(max_ep_rew, episode_reward)
            episode_reward = 0.0
            i += 1
        # if done:
        #     obs = env.reset()
        #     max_ep_rew = max(max_ep_rew, episode_reward)
        #     print('Valid Episode '+str(i)+': '+str(episode_reward))
        #     avg_rew += episode_reward
        #     episode_reward = 0.0
        #     i += 1

    avg_rew = avg_rew/float(nb_valid_episodes)
    print('Valid Avg: '+str(avg_rew))
    print('Best Amplitude: '+str(amplitude))
    print('Best Offset: '+str(offset))
    logger.info('Data gathered')
    # logger.info('Train Size: ' + str(len(train)))
    # logger.info('Valid Size: ' + str(len(valid)))

#
# def run_tests(test_episodes, env, data_log, env_learner, max_action, loop):
#     episode_step = 0
#     # last_d = env.d
#     # d = last_d
#     acts = []
#     all_final_drifts = []
#     all_final_lens = []
#     all_final_real_ds = []
#     all_final_pred_ds = []
#     test = []
#     failures = 0
#     for i in range(test_episodes):
#         start_time = time.time()
#         done = False
#         obs = env.reset()
#         # init_d = np.linalg.norm(env.target - obs[-3:])
#         # data_log.write('Episode: ' + str(i) + ' Start\n')
#         # data_log.write('Target: ' + str(env.target) + '\n')
#         # data_log.write('Pred State: ' + str(obs) + '\n')
#         # data_log.write('Pred D: ' + str(init_d) + '\n')
#         # data_log.write('Real State: ' + str(obs) + '\n')
#         # data_log.write('Real D: ' + str(init_d) + '\n')
#         # data_log.write('Drift: ' + str(0.0) + '\n')
#         # start_pos = [obs[-3], obs[-2], obs[-1]]
#
#         real_Xs = []
#         real_Ys = []
#         real_Zs = []
#         pred_Xs = []
#         pred_Ys = []
#         pred_Zs = []
#
#         real_elbow_Xs = []
#         real_elbow_Ys = []
#         real_elbow_Zs = []
#         pred_elbow_Xs = []
#         pred_elbow_Ys = []
#         pred_elbow_Zs = []
#
#         pred_ds = []
#         real_ds = []
#         drifts = []
#         # real_Xs.append(obs[-3])
#         # real_Ys.append(obs[-2])
#         # real_Zs.append(obs[-1])
#         # pred_Xs.append(obs[-3])
#         # pred_Ys.append(obs[-2])
#         # pred_Zs.append(obs[-1])
#         #
#         # real_elbow_X = [0]
#         # real_elbow_Y = [0]
#         # real_elbow_Z = [0]
#         # pred_elbow_X = [0]
#         # pred_elbow_Y = [0]
#         # pred_elbow_Z = [0]
#         # for j in range(len(env.r) - 1):
#         #     real_elbow_X.append(float(env.r[j] * math.cos(obs[2 * j]) * math.sin(obs[2 * j + 1])))
#         #     real_elbow_Y.append(float(env.r[j] * math.sin(obs[2 * j]) * math.sin(obs[2 * j + 1])))
#         #     real_elbow_Z.append(float(env.r[j] * math.cos(obs[2 * j + 1])))
#         #     pred_elbow_X.append(float(env.r[j] * math.cos(obs[2 * j] * math.sin(obs[2 * j + 1]))))
#         #     pred_elbow_Y.append(float(env.r[j] * math.sin(obs[2 * j]) * math.sin(obs[2 * j + 1])))
#         #     pred_elbow_Z.append(float(env.r[j] * math.cos(obs[2 * j + 1])))
#         # real_elbow_Xs.append(real_elbow_X)
#         # real_elbow_Ys.append(real_elbow_Y)
#         # real_elbow_Zs.append(real_elbow_Z)
#         # pred_elbow_Xs.append(pred_elbow_X)
#         # pred_elbow_Ys.append(pred_elbow_Y)
#         # pred_elbow_Zs.append(pred_elbow_Z)
#         #
#         # pred_ds.append(init_d)
#         # real_ds.append(init_d)
#         # drifts.append(0)
#         real_dx = 0.0
#         pred_dx = 0.0
#         while not done:
#             action = find_next_move_test(env, env_learner, obs, max_action, episode_step, dof=4)
#             # action = find_next_move_train(env, env_learner, obs, max_action, episode_step, dof=4)
#             new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
#             real_obs, r, done, _ = env.step(max_action * action)
#             pred_dx += (0.0165)*new_obs[3]/0.3
#             real_dx += (0.0165)*real_obs[3]/0.3
#             # d = np.linalg.norm(env.target - new_obs[-3:])
#             # real_d = np.linalg.norm(env.target - real_obs[-3:])
#             # test.append([obs, max_action * action, 0.0, new_obs, done, episode_step])
#             # acts.append(action)
#             # drift = np.linalg.norm(real_obs[-3:] - new_obs[-3:])
#             episode_step += 1
#             # data_log.write('Action ' + str(episode_step) + ': ' + str(action) + '\n')
#             # data_log.write('Real Reward: ' + str(r) + '\n')
#             # data_log.write('Pred State: ' + str(new_obs) + '\n')
#             # data_log.write('Pred D: ' + str(d) + '\n')
#             # data_log.write('Real State: ' + str(real_obs) + '\n')
#             # data_log.write('Real D: ' + str(real_d) + '\n')
#             # data_log.write('Drift: ' + str(drift) + '\n')
#
#             # print('Action '+str(episode_step)+': '+str(action)+'\n')
#             # print('Pred D: '+str(d)+'\n')
#             # print('Real D: '+str(real_d)+'\n')
#             # print('Drift: '+str(drift)+'\n')
#
#             # real_Xs.append(real_obs[-3])
#             # real_Ys.append(real_obs[-2])
#             # real_Zs.append(real_obs[-1])
#             # pred_Xs.append(new_obs[-3])
#             # pred_Ys.append(new_obs[-2])
#             # pred_Zs.append(new_obs[-1])
#             #
#             # real_elbow_X = [0]
#             # real_elbow_Y = [0]
#             # real_elbow_Z = [0]
#             # pred_elbow_X = [0]
#             # pred_elbow_Y = [0]
#             # pred_elbow_Z = [0]
#             # for j in range(len(env.r) - 1):
#             #     real_elbow_X.append(float(env.r[j] * math.cos(real_obs[2 * j]) * math.sin(real_obs[2 * j + 1])))
#             #     real_elbow_Y.append(float(env.r[j] * math.sin(real_obs[2 * j]) * math.sin(real_obs[2 * j + 1])))
#             #     real_elbow_Z.append(float(env.r[j] * math.cos(real_obs[2 * j + 1])))
#             #     pred_elbow_X.append(float(env.r[j] * math.cos(new_obs[2 * j] * math.sin(new_obs[2 * j + 1]))))
#             #     pred_elbow_Y.append(float(env.r[j] * math.sin(new_obs[2 * j]) * math.sin(new_obs[2 * j + 1])))
#             #     pred_elbow_Z.append(float(env.r[j] * math.cos(new_obs[2 * j + 1])))
#             # real_elbow_Xs.append(real_elbow_X)
#             # real_elbow_Ys.append(real_elbow_Y)
#             # real_elbow_Zs.append(real_elbow_Z)
#             # pred_elbow_Xs.append(pred_elbow_X)
#             # pred_elbow_Ys.append(pred_elbow_Y)
#             # pred_elbow_Zs.append(pred_elbow_Z)
#
#             # pred_ds.append(d)
#             # real_ds.append(real_d)
#             # drifts.append(drift)
#
#             if loop == 'open':
#                 obs = new_obs
#             elif loop == 'closed':
#                 obs = real_obs
#                 # d = real_d
#             else:
#                 obs = new_obs
#
#             # done = episode_step > env.max_iter
#
#             # if d < 0.01:
#             #     done = True
#             # if episode_step%100 == 0:
#             #     print(episode_step)
#
#             if done:
#                 data_log.write('Episode: ' + str(episode_step) + ' done\n\n')
#
#                 print('Episode: ' + str(i) + ' in ' + str(time.time() - start_time) + ' seconds')
#                 print(str(episode_step) + '\nPred Dx: ' + str(pred_dx) + '\nReal Dx: ' + str(real_dx))
#                 # print(str(episode_step) + '\nPred D: ' + str(d) + '\nReal D: ' + str(real_d))
#                 # print('Drift: ' + str(drift))
#                 # if d < 0.01:
#                 #     all_final_drifts.append(drift)
#                 #     all_final_lens.append(episode_step)
#                 #     all_final_pred_ds.append(d)
#                 #     all_final_real_ds.append(real_d)
#                 # else:
#                 #     failures += 1
#                 # done = False
#                 episode_step = 0
#
#                 # Plotting
#                 # fig = plt.figure()
#                 # ax = fig.add_subplot(111, projection='3d')
#                 #
#                 #
#                 # if show_model:
#                 #     for j in range(len(real_Xs)):
#                 #         armX = []
#                 #         armY = []
#                 #         armZ = []
#                 #         armX.extend(real_elbow_Xs[j])
#                 #         armX.append(real_Xs[j])
#                 #
#                 #         armY.extend(real_elbow_Ys[j])
#                 #         armY.append(real_Ys[j])
#                 #
#                 #         armZ.extend(real_elbow_Zs[j])
#                 #         armZ.append(real_Zs[j])
#                 #
#                 #         ax.plot(armX, armY, armZ, marker='o', linestyle='-', color='blue', alpha=0.3)
#                 #         ax.scatter(armX, armY, armZ, marker='o', linestyle='-', color='blue', alpha=0.3)
#                 #     for j in range(len(pred_Xs)):
#                 #         armX.extend(pred_elbow_Xs[j])
#                 #         armX.append(pred_Xs[j])
#                 #
#                 #         armY.extend(pred_elbow_Ys[j])
#                 #         armY.append(pred_Ys[j])
#                 #
#                 #         armZ.extend(pred_elbow_Zs[j])
#                 #         armZ.append(pred_Zs[j])
#                 #
#                 #         ax.plot(armX, armY, armZ, marker='o', linestyle='-', color='orange', alpha=0.3)
#                 #         ax.scatter(armX, armY, armZ, marker='o', linestyle='-', color='orange', alpha=0.3)
#                 #
#                 # ax.plot(real_Xs, real_Ys, real_Zs, linestyle='--', color='blue', label='real')
#                 # ax.scatter(real_Xs, real_Ys, real_Zs, marker='o', color='blue', label='real')
#                 #
#                 # ax.plot(pred_Xs, pred_Ys, pred_Zs, linestyle='--', color='orange', label='real')
#                 # ax.scatter(pred_Xs, pred_Ys, pred_Zs, marker='o', color='orange', label='real')
#                 #
#                 # ax.scatter(env.target[0], env.target[1], env.target[2], c='r', marker='x')
#                 # ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='r', marker='o')
#                 #
#                 #
#                 # if show_model:
#                 #     ax.scatter(0,0,0, marker='v', c='g')
#                 #
#                 # plt.plot(real_Xs, real_Ys, real_Zs, marker='o', linestyle='--', label='real')
#                 # plt.plot(pred_Xs, pred_Ys, pred_Zs, marker='o', linestyle='--', label='pred')
#                 # ax.set_xlim(-sum(env.r), sum(env.r))
#                 # ax.set_ylim(-sum(env.r), sum(env.r))
#                 # ax.set_zlim(-sum(env.r), sum(env.r))
#                 # plt.savefig(datetime_str+'_'+str(i))
#                 # plt.close(fig)
#     return failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds

def find_next_move_train(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    # return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=False)
    return hill_climb(env.action_space.shape[0], env, env_learner, obs, max_action, episode_step, is_test=False, rand=False)

def find_next_move_test(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    # return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=True)
    inputs =  hill_climb(env.action_space.shape[0]*2, env, env_learner, obs, max_action, episode_step, is_test=True, rand=False)

    joints = env.action_space.shape[0]
    action = np.zeros(joints)
    for i in range(joints):
        amplitude = inputs[2*i]
        offset = inputs[2*i+1]
        action[i] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)
    return action

def evaluate(action, env_learner, obs, max_action, env, episode_step, test=True):
    # return eval_dist(action, env_learner, obs, max_action, env, episode_step, test=True)
    return eval_central_pattern_generator(action, env_learner, obs, max_action, env, episode_step, test=True)

def eval_central_pattern_generator(inputs, env_learner, obs, max_action, env, episode_step, test=True):
    joints = env.action_space.shape[0]
    action = np.zeros(joints)
    for i in range(joints):
        amplitude = inputs[2*i]
        offset = inputs[2*i+1]
        action[i] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)

    if not test: new_obs = env.step(max_action * action, save=False)[0]
    else: new_obs = env_learner.step(obs, max_action * action, episode_step, save=False)
    dz = new_obs[0]
    sin_yaw = new_obs[1]
    cos_yaw = new_obs[2]
    vx = new_obs[3]
    rew = vx
    # rew = new_obs[0] - new_obs[1] + new_obs[2] + new_obs[3]
    return rew

def eval_dist(action, env_learner, obs, max_action, env, episode_step, test=True):
    if not test: new_obs = env.step(max_action * action, save=False)[0]
    else: new_obs = env_learner.step(obs, max_action * action, episode_step, save=False)
    # d = np.linalg.norm(env.target - new_obs[-3:])
    dz = new_obs[0]
    sin_yaw = new_obs[1]
    cos_yaw = new_obs[2]
    vx = new_obs[3]
    rew = vx
    # rew = new_obs[0] - new_obs[1] + new_obs[2] + new_obs[3]
    return rew

# taken from wikipedia
def hill_climb(act_dim, env, env_learner, obs, max_action, episode_step, is_test, rand=False):
    epsilon = 0.001
    if rand:
        current_point = np.random.uniform(-1, 1, act_dim)
    else:
        current_point = np.zeros(act_dim) # the 0 magnitude vector is common
    step_size = 100*epsilon*np.ones(act_dim) # a vector of all 1's is common
    acc = 1.2 # a value of 1.2 is common
    candidate = np.array([-acc, -1.0/acc, 0.0, 1.0/acc, acc])
    while True:
        before = evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test)
        for i in range(act_dim):
            best = -1
            best_score = -10000
            for j in range(5):
                last_pt = current_point[i]
                current_point[i] = current_point[i] + step_size[i] * candidate[j]
                current_point[i] = max(current_point[i], -1)
                current_point[i] = min(current_point[i], 1)

                temp = evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test)
                current_point[i] = last_pt
                if temp > best_score:
                    best_score = temp
                    best = j
            if candidate[best] == 0:
                step_size[i] = step_size[i] / acc
            else:
                current_point[i] = current_point[i] + step_size[i] * candidate[best]
                step_size[i] = step_size[i] * candidate[best] # accelerate
        if (evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test) - before) < epsilon:
            return current_point

def find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1, is_test=False):
    min_act = np.zeros(env.action_space.shape[0])
    min_obs = env.step(max_action*min_act, save=False)[0]
    search_prec = 5
    max_depth = 10
    min_d = np.linalg.norm(env.target - min_obs[-3:])

    new_top = np.ones(min_act.size)*top
    new_bottom = np.ones(min_act.size)*bottom

    # Time Complexity = search_prec^(dof)
    # Assumes convexity in the action space, may not always be true

    # for i in range(search_prec+1):
    action = np.zeros(env.action_space.shape[0])
    min_act, min_d = __rec_next_move__(action, 0, search_prec, new_top, new_bottom, env, env_learner, max_action, dof, min_d, obs, episode_step, test=is_test)
    for i in range(max_depth): # max_depth search
        first_act = np.zeros(min_act.size)+min_act
        inc = ((new_top - new_bottom) / search_prec)
        new_top = np.minimum(np.ones(min_act.size)*first_act + inc, new_top)
        new_bottom = np.maximum(np.ones(min_act.size)*first_act - inc, new_bottom)
        action = np.zeros(env.action_space.shape[0])
        min_act, min_d = __rec_next_move__(action, 0, search_prec, new_top, new_bottom, env, env_learner, max_action, dof, min_d, obs, episode_step, test=is_test)
    return min_act

def __rec_next_move__(action, depth, search_prec, new_top, new_bottom, env, env_learner, max_action, dof, min_d, obs, episode_step, test):
    if depth == action.size:
        if not test: new_obs = env.step(max_action * action, save=False)[0]
        else: new_obs = env_learner.step(obs, max_action * action, episode_step, save=False)
        d = np.linalg.norm(env.target - new_obs[-3:])
        return action, d
    tmp_min_d = 1000000
    min_act = np.zeros(action.size)
    for i in range(search_prec + 1):
        action[depth] = new_bottom[depth] + i * ((new_top[depth] - new_bottom[depth]) / search_prec)
        action, new_min_d = __rec_next_move__(action, depth + 1, search_prec, new_top, new_bottom, env, env_learner, max_action, dof, min_d,
                          obs, episode_step, test)
        if new_min_d < tmp_min_d:
            min_act = action.copy()
            tmp_min_d = new_min_d
    # if depth == action.size-1:
    return min_act, tmp_min_d
    # else: return __rec_next_move__(action, depth+1, search_prec, new_top, new_bottom, env, env_learner, max_action, dof, min_d, obs, episode_step, test)

def test(env, epochs=100, train_episodes=10, test_episodes=100, loop='open', show_model=False, load=None):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    logger.info('Env Learner')
    env_learner = EnvLearner(env)
    logger.info('Done Env Learner')
    logger.info('Using agent with the following configuration:')
    try:
        saver = tf.train.Saver()
    except:
        saver=None
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    gpu_options = None
    num_cpu = 1
    if gpu_options is None:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
    else:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            gpu_options=gpu_options)

    episode_duration = -1
    nb_valid_episodes = 50
    episode_step = 0
    episode_reward = 0.0
    max_ep_rew = -10000
    train = []
    valid = []

    with tf.Session(config=tf_config) as sess:
        sess_start = time.time()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        data_log = open('logs/'+datetime_str+'_log.txt', 'w+')

        if load is not None:
            saver.restore(sess, load)
            logger.info('Model: '+load+' Restored')
            env_learner.initialize(sess, load=True)


        # generic data gathering
        obs = env.reset()
        # import random

        # print('Episode 0')
        # print(obs)
        amp_std = 0.1
        off_std = 0.1
        np.random.seed(0)
        # amplitude = random.normalvariate(0, amp_std)
        # offset = random.normalvariate(0, off_std)

        amplitude = 1.
        offset = 0.

        # Best found in 1000 episodes
        # amplitude = 1.3435346636198768
        # offset = 0.3321054365130438

        # Best found in 2000 episodes
        # amplitude = 1.4577248055402614
        # offset = 0.46320247355716504

        # Best found in 5000 episodes
        # amplitude = 1.6588451211617126
        # offset = 0.5850713467220024

        # Best found in 10000 episodes
        # amplitude = 1.6621371904667739
        # offset = 0.4552602035189509

        best_rew = 0.0
        best_amp = amplitude
        best_off = offset

        if load is None:
            env_learner.initialize(sess)
            # sess.graph.finalize()
            i = 0

            batch_size = 1

            while i < train_episodes/batch_size:
                avg_rew = 0.0
                avg_i = 0
                while avg_i < batch_size:

                    # Below needed for GCP
                    # joints = env.action_space.shape[0]
                    # action = np.zeros(joints)
                    # for j in range(joints):
                    #     action[j] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)

                    # Use uniform for training data generation
                    action = np.random.uniform(-1, 1, env.action_space.shape[0])


                    # action = np.zeros(env.action_space.shape[0])
                    # action = find_next_move(env, env_learner, obs, max_action, episode_step)
                    new_obs, r, done, info = env.step(max_action * action)
                    # env.render()
                    if episode_duration > 0:
                        done = (done or (episode_step >= episode_duration))


                    train.append([obs, max_action * action, r, new_obs, done, episode_step])
                    episode_step += 1
                    obs = new_obs
                    # print(obs)

                    vx = new_obs[3]
                    rew = vx
                    episode_reward += rew
                    if done:
                        # joints = env.action_space.shape[0]
                        # action = np.zeros(joints)
                        # for i in range(joints):
                        #     action[i] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)
                        # print(obs)
                        # print('Episode '+str(i))
                        avg_rew += episode_reward
                        episode_step = 0.0
                        obs = env.reset()
                        max_ep_rew = max(max_ep_rew, episode_reward)
                        episode_reward = 0.0
                        avg_i += 1
                i += 1
                avg_rew = float(avg_rew)/float(batch_size)
                if avg_rew > best_rew:
                    best_rew = avg_rew
                    best_amp = amplitude
                    best_off = offset
                amplitude = best_amp + np.random.normal(0, amp_std)
                offset = best_off + np.random.normal(0, off_std)

                # print('Train Batch: '+str(i)+': '+str(avg_rew)+' Max: '+str(best_rew))

            i = 0
            amplitude = best_amp
            offset = best_off
            avg_rew = 0.0
            episode_reward = 0.0
            obs = env.reset()
            while i < nb_valid_episodes:

                # Below needed for GCP
                # joints = env.action_space.shape[0]
                # action = np.zeros(joints)
                # for j in range(joints):
                #     action[j] = amplitude*math.sin(2*math.pi*0.0165*episode_step+offset)

                # Uniform for random validation
                action = np.random.uniform(-1, 1, env.action_space.shape[0])
                # action = find_next_move(env, env_learner, obs, max_action, episode_step)
                new_obs, r, done, info = env.step(max_action * action)
                if episode_duration > 0:
                    done = (done or (episode_step >= episode_duration))
                valid.append([obs, max_action * action, r, new_obs, done, episode_step])
                episode_step += 1
                obs = new_obs

                vx = new_obs[3]
                rew = vx
                episode_reward += rew
                if done:
                    avg_rew += episode_reward
                    # print('Valid Episode '+str(i)+': '+str(episode_reward))
                    episode_step = 0.0
                    obs = env.reset()
                    max_ep_rew = max(max_ep_rew, episode_reward)
                    episode_reward = 0.0
                    i += 1
                # if done:
                #     obs = env.reset()
                #     max_ep_rew = max(max_ep_rew, episode_reward)
                #     print('Valid Episode '+str(i)+': '+str(episode_reward))
                #     avg_rew += episode_reward
                #     episode_reward = 0.0
                #     i += 1

            avg_rew = avg_rew/float(nb_valid_episodes)
            # print('Valid Avg: '+str(avg_rew))
            # print('Best Amplitude: '+str(amplitude))
            # print('Best Offset: '+str(offset))
            logger.info('Data gathered')
            logger.info('Train Size: ' + str(len(train)))
            logger.info('Valid Size: ' + str(len(valid)))

            # return
            # Training self model
            env_learner.train(train, epochs, valid, logger, saver=saver, save_str=datetime_str)
            logger.info('Trained Self Model')
        # Testing in this env
        run_tests(test_episodes, env, env_learner, loop)

        # return
        # failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = run_tests(test_episodes, env, data_log, env_learner, max_action, loop)
        #
        # import statistics
        # num_bins = 10
        # print('\nModel: \'models/' + str(datetime_str) + '.ckpt\'')
        # print('Percent Failed: '+str(100.0*float(failures)/float(test_episodes))+'%')
        #
        # print('Mean Final Drift: '+str(statistics.mean(all_final_drifts)))
        # print('Median Final Drift: '+str(statistics.median(all_final_drifts)))
        # print('Stdev Final Drift: '+str(statistics.stdev(all_final_drifts)))
        #
        # print('Mean Episode Len: '+str(statistics.mean(all_final_lens)))
        # print('Median Episode Len: '+str(statistics.median(all_final_lens)))
        # print('Stdev Episode Len: '+str(statistics.stdev(all_final_lens)))
        #
        # print('Mean Final Pred D: '+str(statistics.mean(all_final_pred_ds)))
        # print('Median Final Pred D: '+str(statistics.median(all_final_pred_ds)))
        # print('Stdev Final Pred D: '+str(statistics.stdev(all_final_pred_ds)))
        #
        # print('Mean Final Real D: '+str(statistics.mean(all_final_real_ds)))
        # print('Median Final Real D: '+str(statistics.median(all_final_real_ds)))
        # print('Stdev Final Real D: '+str(statistics.stdev(all_final_real_ds)))
        #
        # print('\nCompleted In: '+str(time.time()-sess_start)+' s')
        # #
        # # _, _, _ = plt.hist(all_final_drifts, num_bins, facecolor='blue', alpha=0.5)
        # # plt.title('Final Drifts')
        # # plt.savefig(datetime_str+'_final_drift')
        # # plt.clf()
        # # _, _, _ = plt.hist(all_final_lens, num_bins, facecolor='blue', alpha=0.5)
        # # plt.title('Episode Lengths')
        # # plt.savefig(datetime_str+'_final_lens')
        # # plt.clf()
        # # _, _, _ = plt.hist(all_final_pred_ds, num_bins, facecolor='blue', alpha=0.5)
        # # plt.title('Final Predicted Distances')
        # # plt.savefig(datetime_str+'_final_pred_ds')
        # # plt.clf()
        # # _, _, _ = plt.hist(all_final_real_ds, num_bins, facecolor='blue', alpha=0.5)
        # # plt.title('Final Real Distances')
        # # plt.savefig(datetime_str+'_final_real_ds')
        # # plt.clf()


