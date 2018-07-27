import datetime
import math
import time

import numpy as np
import tensorflow as tf

from misc import logger


def run_tests(test_episodes, env, data_log, env_learner, max_action, loop):
    episode_step = 0
    last_d = env.d
    d = last_d
    acts = []
    all_final_drifts = []
    all_final_lens = []
    all_final_real_ds = []
    all_final_pred_ds = []
    test = []
    failures = 0
    for i in range(test_episodes):
        start_time = time.time()
        done = False
        obs = env.reset()
        init_d = np.linalg.norm(env.target - obs[-2:])
        data_log.write('Episode: ' + str(i) + ' Start\n')
        data_log.write('Target: ' + str(env.target) + '\n')
        data_log.write('Pred State: ' + str(obs) + '\n')
        data_log.write('Pred D: ' + str(init_d) + '\n')
        data_log.write('Real State: ' + str(obs) + '\n')
        data_log.write('Real D: ' + str(init_d) + '\n')
        data_log.write('Drift: ' + str(0.0) + '\n')

        real_Xs = []
        real_Ys = []
        pred_Xs = []
        pred_Ys = []

        # real_elbow_Xs = []
        # real_elbow_Ys = []
        # real_elbow_Zs = []
        # pred_elbow_Xs = []
        # pred_elbow_Ys = []

        pred_ds = []
        real_ds = []
        drifts = []
        real_Xs.append(obs[-2])
        real_Ys.append(obs[-1])
        pred_Xs.append(obs[-2])
        pred_Ys.append(obs[-1])

        # real_elbow_X = [0]
        # real_elbow_Y = [0]
        # pred_elbow_X = [0]
        # pred_elbow_Y = [0]
        # for j in range(len(env.r) - 1):
        #     real_elbow_X.append(float(env.r[j] * math.cos(obs[2 * j]) * math.sin(obs[2 * j + 1])))
        #     real_elbow_Y.append(float(env.r[j] * math.sin(obs[2 * j]) * math.sin(obs[2 * j + 1])))
        #     pred_elbow_X.append(float(env.r[j] * math.cos(obs[2 * j] * math.sin(obs[2 * j + 1]))))
        #     pred_elbow_Y.append(float(env.r[j] * math.sin(obs[2 * j]) * math.sin(obs[2 * j + 1])))
        # real_elbow_Xs.append(real_elbow_X)
        # real_elbow_Ys.append(real_elbow_Y)
        # pred_elbow_Xs.append(pred_elbow_X)
        # pred_elbow_Ys.append(pred_elbow_Y)

        pred_ds.append(init_d)
        real_ds.append(init_d)
        drifts.append(0)

        while not done:
            action = find_next_move_test(env, env_learner, obs, max_action, episode_step, dof=2)
            # action = find_next_move_train(env, env_learner, obs, max_action, episode_step, dof=4)
            new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
            real_obs, r, real_done, _ = env.step(max_action * action)

            d = np.linalg.norm(env.target - new_obs[-2:])
            real_d = np.linalg.norm(env.target - real_obs[-2:])
            test.append([obs, max_action * action, 0.0, new_obs, done, episode_step])
            acts.append(action)
            drift = np.linalg.norm(real_obs[-2:] - new_obs[-2:])
            episode_step += 1
            data_log.write('Action ' + str(episode_step) + ': ' + str(action) + '\n')
            data_log.write('Real Reward: ' + str(r) + '\n')
            data_log.write('Pred State: ' + str(new_obs) + '\n')
            data_log.write('Pred D: ' + str(d) + '\n')
            data_log.write('Real State: ' + str(real_obs) + '\n')
            data_log.write('Real D: ' + str(real_d) + '\n')
            data_log.write('Drift: ' + str(drift) + '\n')

            # print('Action '+str(episode_step)+': '+str(action)+'\n')
            # print('Pred D: '+str(d)+'\n')
            # print('Real D: '+str(real_d)+'\n')
            # print('Drift: '+str(drift)+'\n')

            real_Xs.append(real_obs[-2])
            real_Ys.append(real_obs[-1])
            pred_Xs.append(new_obs[-2])
            pred_Ys.append(new_obs[-1])

            # real_elbow_X = [0]
            # real_elbow_Y = [0]
            # real_elbow_Z = [0]
            # pred_elbow_X = [0]
            # pred_elbow_Y = [0]
            # for j in range(len(env.r) - 1):
            #     real_elbow_X.append(float(env.r[j] * math.cos(real_obs[2 * j]) * math.sin(real_obs[2 * j + 1])))
            #     real_elbow_Y.append(float(env.r[j] * math.sin(real_obs[2 * j]) * math.sin(real_obs[2 * j + 1])))
            #     real_elbow_Z.append(float(env.r[j] * math.cos(real_obs[2 * j + 1])))
            #     pred_elbow_X.append(float(env.r[j] * math.cos(new_obs[2 * j] * math.sin(new_obs[2 * j + 1]))))
            #     pred_elbow_Y.append(float(env.r[j] * math.sin(new_obs[2 * j]) * math.sin(new_obs[2 * j + 1])))
            # real_elbow_Xs.append(real_elbow_X)
            # real_elbow_Ys.append(real_elbow_Y)
            # real_elbow_Zs.append(real_elbow_Z)
            # pred_elbow_Xs.append(pred_elbow_X)
            # pred_elbow_Ys.append(pred_elbow_Y)

            pred_ds.append(d)
            real_ds.append(real_d)
            drifts.append(drift)

            if loop == 'open':
                obs = new_obs
            elif loop == 'closed':
                obs = real_obs
                # d = real_d
            else:
                obs = new_obs

            done = episode_step > env.max_iter

            if d < 0.01:
                done = True

            if done:
                data_log.write('Episode: ' + str(episode_step) + ' done\n\n')

                print('Episode: ' + str(i) + ' in ' + str(time.time() - start_time) + ' seconds')
                print(str(episode_step) + '\nPred D: ' + str(d) + '\nReal D: ' + str(real_d))
                print('Drift: ' + str(drift))
                if d < 0.01:
                    all_final_drifts.append(drift)
                    all_final_lens.append(episode_step)
                    all_final_pred_ds.append(d)
                    all_final_real_ds.append(real_d)
                else:
                    failures += 1

                episode_step = 0

                # Plotting
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                #
                #
                # if show_model:
                #     for j in range(len(real_Xs)):
                #         armX = []
                #         armY = []
                #         armZ = []
                #         armX.extend(real_elbow_Xs[j])
                #         armX.append(real_Xs[j])
                #
                #         armY.extend(real_elbow_Ys[j])
                #         armY.append(real_Ys[j])
                #
                #         armZ.extend(real_elbow_Zs[j])
                #         armZ.append(real_Zs[j])
                #
                #         ax.plot(armX, armY, armZ, marker='o', linestyle='-', color='blue', alpha=0.3)
                #         ax.scatter(armX, armY, armZ, marker='o', linestyle='-', color='blue', alpha=0.3)
                #     for j in range(len(pred_Xs)):
                #         armX.extend(pred_elbow_Xs[j])
                #         armX.append(pred_Xs[j])
                #
                #         armY.extend(pred_elbow_Ys[j])
                #         armY.append(pred_Ys[j])
                #
                #         armZ.extend(pred_elbow_Zs[j])
                #         armZ.append(pred_Zs[j])
                #
                #         ax.plot(armX, armY, armZ, marker='o', linestyle='-', color='orange', alpha=0.3)
                #         ax.scatter(armX, armY, armZ, marker='o', linestyle='-', color='orange', alpha=0.3)
                #
                # ax.plot(real_Xs, real_Ys, real_Zs, linestyle='--', color='blue', label='real')
                # ax.scatter(real_Xs, real_Ys, real_Zs, marker='o', color='blue', label='real')
                #
                # ax.plot(pred_Xs, pred_Ys, pred_Zs, linestyle='--', color='orange', label='real')
                # ax.scatter(pred_Xs, pred_Ys, pred_Zs, marker='o', color='orange', label='real')
                #
                # ax.scatter(env.target[0], env.target[1], env.target[2], c='r', marker='x')
                # ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='r', marker='o')
                #
                #
                # if show_model:
                #     ax.scatter(0,0,0, marker='v', c='g')
                #
                # plt.plot(real_Xs, real_Ys, real_Zs, marker='o', linestyle='--', label='real')
                # plt.plot(pred_Xs, pred_Ys, pred_Zs, marker='o', linestyle='--', label='pred')
                # ax.set_xlim(-sum(env.r), sum(env.r))
                # ax.set_ylim(-sum(env.r), sum(env.r))
                # ax.set_zlim(-sum(env.r), sum(env.r))
                # plt.savefig(datetime_str+'_'+str(i))
                # plt.close(fig)
    return failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds


def walk_with_model(env, self_model, loop):
    import matplotlib.pyplot as plt
    global fid
    fid = 0
    def draw_robot(obs, pos, color='b'):
        global fid
        x = [0]
        y = [0]
        last_deg = 0.0
        # print(obs)
        for i in range(2):
            x.append(obs[2+2*i])
            y.append(obs[2+2*i+1])
        # for j in range(env.r.size):
        #     x.append(x[-1] + float(env.r[j] * math.cos(last_deg + env.x[j])))
        #     y.append(y[-1] + float(env.r[j] * math.sin(last_deg + env.x[j])))
        #     last_deg += env.x[j]
        #     # masses.append(y)
        # plt.clf()
        x = [p + pos[0] for p in x]
        # y = [p + pos[1] for p in y]
        plt.plot(x, y, 'o', linestyle='-', color=color)
        if color == 'orange':
            minX = -2
            maxX = 2
            plt.axis([minX, maxX, 0.0, 2.0])
            plt.savefig('images/'+str(fid)+'.png')
            fid+=1


        # d = math.sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2)
        # print(d)

    plt.figure()
    # timer = fig.canvas.new_timer(interval=300)  # creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    plt.subplot(111, aspect='equal')
    minX = -2
    maxX = 2
    # plt.show()
    # plt.ion()
    obs = env.reset()
    self_obs = np.zeros_like(obs)+obs
    pos = np.zeros(2)
    self_pos = np.zeros(2)
    episode_step = 0
    # Inch forward
    draw_robot(obs, pos)
    draw_robot(obs, pos, color='orange')

    # Resetting
    x = self_obs[:2]
    y = self_obs[4:6]
    target = np.array([3 * math.pi / 4, math.pi / 4])
    while sum(np.abs(target - x)) > 0.1:
        action = target - x
        for i in range(len(action)):
            if np.abs(action[i]) > math.pi / 16: action[i] = np.sign(action[i]) * math.pi / 16

        if loop == 'open' or (loop.isdigit() and episode_step%int(loop) != 0):
            self_obs = self_model.step(self_obs, action, episode_step)
        elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) == 0):
            self_obs = self_model.step(obs, action, episode_step)


        obs, _, _, _ = env.step(action)
        episode_step += 1
        x = self_obs[:2]
        y = self_obs[4:6]
        pos -= obs[-2:]
        self_pos -= self_obs[-2:]
        plt.clf()
        draw_robot(obs, pos)
        draw_robot(self_obs, self_pos, color='orange')
        plt.axis([minX, maxX, 0.0, 2.0])
        # plt.pause(0.001)
        # plt.draw()
    i = 0
    while pos[0] < 2 and self_pos[0] < 2:
        y = self_obs[4:6]
        while y[1] > 0.01:
            # env.last_x = np.zeros(2)+env.x
            # env.x[0] += math.pi/64
            action = np.array([math.pi / 64, 0])
            if loop == 'open' or (loop.isdigit() and episode_step%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, episode_step)
            elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) == 0):
                self_obs = self_model.step(obs, action, episode_step)
            obs, _, _, _ = env.step(action)
            episode_step += 1
            x = self_obs[:2]
            y = self_obs[4:6]
            pos -= obs[-2:]
            self_pos -= self_obs[-2:]
            plt.clf()
            draw_robot(obs, pos)
            draw_robot(self_obs, self_pos, color='orange')
            plt.axis([minX, maxX, 0.0, 2.0])
            # plt.pause(0.001)
            # plt.draw()
        # print(y[1])
        # while y[0] > -1.99:
        # print(x)
        while x[0] < 63*math.pi/64:
            # env.last_x = np.zeros(2)+env.x
            # env.x[0] += math.pi/64
            # env.x[1] -= math.pi/32
            action = np.array([math.pi / 64, -math.pi / 32])
            if loop == 'open' or (loop.isdigit() and episode_step%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, episode_step)
            elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) == 0):
                self_obs = self_model.step(obs, action, episode_step)
            obs, _, _, _ = env.step(action)
            episode_step += 1
            x = self_obs[:2]
            y = self_obs[4:6]
            pos -= obs[-2:]
            self_pos -= self_obs[-2:]
            plt.clf()
            draw_robot(obs, pos)
            draw_robot(self_obs, self_pos, color='orange')
            plt.axis([minX, maxX, 0.0, 2.0])
            # plt.pause(0.001)
            # plt.draw()
            # print(x)
        print(pos)
        print(self_pos[0])
        print

        # target = np.array([7 * math.pi / 8, math.pi / 6])
        # while sum(np.abs(target - x)) > 0.1:
        #     action = target - x
        #     for i in range(len(action)):
        #         if np.abs(action[i]) > math.pi / 16: action[i] = np.sign(action[i]) * math.pi / 16
        #
        #     if loop == 'open' or (loop.isdigit() and episode_step % int(loop) == 0):
        #         self_obs = self_model.step(self_obs, action, episode_step)
        #     elif loop == 'closed' or (loop.isdigit() and episode_step % int(loop) != 0):
        #         self_obs = self_model.step(obs, action, episode_step)
        #
        #     obs, _, _, _ = env.step(action)
        #     episode_step += 1
        #     x = self_obs[:2]
        #     y = self_obs[4:6]
        #     pos -= obs[-2:]
        #     self_pos -= self_obs[-2:]
        #     plt.clf()
        #     draw_robot(obs, pos)
        #     draw_robot(self_obs, self_pos, color='orange')
        #     minX = -2
        #     maxX = 5
        #     plt.axis([minX, maxX, 0.0, 2.0])
        #     plt.pause(0.001)
        #     plt.draw()
        while x[0] > 7 * math.pi / 8:
            action = np.zeros(2)
            action[0] -= math.pi / 16
            if loop == 'open' or (loop.isdigit() and episode_step%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, episode_step)
            elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) == 0):
                self_obs = self_model.step(obs, action, episode_step)
            obs, _, _, _ = env.step(action)
            episode_step += 1
            x = self_obs[:2]
            y = self_obs[4:6]
            pos -= obs[-2:]
            self_pos -= self_obs[-2:]
            plt.clf()
            draw_robot(obs, pos)
            draw_robot(self_obs, self_pos, color='orange')
            plt.axis([minX, maxX, 0.0, 2.0])
            # plt.pause(0.001)
            # plt.draw()
        while x[1] < math.pi / 6:
            action = np.zeros(2)
            action[1] += math.pi / 16
            if loop == 'open' or (loop.isdigit() and episode_step%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, episode_step)
            elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) == 0):
                self_obs = self_model.step(obs, action, episode_step)
            obs, _, _, _ = env.step(action)
            episode_step += 1
            x = self_obs[:2]
            y = self_obs[4:6]
            pos -= obs[-2:]
            self_pos -= self_obs[-2:]
            plt.clf()
            draw_robot(obs, pos)
            draw_robot(self_obs, self_pos, color='orange')
            plt.axis([minX, maxX, 0.0, 2.0])
            # plt.pause(0.001)
            # plt.draw()
        # while x[0] > 7 * math.pi / 8 or x[1] < math.pi / 6:
        #     # env.last_x = np.zeros(2)+env.x
        #     action = np.zeros(2)
        #     if x[0] > 7 * math.pi / 8:
        #         # env.x[0] -= math.pi / 64
        #         action[0] -= math.pi / 64
        #     if x[1] < math.pi / 6:
        #         # env.x[1] += math.pi / 64
        #         action[1] += math.pi / 64
        #     if loop == 'open' or (loop.isdigit() and episode_step%int(loop) == 0):
        #         self_obs = self_model.step(self_obs, action, episode_step)
        #     elif loop == 'closed' or (loop.isdigit() and episode_step%int(loop) != 0):
        #         self_obs = self_model.step(obs, action, episode_step)
        #     obs, _, _, _ = env.step(action)
        #     episode_step += 1
        #     x = self_obs[:2]
        #     y = self_obs[4:6]
        #     pos -= obs[-2:]
        #     self_pos -= self_obs[-2:]
        #     plt.clf()
        #     draw_robot(obs, pos)
        #     draw_robot(self_obs, self_pos, color='orange')
        #     minX = -2
        #     maxX = 5
        #     plt.axis([minX, maxX, 0.0, 2.0])
        #     plt.pause(0.001)
        #     plt.draw()
        #     # print(x)
        i += 1
    print(i)


def test(env, env_learner, epochs=100, train_episodes=10, test_episodes=100, loop='open', show_model=False, load=None):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
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
            logger.info('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)


        # generic data gathering
        obs = env.reset()

        if load is None:
            env_learner.initialize(sess)
            # sess.graph.finalize()
            i = 0
            while i < train_episodes:
                action = np.random.uniform(-1, 1, env.action_space.shape[0])
                # action = find_next_move(env, env_learner, obs, max_action, episode_step)
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
                # action = find_next_move(env, env_learner, obs, max_action, episode_step)
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

            logger.info('Data gathered')
            logger.info('Train Size: ' + str(len(train)))
            logger.info('Valid Size: ' + str(len(valid)))


            # Training self model
            env_learner.train(train, epochs, valid, logger, saver=saver, save_str=datetime_str)
            logger.info('Trained Self Model')
        else:
            walk_with_model(env, env_learner, loop)



        # return
        # Testing in this env

        # failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = run_tests(test_episodes, env, data_log, env_learner, max_action, loop)

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

