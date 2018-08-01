import datetime
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from env_learners.env_learner import EnvLearner


def walk_with_model(env, self_model, loop):
    def draw_compare(real_pos, pred_pos, env):
        env.render()
        # plt.clf()
        minX = -0.5
        maxX = 0.5
        x = [real_pos[1],real_pos[4],real_pos[7]]
        y = [real_pos[2],real_pos[5],real_pos[8]]
        plt.plot(x, y, 'o', linestyle='-', color='blue')
        x = [pred_pos[1],pred_pos[4],pred_pos[7]]
        y = [pred_pos[2],pred_pos[5],pred_pos[8]]
        plt.plot(x, y, 'o', linestyle='-', color='orange')
        plt.axis([minX, maxX, 0.0, 3*(maxX-minX)/4])
        plt.pause(0.0001)
        plt.draw()

    obs = env.reset()
    self_obs = obs
    done = False
    real_pos = np.zeros(9)+env.pos
    pred_pos = np.zeros(9)+env.pos
    while not done:
        # print(env.muscles[1].l0)
        # print(env.masses[-2].p)
        print('Step 1')
        while env.muscles[1].l0 < 0.4:
            action = np.array([0.,0.01])
            obs, _, done, _ = env.step(action)

            if loop == 'open' or (loop.isdigit() and env.i%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, env.i)
            elif loop == 'closed' or (loop.isdigit() and env.i%int(loop) == 0):
                self_obs = self_model.step(obs, action, env.i)
            real_pos += obs
            pred_pos += self_obs
            draw_compare(real_pos, pred_pos, env)
            # env.render()
        # print(env.masses[-2].p)
        print(real_pos[:3])
        print(pred_pos[:3])
        print('Step 2')
        for i in range(1):
            action = np.array([-0.01,0.])
            obs, _, done, _ = env.step(action)
            if loop == 'open' or (loop.isdigit() and env.i%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, env.i)
            elif loop == 'closed' or (loop.isdigit() and env.i%int(loop) == 0):
                self_obs = self_model.step(obs, action, env.i)
            real_pos += obs
            pred_pos += self_obs
            draw_compare(real_pos, pred_pos, env)
            # env.render()
        print(real_pos[:3])
        print(pred_pos[:3])
        print('Step 3')
        while env.muscles[1].l0 > 0.2449:
            action = np.array([0.,0.])
            action[1] -= 0.01
            obs, _, done, _ = env.step(action)
            if loop == 'open' or (loop.isdigit() and env.i%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, env.i)
            elif loop == 'closed' or (loop.isdigit() and env.i%int(loop) == 0):
                self_obs = self_model.step(obs, action, env.i)
            real_pos += obs
            pred_pos += self_obs
            draw_compare(real_pos, pred_pos, env)
            # env.render()
        print(real_pos[:3])
        print(pred_pos[:3])
        print('Step 4')
        for i in range(1):
            action = np.array([0.01,0.])
            obs, _, done, _ = env.step(action)
            if loop == 'open' or (loop.isdigit() and env.i%int(loop) != 0):
                self_obs = self_model.step(self_obs, action, env.i)
            elif loop == 'closed' or (loop.isdigit() and env.i%int(loop) == 0):
                self_obs = self_model.step(obs, action, env.i)
            real_pos += obs
            pred_pos += self_obs
            draw_compare(real_pos, pred_pos, env)
            # env.render()
        print(real_pos[:3])
        print(pred_pos[:3])
        # print(env.i)


def test(env, epochs=100, train_episodes=10, test_episodes=100, loop='open', show_model=False, load=None):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    print('Env Learner')
    env_learner = EnvLearner(env)
    print('Done Env Learner')
    print('Using agent with the following configuration:')
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
    nb_valid_episodes = 10
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
            print('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)


        # generic data gathering
        obs = env.reset()
        import sys
        if load is None:
            train_max = 0
            train_min = 0
            env_learner.initialize(sess)
            # sess.graph.finalize()
            i = 0
            while i < train_episodes:
                action = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape[0])
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
                    sys.stdout.write(str(i)+'/'+str(train_episodes)+' Training Episodes Generated                   \r')
                    # print(i)
            sys.stdout.write('Training Data Generated                          \r\n')
            print('Training Data Max: '+str( max([np.max(episode[0]) for episode in train])))
            print('Training Data Min: '+str( min([np.min(episode[0]) for episode in train])))
            i = 0
            while i < nb_valid_episodes:
                action = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape[0])
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
                    sys.stdout.write(str(i)+'/'+str(nb_valid_episodes)+' Validation Episodes Generated\r')
                    i += 1
            sys.stdout.write('Validation Data Generated\r\n')

            print('All Data gathered')
            print('Training set Size: ' + str(len(train)))
            print('Validation set Size: ' + str(len(valid)))


            # Training self model
            env_learner.train(train, epochs, valid, saver=saver, save_str=datetime_str)
            print('Trained Self Model')
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

