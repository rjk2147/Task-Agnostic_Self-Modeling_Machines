import time, datetime
from collections import deque
import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env_learner import EnvLearner

def train_env_learner(env_learner, train, total_steps, valid=None, logger=None, log_interval=10, early_stopping=-1,
                      saver=None, save_str=None):
    min_loss = 10000000000
    stop_count = 0
    for i in range(total_steps):
        if i > 0 and i%(total_steps/env_learner.max_seq_len) == 0 and env_learner.seq_len < env_learner.max_seq_len:
            env_learner.seq_len += 1
            print('Sequence Length: '+str(env_learner.seq_len))

        if i % log_interval == 0 and logger is not None and valid is not None:
            (vGen, vDisc, vC) = env_learner.get_loss(valid)
            logger.info('Epoch: ' + str(i) + '/' + str(total_steps))
            logger.info('Valid Loss')
            logger.info('Gen:  '+str(vGen))
            logger.info('Disc: '+str(vDisc))
            logger.info('Close: '+str(vC))
            logger.info()
            if saver is not None and save_str is not None:
                save_path = saver.save(env_learner.sess, 'models/' + str(save_str) + '.ckpt')
                logger.info("Model saved in path: %s" % save_path)
        start = time.time()
        tlGen, tlDisc = env_learner.train_adv(train)
        duration = time.time() - start
        if tlGen < min_loss:
            min_loss = tlGen
            stop_count = 0
        else:
            stop_count += 1
        if stop_count > early_stopping and early_stopping > 0:
            break
        if i % log_interval != 0 and logger is not None:
            logger.info('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
            logger.info('Train Loss')
            logger.info('Gen:  '+str(tlGen))
            logger.info('Disc: '+str(tlDisc))
            logger.info()
    if logger is not None and valid is not None:
        (vGen, vDisc, vC) = env_learner.get_loss(valid)
        logger.info('Final Epoch: ')
        logger.info('Valid Loss')
        logger.info('Gen:  '+str(vGen))
        logger.info('Disc: '+str(vDisc))
        logger.info('Close: '+str(vC))
        logger.info()
    if saver is not None and save_str is not None:
        save_path = saver.save(env_learner.sess, 'models/' + str(save_str) + '.ckpt')
        logger.info("Final Model saved in path: %s" % save_path)

def find_next_move_train(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=False)

def find_next_move_test(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=True)

def find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1, is_test=False):
    min_act = np.zeros(env.action_space.shape[0])
    min_obs = env.step(max_action*min_act, save=False)[0]
    search_prec = 5
    max_depth = 10
    min_d = np.linalg.norm(env.target - min_obs[dof:])

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
        d = np.linalg.norm(env.target - new_obs[dof:])
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

def test(env, epochs=100, train_episodes=10, test_episodes=100):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    logger.info('Env Learner')
    env_learner = EnvLearner(env)
    logger.info('Done Env Learner')
    logger.info('Using agent with the following configuration:')

    saver = tf.train.Saver()

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
    test = []
    with tf.Session(config=tf_config) as sess:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        env_learner.initialize(sess)
        sess.graph.finalize()

        # generic data gathering
        obs = env.reset()
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
        train_env_learner(env_learner, train, epochs, valid, logger, saver=saver, save_str=datetime_str)
        logger.info('Trained Self Model')


        # Testing in this env
        episode_step = 0
        last_d = env.d
        d = last_d
        acts = []
        all_final_drifts = []
        all_final_lens = []
        all_final_real_ds = []
        all_final_pred_ds = []

        for i in range(test_episodes):
            start_time = time.time()
            done = False
            obs = env.reset()
            start_pos = [obs[4], obs[5], obs[6]]

            real_Xs = []
            real_Ys = []
            real_Zs = []
            pred_Xs = []
            pred_Ys = []
            pred_Zs = []
            pred_ds = []
            real_ds = []
            drifts = []
            init_d = np.linalg.norm(env.target - obs[4:])
            real_Xs.append(obs[4])
            real_Ys.append(obs[5])
            real_Zs.append(obs[6])
            pred_Xs.append(obs[4])
            pred_Ys.append(obs[5])
            pred_Zs.append(obs[6])
            pred_ds.append(init_d)
            real_ds.append(init_d)
            drifts.append(0)

            while not done:
                action = find_next_move_test(env, env_learner, obs, max_action, episode_step, dof=4)
                new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
                real_obs, r, real_done, _ = env.step(max_action * action)
                d = np.linalg.norm(env.target-new_obs[4:])
                real_d = np.linalg.norm(env.target-real_obs[4:])
                test.append([obs, max_action * action, 0.0, new_obs, done, episode_step])
                acts.append(action)
                drift = np.linalg.norm(real_obs[4:]-new_obs[4:])
                episode_step += 1

                real_Xs.append(real_obs[4])
                real_Ys.append(real_obs[5])
                real_Zs.append(real_obs[6])
                pred_Xs.append(new_obs[4])
                pred_Ys.append(new_obs[5])
                pred_Zs.append(new_obs[6])
                pred_ds.append(d)
                real_ds.append(real_d)
                drifts.append(drift)

                # Reverse the commenting of the next 2 lines to change from open loop to closed loop
                obs = new_obs
                # obs = real_obs


                episode_step += 1
                done = episode_step > env.max_iter

                if d < 0.01:
                    done = True

                if done:
                    print('Episode: '+str(i)+' in '+str(time.time() - start_time)+' seconds')
                    print(str(episode_step)+'\nPred D: '+str(d)+'\nReal D: '+str(real_d))
                    print('Drift: '+str(drift))
                    all_final_drifts.append(drift)
                    all_final_lens.append(episode_step)
                    all_final_pred_ds.append(d)
                    all_final_real_ds.append(real_d)

                    episode_step = 0

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    ax.plot(real_Xs, real_Ys, real_Zs)
                    ax.scatter(real_Xs, real_Ys, real_Zs)

                    ax.plot(pred_Xs, pred_Ys, pred_Zs)
                    ax.scatter(pred_Xs, pred_Ys, pred_Zs)

                    ax.scatter(env.target[0], env.target[1], env.target[2], c='r', marker='x')
                    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='r', marker='o')
                    plt.plot(real_Xs, real_Ys, real_Zs, marker='o', linestyle='--', label='real')
                    plt.plot(pred_Xs, pred_Ys, pred_Zs, marker='o', linestyle='--', label='pred')
                    ax.set_xlim(-2, 2)
                    ax.set_ylim(-2, 2)
                    ax.set_zlim(-2, 2)
                    plt.savefig(datetime_str+'_'+str(i))
                    plt.close(fig)

        import statistics
        num_bins = 10
        print('Mean Final Drift: '+str(statistics.mean(all_final_drifts)))
        print('Median Final Drift: '+str(statistics.median(all_final_drifts)))
        print('Stdev Final Drift: '+str(statistics.stdev(all_final_drifts)))
        _, _, _ = plt.hist(all_final_drifts, num_bins, facecolor='blue', alpha=0.5)
        plt.title('Final Drifts')
        plt.savefig(datetime_str+'_final_drift')
        plt.clf()

        print('Mean Episode Len: '+str(statistics.mean(all_final_lens)))
        print('Median Episode Len: '+str(statistics.median(all_final_lens)))
        print('Stdev Episode Len: '+str(statistics.stdev(all_final_lens)))
        _, _, _ = plt.hist(all_final_lens, num_bins, facecolor='blue', alpha=0.5)
        plt.title('Episode Lengths')
        plt.savefig(datetime_str+'_final_lens')
        plt.clf()

        print('Mean Final Pred D: '+str(statistics.mean(all_final_pred_ds)))
        print('Median Final Pred D: '+str(statistics.median(all_final_pred_ds)))
        print('Stdev Final Pred D: '+str(statistics.stdev(all_final_pred_ds)))
        _, _, _ = plt.hist(all_final_pred_ds, num_bins, facecolor='blue', alpha=0.5)
        plt.title('Final Predicted Distances')
        plt.savefig(datetime_str+'_final_pred_ds')
        plt.clf()

        print('Mean Final Real D: '+str(statistics.mean(all_final_real_ds)))
        print('Median Final Real D: '+str(statistics.median(all_final_real_ds)))
        print('Stdev Final Real D: '+str(statistics.stdev(all_final_real_ds)))
        _, _, _ = plt.hist(all_final_real_ds, num_bins, facecolor='blue', alpha=0.5)
        plt.title('Final Real Distances')
        plt.savefig(datetime_str+'_final_real_ds')
        plt.clf()


