import tensorflow as tf
import numpy as np
import time, datetime, pickle
from misc import losses
from misc.models import fk_learner
from test_planners.control_widowx import calc_end_effector_pos
from env_learners.dnn_env_learner import DNNEnvLearner
from envs.widowx_arm import WidowxROS
from test_planners.test_plan_widowx import run_tests

if __name__ == '__main__':
    train_episodes = 10

    data_file = 'real_widowx_train_10hz_100K_default_processed.pkl'
    train_data = pickle.load(open(data_file, 'rb+'))
    train = train_data[:(train_episodes*100)]
    env = WidowxROS()
    max_action = env.action_space.high
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

    epochs = 100
    valid = None
    nb_models = 2
    test_episodes = 100
    loop = 'open'

    scope_strs = []
    datetime_strs = []
    graphs = []
    env_learners = []
    for i in range(nb_models):
        scope_strs.append('model'+str(i))
        datetime_strs.append(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

    # print('Traditional Deep Neural Network architecture chosen')

    with tf.Session(config=tf_config) as sess:
        print('Initial Model training...')
        for i in range(nb_models):
            with tf.variable_scope(scope_strs[i]):
                env_learner = DNNEnvLearner(env)
                env_learner.initialize(sess)
                env_learners.append(env_learner)
        for i in range(nb_models):
            with tf.variable_scope(scope_strs[i]):
                saver = tf.train.Saver()
                env_learners[i].train(train, epochs, valid, saver=saver, save_str=datetime_strs[i], verbose=False)
                print('Model '+str(i)+' trained')
                # print("Final Model saved in path: %s" % datetime_strs[i])

        print('Testing for each model..')
        print('')
        model_perc = []
        model_medians = []
        model_means = []
        model_stds = []
        for i in range(nb_models):
            data_log = open('logs/'+datetime_strs[i]+'_log.txt', 'w+')
            failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = \
                run_tests(test_episodes, env, data_log, env_learners[i], max_action, loop, verbose=False)
            model_perc.append(100.0 * float(failures) / float(test_episodes))
            model_means.append(np.mean(all_final_drifts))
            model_medians.append(np.median(all_final_drifts))
            model_stds.append(np.std(all_final_drifts))

        print('')
        print('Trained On '+str(1000)+' Datapoints')
        index = np.argmin(model_medians)

        print('Best Model '+str(index)+':')
        print('Percent Failed: ' + str(model_perc[index]) + '%')
        print('Mean Final Drift: '+str(model_medians[index]))
        print('Median Final Drift: '+str(model_means[index]))
        print('Stdev Final Drift: '+str(model_stds[index]))
        print('')

        index = np.argsort(model_medians)[len(model_medians)//2]

        print('Median Model '+str(index)+':')
        print('Percent Failed: ' + str(model_perc[index]) + '%')
        print('Mean Final Drift: '+str(model_medians[index]))
        print('Median Final Drift: '+str(model_means[index]))
        print('Stdev Final Drift: '+str(model_stds[index]))
        print('')

        print('Collecting difference maximizing data...')
        for d in range(9):
            new_train = []
            for n in range(10):
                obs = env.reset()
                episode = []
                for episode_step in range(100):
                    avg_drifts = []
                    all_drifts = []
                    actions = []
                    start = time.time()
                    for i in range(1000):
                        action = np.random.uniform(-1, 1, 4)
                        obs_chart = []
                        for j in range(nb_models):
                            with tf.variable_scope(scope_strs[j]):
                                new_obs = env_learners[j].step(obs, max_action * action, episode_step, save=False)
                                obs_chart.append(new_obs)
                        drift_chart = np.zeros((nb_models, nb_models))
                        drifts = []
                        for j in range(nb_models):
                            for k in range(j+1, nb_models):
                                drift = np.linalg.norm(obs_chart[j][-3:]-obs_chart[k][-3:])
                                drift_chart[j][k] = drift
                                drifts.append(drift)
                        avg_drifts.append(np.mean(drifts))
                        all_drifts.append(drift_chart)
                        actions.append(action)
                    index = np.argmax(avg_drifts)
                    duration = time.time()-start

                    action = actions[index]
                    real_obs, r, real_done, _ = env.step(max_action * action)
                    episode.append([obs, action * max_action, r, real_obs, real_done, episode_step])
                    obs = np.zeros_like(real_obs)+real_obs

                    # print('Median Average Drift: '+str(np.median(avg_drifts)))
                    # print('Action: '+str(actions[index])+' Chosen with avg drift: '+str(avg_drifts[index])+' in '+str(duration)+'s')
                print('Episode '+str(n)+' complete')
                new_train.extend(episode)
            train.extend(new_train)
            print('Training on newly collected data...')
            for i in range(nb_models):
                with tf.variable_scope(scope_strs[i]):
                    saver = tf.train.Saver()
                    env_learners[i].train(new_train, epochs, valid, saver=saver, save_str=datetime_strs[i], verbose=False)
                    print('Model '+str(i)+' complete')


            print('Testing for each model..')
            print('')
            model_perc = []
            model_medians = []
            model_means = []
            model_stds = []
            for i in range(nb_models):
                data_log = open('logs/'+datetime_strs[i]+'_log.txt', 'w+')
                failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = \
                    run_tests(test_episodes, env, data_log, env_learners[i], max_action, loop, verbose=False)
                model_perc.append(100.0 * float(failures) / float(test_episodes))
                model_means.append(np.mean(all_final_drifts))
                model_medians.append(np.median(all_final_drifts))
                model_stds.append(np.std(all_final_drifts))

            print('')
            print('Trained On '+str((d+2)*1000)+' Datapoints')
            index = np.argmin(model_medians)

            print('Best Model '+str(index)+':')
            print('Percent Failed: ' + str(model_perc[index]) + '%')
            print('Mean Final Drift: '+str(model_medians[index]))
            print('Median Final Drift: '+str(model_means[index]))
            print('Stdev Final Drift: '+str(model_stds[index]))
            print('')

            index = np.argsort(model_medians)[len(model_medians)//2]

            print('Median Model '+str(index)+':')
            print('Percent Failed: ' + str(model_perc[index]) + '%')
            print('Mean Final Drift: '+str(model_medians[index]))
            print('Median Final Drift: '+str(model_means[index]))
            print('Stdev Final Drift: '+str(model_stds[index]))
            print('')

