import pickle
import numpy as np
from envs.widowx_arm import WidowxROS
from env_learners.dnn_env_learner import DNNEnvLearner
import tensorflow as tf


def read_until(file, char):
    ret_str = ''
    cursor = file.read(1)
    while cursor != char:
        ret_str += cursor
        cursor = file.read(1)
    return ret_str

def get_angles_from_log(log_path):
    from os import listdir
    from os.path import isfile, join
    # files = [f for f in listdir(log_path) if isfile(join(log_path, f)) and '_log.txt' in f]

    log_file = open(log_path)
    start_at = 0
    nb_episodes = 100
    max_steps = 100

    start = log_file.readline()
    idx = int(start.split()[1])

    all_real_angles = []
    all_pred_angles = []

    start_states = []
    actions = []

    while idx < nb_episodes:
        print('Episode '+str(idx))
        episode_pred_states = []
        episode_real_states = []

        read_until(log_file, '[')
        target = np.array(read_until(log_file, ']').split(), dtype='float64')
        read_until(log_file, '[')

        pred_state = np.array(read_until(log_file, ']').split(), dtype='float64')
        log_file.read(1)
        pred_d = float(log_file.readline().split()[2])

        read_until(log_file, '[')
        real_state = np.array(read_until(log_file, ']').split(), dtype='float64')
        log_file.read(1)
        real_d = float(log_file.readline().split()[2])
        drift = float(log_file.readline().split()[1])
        act_idx = 0

        episode_pred_states.append(pred_state)
        episode_real_states.append(real_state)

        start_state = real_state
        episode_actions = []

        while True:
            test_char = log_file.read(1)
            if test_char == 'E':
                test_str = test_char+log_file.readline()
                if test_str[1] == 'p':
                    break
                else:
                    print('Error Reading Log')
                    exit()

            # Get Action
            action_str = read_until(log_file, ']').split()
            act_idx = int(action_str[1][:-1])
            action_str[2] = action_str[2][1:]
            action = np.array(action_str[2:])
            read_until(log_file, ':')
            log_file.read(1)
            reward = float(read_until(log_file, 'P'))
            read_until(log_file, '[')

            # Get Pred State
            pred_state = np.array(read_until(log_file, ']').split(), dtype='float64')

            # Pred D
            read_until(log_file, ':')
            pred_d = float(log_file.readline().strip())

            read_until(log_file, '[')
            # Get Real State
            real_state = np.array(read_until(log_file, ']').split(), dtype='float64')

            # Real D
            read_until(log_file, ':')
            real_d = float(log_file.readline().strip())
            drift = float(log_file.readline().split()[1])


            episode_pred_states.append(pred_state)
            episode_real_states.append(real_state)

            episode_actions.append(action)

        # if idx >= start_at:
        #     make_gif(episode_pred_states, episode_real_states, target=target, name='episode_'+str(idx), path=log_path, successes=True)
        #     make_gif(episode_pred_states, episode_real_states, target=target, name='episode_'+str(idx), path=log_path, successes=False)
        if len(episode_pred_states) <= 100:
            # all_pred_angles.append(episode_pred_states)
            all_real_angles.append(episode_real_states)
            start_states.append(start_state)
            actions.append(episode_actions)


        log_file.readline()
        start = log_file.readline()
        try:
            idx = int(start.split()[1])
        except:
            print
            print('Finished '+log_path)
            return start_states, actions, all_real_angles
    return start_states, actions, all_real_angles

if __name__ == '__main__':
    loaders = [
        'models/2018-08-05-17:49:46.ckpt', # 1K DNN Seq (5)
        'models/2018-08-05-18:10:52.ckpt', # 10K DNN Seq (5)
        'models/2018-08-05-18:42:42.ckpt', # 20K DNN Seq (5)
        'models/2018-08-05-19:39:51.ckpt', # 30K DNN Seq (5)
        'models/2018-08-05-20:51:28.ckpt', # 40K DNN Seq (5)
        'models/2018-08-05-23:35:08.ckpt', # 50K DNN Seq (5)
        'models/2018-08-06-01:24:49.ckpt', # 60K DNN Seq (5)
        'models/2018-08-06-03:29:30.ckpt', # 70K DNN Seq (5)
        'models/2018-08-06-05:52:05.ckpt', # 80K DNN Seq (5)
        'models/2018-08-06-08:36:51.ckpt', # 90K DNN Seq (5)
        'models/2018-08-05-12:11:16.ckpt' # 100K DNN Seq (5)
    ]

    file_names = [
        '1K_open_dnn5.pkl',
        '10K_open_dnn5.pkl',
        '20K_open_dnn5.pkl',
        '30K_open_dnn5.pkl',
        '40K_open_dnn5.pkl',
        '50K_open_dnn5.pkl',
        '60K_open_dnn5.pkl',
        '70K_open_dnn5.pkl',
        '80K_open_dnn5.pkl',
        '90K_open_dnn5.pkl',
        '100K_open_dnn5.pkl'
    ]
    start_states, actions, all_real_angles = get_angles_from_log('/home/robert/Research/Self-Modeling/logs/2018-08-01-00:07:08_log.txt')
    # all_real_angles, all_pred_angles = get_angles_from_log('/home/robert/Research/Self-Modeling/logs/2018-08-05-12:11:16_log.txt')


    for n in range(len(all_real_angles)):
        angles_list = all_real_angles[n]
        # Linear Interpolation along angles for smoother/slower traveling (less wear on motors)
        int_len = 10
        for i in range(len(angles_list) - 1):
            diff = angles_list[i * int_len + 1] - angles_list[i * int_len]
            for j in range(1, int_len):
                val = angles_list[i * int_len] + float(j) * diff / float(int_len)
                angles_list.insert(i * int_len + j, val)
    env = WidowxROS()
    max_action = env.action_space.high

    env_learner = DNNEnvLearner(env)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    try:
        saver = tf.train.Saver()
    except:
        saver=None
    with tf.Session(config=tf_config) as sess:

        for l in range(len(loaders)):
            load = loaders[l]
            file_name = file_names[l]
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)

            print('')

            logged_drifts = []
            fk_drifts = []
            internal_drifts = []

            all_observations = []
            drifts = []
            for n in range(len(all_real_angles)):
                angles_list = all_real_angles[n]
                episode_observations = []
                if len(angles_list) < 120:
                    obs = start_states[n]
                    episode_step = 0
                    episode_observations.append(obs)
                    for action_str in actions[n]:
                        if action_str[0] == '':
                            action_str = action_str[1:]
                        # print(episode_step)
                        action = np.array([float(action_str[k]) for k in range(len(action_str))])
                        new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
                        episode_step += 1
                        obs = np.zeros_like(new_obs)+new_obs
                        episode_observations.append(new_obs)
                    drift = np.linalg.norm(all_real_angles[n][-1][-3:] - new_obs[-3:])
                    drifts.append(drift)
                all_observations.append(episode_observations)

            # sanity check
            assert len(all_observations) == len(all_real_angles)

            import statistics
            print('Median Drift Recorded On This Dataset: '+str(statistics.median(drifts)))
            pickle.dump(all_observations, open(file_name, 'wb+'))
            print('Dumped to ' + str(file_name))

