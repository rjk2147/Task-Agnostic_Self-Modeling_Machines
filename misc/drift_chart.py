import math, time
import numpy as np
import matplotlib.pyplot as plt


def read_until(file, char):
    ret_str = ''
    cursor = file.read(1)
    while cursor != char:
        ret_str += cursor
        cursor = file.read(1)
    return ret_str

def draw_cone(ax, h):
    # Make data
    u = np.linspace(0, 2 * np.pi, 1000)
    r = np.linspace(0, 0.5, 10)
    x = np.outer(np.cos(u), r)
    y = np.outer(np.sin(u), r)
    z = np.outer(np.ones(np.size(u)), np.linspace(0, -h, 10))

    # Plot the surface
    ax.plot_surface(x, y, z, color='yellow', linewidth=0, alpha=0.5)
def make_drift_chart_from_path(log_path):
    from os import listdir
    from os.path import isfile, join
    # files = [f for f in listdir(log_path) if isfile(join(log_path, f)) and '_log.txt' in f]
    #os.system('rm -rf '+log_path+'/successes')
    #os.system('rm -rf '+log_path+'/failures')

    log_file = open(log_path)
    start_at = 0
    nb_episodes = 100
    max_steps = 100

    start = log_file.readline()
    idx = int(start.split()[1])
    all_drifts = []
    for i in range(101): all_drifts.append([])
    while True:
        print('Episode '+str(idx))
        episode_pred_states = []
        episode_real_states = []

        drifts = []

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
            drifts.append(drift)
            # drifts[act_idx].append(drift)

            episode_pred_states.append(pred_state)
            episode_real_states.append(real_state)

        # drifts.append(drift)
        # > 100 is only failures
        # < 100 is only successes
        # < 32 is all it should need to find any goal excluding collision
        if len(drifts) > 100:
        # if True:
        # if act_idx < 100:
            for i in range(len(drifts)):
                all_drifts[i].append(drifts[i])

        log_file.readline()
        start = log_file.readline()
        try:
            idx = int(start.split()[1])
        except:
            print
            # plt.hist(100*np.array(all_drifts[0]), 100)
            # plt.title('Distribution of Drifts')
            # plt.xlabel('Drift in cm')
            # plt.ylabel('Frequency')
            import statistics
            all_drifts_medians = []
            all_drifts_low = []
            all_drifts_high = []
            all_drifts_stdevs = []
            x = []
            for i in range(len(all_drifts)):
                if len(all_drifts[i]) > 1:
                    all_drifts[i] = 100 * np.array(all_drifts[i])
                    all_drifts_medians.append(statistics.median(all_drifts[i]))
                    all_drifts_stdevs.append(statistics.stdev(all_drifts[i]))
                    all_drifts_low.append(np.percentile(all_drifts[i], 25))
                    all_drifts_high.append(np.percentile(all_drifts[i], 75))
                    x.append(i)

            all_drifts_medians = np.array(all_drifts_medians)
            # error = 0.67449*np.array(all_drifts_stdevs)
            print('Finished '+log_path)
            return all_drifts_medians, all_drifts_low, all_drifts_high



plt.title('Progression of Drift with Time on Failure Cases')
plt.ylabel('Drift in cm')
plt.xlabel('Timesteps')
# No Seq
all_drifts_medians, low, high = make_drift_chart_from_path('/home/robert/Research/Self-Modeling/logs/2018-08-06-13:13:58_log.txt')
x = np.arange(len(low))
plt.plot(x, all_drifts_medians)
plt.fill_between(x, low, high, alpha=0.5)

# Inc Seq (5) 4000
all_drifts_medians, low, high = make_drift_chart_from_path('/home/robert/Research/Self-Modeling/logs/2018-08-05-02:38:15_log.txt')
x = np.arange(len(low))
plt.plot(x, all_drifts_medians)
plt.fill_between(x, low, high, alpha=0.5)

# # Always Seq (5)
all_drifts_medians, low, high = make_drift_chart_from_path('/home/robert/Research/Self-Modeling/logs/2018-08-05-23:38:09_log.txt')
# print(low[0])
# print(high[0])
x = np.arange(len(low))
plt.plot(x, all_drifts_medians)
plt.fill_between(x, low, high, alpha=0.5)

# # Always Seq (10)
# all_drifts_medians, low, high = make_drift_chart_from_path('/home/robert/Research/Self-Modeling/logs/2018-08-06-13:20:28_log.txt')
# x = np.arange(len(low))
# plt.plot(x, all_drifts_medians)
# plt.fill_between(x, low, high, alpha=0.5)

# plt.hist(all_drifts_medians[0], 10)

plt.show()

# # DNN Seq Inc
# log_files = [
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-15:14:04_log.txt', # 10K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-16:37:05_log.txt', # 20K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-16:51:25_log.txt', # 30K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-17:23:32_log.txt', # 40K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-18:03:39_log.txt', # 50K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-21:48:56_log.txt', # 60K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-18:46:40_log.txt', # 70K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-19:40:16_log.txt', # 80K
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-20:38:11_log.txt', # 90K
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-02:38:15_log.txt' # 100K
# ]

# DNN Seq
# log_files = [
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-18:10:52_log.txt', # 10K
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-18:42:42_log.txt', # 20K
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-19:39:51_log.txt', # 30K
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-20:51:28_log.txt', # 40K
# # '/home/robert/Research/Self-Modeling/logs/2018-08-04-18:03:39_log.txt', # 50K
# # '/home/robert/Research/Self-Modeling/logs/2018-08-04-21:48:56_log.txt', # 60K
# # '/home/robert/Research/Self-Modeling/logs/2018-08-04-18:46:40_log.txt', # 70K
# # '/home/robert/Research/Self-Modeling/logs/2018-08-04-19:40:16_log.txt', # 80K
# # '/home/robert/Research/Self-Modeling/logs/2018-08-04-20:38:11_log.txt', # 90K
# '/home/robert/Research/Self-Modeling/logs/2018-08-05-12:11:16_log.txt' # 100K
# ]

# Other Methods
# log_files = [
# '/home/robert/Research/Self-Modeling/logs/2018-08-03-21:26:56_log.txt', # KNN
# '/home/robert/Research/Self-Modeling/logs/2018-08-03-22:56:21_log.txt', # GAN
# '/home/robert/Research/Self-Modeling/logs/2018-08-03-18:30:54_log.txt', # VAE
# # '/home/robert/Research/Self-Modeling/logs/2018-08-03-03:13:16_log.txt', # DDPG
# ]

# # DNN Baseline
# log_files = [
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-11:12:42_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-01:59:10_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-02:23:51_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-02:59:20_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-04:20:07_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-04:20:07_log.txt', # Problem duplicate model!
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-05:08:02_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-06:02:19_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-04-07:02:52_log.txt',
# '/home/robert/Research/Self-Modeling/logs/2018-08-03-21:40:24_log.txt'
# ]
# i = 10



# for log in log_files:
#     # 10K
#     all_drifts_medians, low, high = make_drift_chart_from_path(log)
#     x = np.arange(len(error))
#     plt.plot(x, all_drifts_medians)
#     # plt.ylim(0, 0.74)
#     plt.fill_between(x, low, high, alpha=0.5)
#     # plt.savefig(str(i)+'K')
#     # plt.clf()
#     # i += 10
# plt.show()