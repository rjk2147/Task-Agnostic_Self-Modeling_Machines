from test_planners.control_widowx import calc_end_effector_pos, clip_joints
from envs import widowx_arm
import numpy as np
import tensorflow as tf
import statistics
from env_learners.dnn_env_learner import DNNEnvLearner

# load = 'models/2018-08-05-12:11:16.ckpt' # 100K DNN Seq (5)
# load = 'models/2018-08-05-18:10:52.ckpt' # 10K DNN Seq (5)
load = 'models/2018-08-14-22:53:53.ckpt' # 10K DNN Seq (5) Deformed

env = widowx_arm.WidowxROS()
max_action = env.action_space.high
env_learner = DNNEnvLearner(env)
nb_test_episodes = 100


tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
try:
    saver = tf.train.Saver()
except:
    saver = None

diffs = []
with tf.Session(config=tf_config) as sess:
    saver.restore(sess, load)
    print('Model: ' + load + ' Restored')
    print('Starting...')
    env_learner.initialize(sess, load=True)
    for i in range(nb_test_episodes):
        obs = env.reset()
        for episode_step in range(100):
            action = np.random.uniform(-1, 1, 4)
            new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
            real_obs, r, real_done, _ = env.step(max_action * action)
            diffs.append(new_obs-real_obs)
            obs = real_obs
        print('Episode '+str(i+1)+' Completed')
drifts = []
for i in range(len(diffs)):
    drifts.append(np.linalg.norm(diffs[i][-3:]))

print('')
print('Median Drift: '+str(statistics.median(drifts)))
print('Mean Drift: '+str(statistics.mean(drifts)))
print('Stdev Drift: '+str(statistics.stdev(drifts)))
print('')
print('Median Diff: '+str(np.median(diffs, axis=0)))
print('Mean Diff: '+str(np.mean(diffs, axis=0)))
print('Stdev Diff: '+str(np.std(diffs, axis=0)))