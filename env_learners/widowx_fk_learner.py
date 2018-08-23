import tensorflow as tf
import numpy as np
import time, datetime, pickle
from misc import losses
from misc.models import fk_learner
from test_planners.control_widowx import calc_end_effector_pos

def batch(data, batch_size):
    batches = []
    while len(data) >= batch_size:
        batches.append(data[:batch_size])
        data = data[batch_size:]
    return batches

if __name__ == '__main__':
    load = None

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
    valid = None

    angle_bounds = np.array([2.617, 1.571, 1.571, 1.745])
    pos_bounds = np.array([0.37, 0.37, 0.51])
    nb_epochs = 100

    with tf.Session(config=tf_config) as sess:
        sess_start = time.time()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        data_log = open('logs/'+datetime_str+'_log.txt', 'w+')


        # generic data gathering

        X_data = []
        y_data = []

        x = tf.placeholder(dtype=tf.float32, shape=([None, 4]))
        real_y = tf.placeholder(dtype=tf.float32, shape=([None, 3]))
        y = fk_learner(x)*pos_bounds
        loss = losses.loss_p(real_y, y)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        if load is None:
            print('Loading Data...')
            data_file = 'real_widowx_train_10hz_100K_default_processed.pkl'
            train_data = pickle.load(open(data_file, 'rb+'))
            for episode in train_data:
                X_data.append(episode[0][:-3]*angle_bounds)
                y_data.append(episode[0][-3:])
            x_batches = batch(X_data, 32)
            y_batches = batch(y_data, 32)
            print('Training...')
            for epoch in range(nb_epochs):
                start = time.time()
                losses = []
                for i in range(len(x_batches)):
                    x_batch = np.array(x_batches[i])
                    y_batch = np.array(y_batches[i])
                    l, _ = sess.run([loss, train_step], feed_dict={x: x_batch, real_y: y_batch})
                    losses.append(l)

                duration = time.time()-start
                print('Epoch '+str(epoch)+' with loss '+str(np.mean(losses))+' Completed in '+str(duration)+'s')
                if epoch%10 == 9:
                    save_path = saver.save(sess, 'models/' + str(datetime_str) + '.ckpt')
                    print('Model Saved to: models/' + str(datetime_str) + '.ckpt')
        else:
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')

        print('Testing...')
        drifts = []
        for i in range(10000):
            angles = np.random.uniform(-1, 1, (1,4))*angle_bounds
            pos = calc_end_effector_pos(angles[0])
            pred = sess.run([y], feed_dict={x: angles})
            drift = np.linalg.norm(pos-pred)
            drifts.append(drift)
        print(np.median(drifts))
        print(np.std(drifts))