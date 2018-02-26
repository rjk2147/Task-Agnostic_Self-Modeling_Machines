import tensorflow as tf


def generator_model(x, out_dim, drop_rate=0.5):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        # x_seq = tf.split(x, buff_len, 1)
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
        rnn_cell = tf.contrib.rnn.GRUCell(512)
        outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
        x = outputs[-1]

        # x_new = []
        # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        # for x in x_seq:
        #     x = tf.expand_dims(x, -1)
        #     x_new.append(x)
        # x = tf.concat(x_new, axis=2)
        # x = tf.layers.conv1d(x, 64, 3)
        # x = tf.layers.conv1d(x, 32, 1)
        # x = tf.layers.flatten(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 128)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, out_dim)

        return tf.nn.tanh(x)


def discriminator_model(x, drop_rate=0.5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:

        # x_seq = tf.split(x, buff_len, 1)
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
        # rnn_cell = tf.contrib.rnn.GRUCell(512)
        outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        x = outputs[-1]

        # x_new = []
        # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        # for x in x_seq:
        #     x = tf.expand_dims(x, -1)
        #     x_new.append(x)
        # x = tf.concat(x_new, axis=2)
        # x = tf.layers.conv1d(x, 64, 3)
        # x = tf.layers.conv1d(x, 32, 1)
        # x = tf.layers.flatten(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 128)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 1)
        # return x
        return tf.nn.sigmoid(x)


def simple_gen(x, out_dim, buff_len=None, drop_rate=None):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 64)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, out_dim)
        return tf.nn.tanh(x)
def simple_disc(x, drop_rate=None, buff_len=None):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:

        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(64)
        # rnn_cell = tf.contrib.rnn.GRUCell(512)
        outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
        x = outputs[-1]

        x = tf.layers.dense(x, 64)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 1)
        # return x
        return tf.nn.sigmoid(x)

def done_model(x, out_dim, buff_len, drop_rate=0.5):
    # x = tf.split(x, buff_len, 1)
    # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    # # for x in x_all:
    # #     x = tf.expand_dims(x, -1)
    # #     x_new.append(x)
    # # x = tf.concat(x_new, axis=2)
    # # # x = tf.layers.conv1d(x, 128, 7)
    # # x = tf.layers.conv1d(x, 64, 5)
    # # x = tf.layers.conv1d(x, 32, 3)
    # # x = tf.layers.flatten(x)
    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
    #
    # # rnn_cell = tf.contrib.rnn.GRUCell(512)
    # outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # x = outputs[-1]
    #
    # # x_new = []
    # # for x in outputs:
    # #     x = tf.expand_dims(x, -1)
    # #     x_new.append(x)
    # # x = tf.concat(x_new, axis=2)
    # #
    # # x = tf.layers.conv1d(x, 128, 7)
    # # x = tf.layers.conv1d(x, 64, 5)
    # # x = tf.layers.conv1d(x, 32, 3)
    # # x = tf.layers.flatten(x)
    # # x = tf.layers.batch_normalization(x)
    #
    #
    # x = tf.layers.dense(x, 1024)
    # # x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate=drop_rate)
    #
    # # # x = tf.layers.batch_normalization(x)
    # # x = tf.layers.dense(x, 1024)
    # # # x = tf.nn.relu(x)
    # # x = tf.layers.dropout(x, rate=drop_rate)
    #
    # x = tf.layers.dense(x, 512)
    # # x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate=drop_rate)

    # x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    # x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    # x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    # x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    # x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)
    # return x
    return tf.nn.sigmoid(x)


def explore_model(state, act_dim, drop_rate=0.5):
    with tf.variable_scope('explore') as scope:
        x = state
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=drop_rate)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=drop_rate)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=drop_rate)

        x = tf.layers.dense(x, 512)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=drop_rate)

        x = tf.layers.dense(x, act_dim)
        return tf.nn.tanh(x)


def gen_state(x, out_dim, buff_len, drop_rate=0.0):
    # # x = tf.layers.batch_normalization(x)
    # #
    # # x_all = tf.split(x, buff_len, 1)
    # # x_new = []
    # # for x in x_all:
    # #     x = tf.layers.batch_normalization(x)
    # #     x = tf.layers.dense(x, 1024)
    # #     x = tf.layers.dropout(x, rate=drop_rate)
    # #
    # #     x = tf.layers.batch_normalization(x)
    # #     x = tf.layers.dense(x, 1024)
    # #     x = tf.layers.dropout(x, rate=drop_rate)
    # #
    # #     # x = tf.layers.batch_normalization(x)
    # #     # x = tf.layers.dense(x, 512)
    # #     # x = tf.layers.dropout(x, rate=drop_rate)
    # #
    # #     x_new.append(x)
    # # # # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
    # #
    # # x = x_new
    # #
    # x = tf.split(x, buff_len, 1)
    # lstm = tf.contrib.rnn.MultiRNNCell([
    #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(2048)),
    #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1024), output_keep_prob=1.0-drop_rate),
    #     # tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(512), output_keep_prob=1.0-drop_rate),
    #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1024), output_keep_prob=1.0 - drop_rate),
    #     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(256), output_keep_prob=1.0-drop_rate),
    #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(256), output_keep_prob=1.0-drop_rate),
    #     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(128), output_keep_prob=1.0-drop_rate),
    #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(128), output_keep_prob=1.0-drop_rate),
    #     #     # tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(64), output_keep_prob=1.0-drop_rate),
    #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(64), output_keep_prob=1.0-drop_rate)
    # ])
    # #
    # outputs, states = tf.nn.static_rnn(lstm, x, dtype=tf.float32)
    # x = outputs[-1]




    # x_seq = tf.split(x, buff_len, 1)

    x_seq = tf.split(x, buff_len, 1)

    # rnn_cell = tf.contrib.rnn.BasicLSTMCell(2048)
    # # rnn_cell = tf.contrib.rnn.GRUCell(2048)
    # outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
    # # x = outputs[-1]
    # x_seq = outputs

    # x_new = []
    # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
    # for x in x_seq:
    #     x = tf.expand_dims(x, -1)
    #     x_new.append(x)
    # x = tf.concat(x_new, axis=2)
    # x = tf.layers.conv1d(x, 64, 3)
    # x = tf.layers.conv1d(x, 32, 1)
    # x = tf.layers.flatten(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)
    #
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 64)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=drop_rate)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)

    return tf.nn.tanh(x)
