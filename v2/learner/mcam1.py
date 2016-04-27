
import tensorflow as tf

import learner.base


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(
        shape, stddev=0.1,
        name='weights'
    ))

def bias_variable(shape):
    return tf.Variable(tf.constant(
        0.1, shape=shape,
        name='biases'
    ))

def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='VALID',
        name='conv'
    )

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
        name='maxpool'
    )


class McamLearner(learner.base.Learner):

    def __init__(self, learning_rate=1e-4, window_size=100, N1=32, N2=64, kp=0.8, **kwargs):

        S = ((window_size - (5-1)) / 2. - (5-1)) / 2.
        if S != int(S):
            raise RuntimeError('McamLearner.__init__: Invalid window size')

        S = int(S)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.inputs_pl = tf.placeholder(
            tf.float32,
            shape = (None, None, None, 3),
            name = 'inputs'
        )
        self.labels_pl = tf.placeholder(
            tf.float32,
            shape = (None, 2),
            name = 'labels'
        )
        self.keep_prob_pl = tf.placeholder(
            tf.float32,
            name = 'keep_prob'
        )

        self.kp = kp

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 3, N1])
            b_conv1 = bias_variable([N1])
            h_conv1 = tf.nn.relu(conv2d(self.inputs_pl, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, N1, N2])
            b_conv2 = bias_variable([N2])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([S*S*N2, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, S*S*N2])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('drop'):
            keep_prob = self.keep_prob_pl
            h_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([1024, 2])
            b_fc2 = bias_variable([2])
            logits = tf.matmul(h_drop, W_fc2) + b_fc2

        self.outputs = tf.nn.softmax(logits, name='outputs')

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits, self.labels_pl, name='xent'
            )
        )

        self.stats = self.loss

        super(self.__class__, self).__init__(**kwargs)


    def train(self, inputs, labels):
        return super(self.__class__, self).train(
            inputs, labels, feed_dict={self.keep_prob_pl: self.kp}
        )

    def test(self, inputs, labels):
        return super(self.__class__, self).test(
            inputs, labels, feed_dict={self.keep_prob_pl: 1.0}
        )

    def __call__(self, inputs):
        return super(self.__class__, self).__call__(
            inputs, feed_dict={self.keep_prob_pl: 1.0}
        )







