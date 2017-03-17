
import tensorflow as tf
from math import sqrt
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

def put_kernels_on_grid(kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7


class McamLearner(learner.base.Learner):

    def __init__(self, learning_rate=1e-4, window_size=100, N1=32, N2=64, kp=0.4, **kwargs):

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

        self.writer = tf.summary.FileWriter('./log')

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 3, N1])
            b_conv1 = bias_variable([N1])
            h_conv1 = tf.nn.relu(conv2d(self.inputs_pl, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            # Visualize filters for first convolution
            grid = put_kernels_on_grid(W_conv1)
            tf.summary.image('conv1/features', grid, max_outputs=1)

        # Visualize first layer of convolutions
        V = tf.slice(h_conv1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
        V = tf.reshape(V, (156, 156, N1))
        # Reorder so the channels are in the first dimension, x and y follow.
        V = tf.transpose(V, (2, 0, 1))
        # Bring into shape expected by image_summary
        V = tf.reshape(V, (-1, 156, 156, 1))
        tf.summary.image("first_conv", V, max_outputs=32)

        # Visualize first max pooling layer
        V1 = tf.slice(h_pool1, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
        V1 = tf.reshape(V1, (78, 78, N1))
        # Reorder so the channels are in the first dimension, x and y follow.
        V1 = tf.transpose(V1, (2, 0, 1))
        # Bring into shape expected by image_summary
        V1 = tf.reshape(V1, (-1, 78, 78, 1))
        tf.summary.image("first_mp", V1, max_outputs=32)

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, N1, N2])
            b_conv2 = bias_variable([N2])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # Visualize second convolution layer
        V2 = tf.slice(h_conv2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_second')
        V2 = tf.reshape(V2, (74, 74, N2))
        # Reorder so the channels are in the first dimension, x and y follow.
        V2 = tf.transpose(V2, (2, 0, 1))
        # Bring into shape expected by image_summary
        V2 = tf.reshape(V2, (-1, 74, 74, 1))
        tf.summary.image("second_conv", V2, max_outputs=64)

        # Visualize second max pooling layer
        V3 = tf.slice(h_pool2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
        V3 = tf.reshape(V3, (37, 37, N2))
        # Reorder so the channels are in the first dimension, x and y follow.
        V3 = tf.transpose(V3, (2, 0, 1))
        # Bring into shape expected by image_summary
        V3 = tf.reshape(V3, (-1, 37, 37, 1))
        tf.summary.image("second_mp", V3, max_outputs=64)

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
                logits=logits, labels=self.labels_pl, name='xent'
            )
        )

        self.stats = self.loss

        self.merged = tf.summary.merge_all()

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







