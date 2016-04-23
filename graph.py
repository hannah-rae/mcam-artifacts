
import math

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='biases')
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv'
    )

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool'
    )


def inference(images, keep_prob):

    N1 = 32
    N2 = 64

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, N1])
        b_conv1 = bias_variable([N1])
        h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, N1, N2])
        b_conv2 = bias_variable([N2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4*4*N2, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*N2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('drop'):
        h_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('linear'):
        W_linear = weight_variable([1024, 2])
        b_linear = bias_variable([2])
        y = tf.matmul(h_drop, W_linear) + b_linear

    return y


def loss(logits, labels):
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    mean_xent = tf.reduce_mean(xent)
    return mean_xent


def training(optimizer, loss):
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy




