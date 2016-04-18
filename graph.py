
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


def inference(image_placeholder, keep_prob_placeholder):

    N1 = 32
    N2 = 64

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, N1])
        b_conv1 = bias_variable([N1])
        h_conv1 = tf.nn.relu(conv2d(image_placeholder, W_conv1) + b_conv1)
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
        keep_prob = keep_prob_placeholder
        h_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 1])
        b_fc2 = bias_variable([1])
        y = tf.matmul(h_drop, W_fc2) + b_fc2

    return y


def loss(predicted, actual):
    rmse = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(predicted, actual))))
    return rmse


def training(optimizer, loss):
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(predicted, actual):
    errors = tf.abs(tf.sub(predicted, actual))
    avg_error = tf.reduce_mean(errors)
    return avg_error




