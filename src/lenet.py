#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import namedtuple
from enum import Enum

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from src.utils import get_logger, shuffle

logger = get_logger('LeNet')

Placeholders = namedtuple('Placeholders', ['x', 'y', 'keep_probability'])
Operations = namedtuple('Operations', ['logits', 'training', 'accuracy'])


HYPER_PARAMETERS = {
    'LEARNING_RATE': 0.001,
    'EPOCHS': 100,
    'BATCH_SIZE': 512,
    'KEEP_PROBABILITY_DURING_TRAINING': 0.7,
    'mu': 0,
    'sigma': 0.1
}


def train_one_epoch(data_set, placeholders, operations):
    num_examples = data_set.count
    sess = tf.get_default_session()
    data_set = shuffle(data_set)

    for offset in range(0, num_examples, HYPER_PARAMETERS['BATCH_SIZE']):
        end = offset + HYPER_PARAMETERS['BATCH_SIZE']
        batch_x, batch_y = data_set.X[offset:end], data_set.y[offset:end]
        sess.run(operations.training, feed_dict={
            placeholders.x: batch_x,
            placeholders.y: batch_y,
            placeholders.keep_probability: HYPER_PARAMETERS['KEEP_PROBABILITY_DURING_TRAINING']
        })


def evaluate(data_set, placeholders, operations):
    num_examples = data_set.count
    total_accuracy = 0
    sess = tf.get_default_session()
    batch_size = HYPER_PARAMETERS['BATCH_SIZE']
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = data_set.X[offset:offset + batch_size], data_set.y[offset:offset + batch_size]
        accuracy = sess.run(operations.accuracy, feed_dict={
            placeholders.x: batch_x,
            placeholders.y: batch_y,
            placeholders.keep_probability: 1
        })
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def setup_graph():
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None,))
    keep_probability = tf.placeholder(tf.float32)

    placeholders = Placeholders(x, y, keep_probability)

    one_hot_y = tf.one_hot(y, 43)

    logits_op = LeNet(x, keep_probability)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits_op)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=HYPER_PARAMETERS['LEARNING_RATE'])
    training_op = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits_op, 1), tf.argmax(one_hot_y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    operations = Operations(logits_op, training_op, accuracy_op)
    return placeholders, operations


def LeNet(x, keep_probability):
    """
    ### Model Architecture
    [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

    ### Input
    The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since the images are grayscale, C is 1 in this case.

    * Layer 1:
        * Convolutional: output shape is 28x28x6.
        * Activation.ReLU
        * Dropout
    * Pooling (max): output shape is 14x14x6.
    * Layer 2:
        * Convolutional.output shape is 10x10x16.
        * Activation.ReLU
        * Dropout
    * Pooling (max): output shape is 5x5x16.
    * Layer 3:
        * Fully Connected: number of outputs 120.
        * Activation.ReLU
        * Dropout
    * Layer 4:
        * Fully Connected: number of outputs 84.
        * Activation.ReLU
        * Dropout
    * Layer 5:
        * Fully Connected: number of outputs 43 (logits).

    :param x: Input to the network
    :param keep_probability: keep probability for the droput
    :return: logits
    """
    logger.info('Running LeNet with keep probability of {}...'.format(keep_probability))

    # Hyper parameters
    channels = x.shape[3].value

    # Layer 1: Convolutional. Input: 32x32xC, Output: 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, channels, 6), mean=(HYPER_PARAMETERS['mu']), stddev=(
        HYPER_PARAMETERS['sigma'])))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Layer 1: Activation
    conv1 = tf.nn.relu(conv1)

    # Layer 1: Dropout
    conv1 = tf.nn.dropout(conv1, keep_probability)

    # Layer 1: Pooling. Input: 28x28x6, Output: 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=(HYPER_PARAMETERS['mu']), stddev=(
        HYPER_PARAMETERS['sigma'])))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Layer 2: Activation.
    conv2 = tf.nn.relu(conv2)

    # Dropout
    conv2 = tf.nn.dropout(conv2, keep_probability)

    # Layer 2: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=(HYPER_PARAMETERS['mu']), stddev=(
        HYPER_PARAMETERS['sigma'])))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Layer 3: Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_probability)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=(HYPER_PARAMETERS['mu']), stddev=(
        HYPER_PARAMETERS['sigma'])))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Layer 4: Activation.
    fc2 = tf.nn.relu(fc2)

    fc2 = tf.nn.dropout(fc2, keep_probability)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=(HYPER_PARAMETERS['mu']), stddev=(
        HYPER_PARAMETERS['sigma'])))
    fc3_b = tf.Variable(tf.zeros(43))

    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
