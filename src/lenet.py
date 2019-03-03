#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import flatten

HYPER_PARAMETERS = {
    'LEARNING_RATE': 0.01,
    'EPOCHS': 10,
    'BATCH_SIZE': 128
}


def evaluate(X_data, y_data, x, y, accuracy_operation):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    batch_size = HYPER_PARAMETERS['BATCH_SIZE']
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def setup_training_pipeline(x, y):
    one_hot_y = tf.one_hot(y, 43)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=HYPER_PARAMETERS['LEARNING_RATE'])
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return training_operation, accuracy_operation


def LeNet(x):
    """
    ### Model Architecture
    [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

    ### Input
    The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since the images are grayscale, C is 1 in this case.

    * Layer 1:
      * Convolutional: output shape is 28x28x6.
      * Activation.ReLU
    * Pooling (max): output shape is 14x14x6.
    * Layer 2:
        * Convolutional.output shape is 10x10x16.
        * Activation.ReLU
    * Pooling (max): output shape is 5x5x16.
    * Layer 3:
        * Fully Connected: number of outputs 120.
        * Activation.ReLU
    * Layer 4:
        * Fully Connected: number of outputs 84.
        * Activation.ReLU
    * Layer 5:
        * Fully Connected: number of outputs 43 (logits).

    :param x: Input to the network
    :return: logits
    """
    # Hyper parameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input: 32x32x3, Output: 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Layer 1: Activation
    conv1 = tf.nn.relu(conv1)

    # Layer 1: Pooling. Input: 28x28x6, Output: 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Layer 2: Activation.
    conv2 = tf.nn.relu(conv2)

    # Layer 2: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Layer 3: Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Layer 4: Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
