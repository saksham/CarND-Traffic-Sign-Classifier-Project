#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from src import loading, utils, lenet, preprocessing, augmentation
from src.lenet import HYPER_PARAMETERS

logger = utils.get_logger('main')

logger.info('-' * 200)
logger.info('Loading data set...')
training, validation, test = loading.load_all()
logger.info(utils.get_summary([training, validation]))

TRAINING_DATA_AUGMENTERS = [
    augmentation.GaussianBlurAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
]
# List of enabled data augmenters for training data set
logger.info('Augmenting training data...')
d_train = augmentation.Augmenter.apply(training, TRAINING_DATA_AUGMENTERS)
print('Augmented training data: ', utils.get_summary([d_train]))

# List of enabled data pre-processors
PRE_PROCESSORS = [
    preprocessing.GrayScaleConverter(),
    preprocessing.ZNormaliser(),
]

# Perform pre-processing on augmented training and validation data sets
logger.info('Pre-processing training and validation data sets...')
d_train = preprocessing.PreProcessor.apply(d_train, PRE_PROCESSORS)
d_validation = preprocessing.PreProcessor.apply(validation, PRE_PROCESSORS)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
mode = tf.placeholder(tf.string, (None))

training_operation, accuracy_operation, logits = lenet.setup_graph(x, y, mode)

logger.info('Hyper-parameters: %s', HYPER_PARAMETERS)
from src.lenet import HYPER_PARAMETERS, Mode

HYPER_PARAMETERS['EPOCHS'] = 0
logger.info('Hyper-parameters: %s', HYPER_PARAMETERS)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = d_train.count

    logger.info("Training...")
    for i in range(HYPER_PARAMETERS['EPOCHS']):
        d_train = utils.shuffle(d_train)
        for offset in range(0, num_examples, HYPER_PARAMETERS['BATCH_SIZE']):
            end = offset + HYPER_PARAMETERS['BATCH_SIZE']
            batch_x, batch_y = d_train.X[offset:end], d_train.y[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, mode: Mode.TRAINING.value})

        training_accuracy = lenet.evaluation(d_train.X, d_train.y, x, y, mode, accuracy_operation)
        validation_accuracy = lenet.evaluation(d_validation.X, d_validation.y, x, y, mode, accuracy_operation)
        logger.info("EPOCH {} ...".format(i + 1))
        logger.info("Accuracy on training dataset = {:.3f}".format(training_accuracy))
        logger.info("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './data/model/lenet')
    logger.info("Model saved")
