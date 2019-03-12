#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.utils import shuffle

from src import loading, utils, lenet, preprocessing, augmentation
from src.lenet import HYPER_PARAMETERS

logger = utils.get_logger('main')

logger.info('-' * 200)
logger.info('Loading data set...')
training, validation, test = loading.load_all()

# Convert to grayscale to save memory and processing time
training, validation, test = [preprocessing.GrayScaleConverter().process(d) for d in [training, validation, test]]
logger.info(utils.get_summary([training, validation]))

# List of enabled data augmenters for training data set
TRAINING_DATA_AUGMENTERS = [
    augmentation.HorizontalFlipper(),
    augmentation.AffineTransformAugmenter(),
    augmentation.GaussianBlurAugmenter(),
    augmentation.RandomScalerAugmenter()
]
logger.info('Augmenting training data...')
d_train = augmentation.augment_data_set(training, TRAINING_DATA_AUGMENTERS)
print('Augmented training data: ', utils.get_summary([d_train]))

# List of enabled data pre-processors
PRE_PROCESSORS = [
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

logger.info('Hyper-parameters: %s', HYPER_PARAMETERS)

# TODO: this override is just for local testing - remove it in the final version
HYPER_PARAMETERS['EPOCHS'] = 1

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = d_train.count

    logger.info("Training...")
    for i in range(HYPER_PARAMETERS['EPOCHS']):
        X_train, y_train = shuffle(d_train.X, d_train.y)
        training_accuracy = 0
        for offset in range(0, num_examples, HYPER_PARAMETERS['BATCH_SIZE']):
            end = offset + HYPER_PARAMETERS['BATCH_SIZE']
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, mode: Mode.TRAINING.value})

        training_accuracy = lenet.evaluation(d_train.X, d_train.y, x, y, mode, accuracy_operation)
        validation_accuracy = lenet.evaluation(d_validation.X, d_validation.y, x, y, mode, accuracy_operation)
        logger.info("EPOCH {} ...".format(i + 1))
        logger.info("Accuracy on training dataset = {:.3f}".format(training_accuracy))
        logger.info("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './data/model/lenet')
    logger.info("Model saved")
