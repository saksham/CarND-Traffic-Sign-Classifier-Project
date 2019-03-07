#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn.utils import shuffle

from src import lenet, loading, preprocessing, augmentation
from src.utils import get_logger, get_summary
from src.lenet import HYPER_PARAMETERS

logger = get_logger('main')

logger.info('Loading data set...')
training, validation, test = loading.load_all()
logger.info(get_summary([training, validation, test]))

training_data_augmenters = [
    augmentation.AffineTransformAugmenter(),
    augmentation.GaussianBlurAugmenter()
]
logger.info('Augmenting training data...')
training = preprocessing.PreProcessor.apply(training, training_data_augmenters)
logger.info(get_summary([training]))

preprocessors = [
    preprocessing.DataShuffler(),
    preprocessing.GrayScaleConverter(),
#    preprocessing.MinMaxNormaliser(),
    preprocessing.ZNormaliser(),
]
logger.info('Pre-processing training and validation data sets...')
training, validation = tuple(preprocessing.PreProcessor.apply(d, preprocessors) for d in [training, validation])

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))

training_operation, accuracy_operation = lenet.setup_training_pipeline(x, y)

logger.info('Hyper-parameters: %s', HYPER_PARAMETERS)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = training.count

    logger.info("Training...")
    for i in range(HYPER_PARAMETERS['EPOCHS']):
        X_train, y_train = shuffle(training.X, training.y)
        training_accuracy = 0
        for offset in range(0, num_examples, HYPER_PARAMETERS['BATCH_SIZE']):
            end = offset + HYPER_PARAMETERS['BATCH_SIZE']
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        training_accuracy = lenet.evaluate(X_train, y_train, x, y, accuracy_operation)
        validation_accuracy = lenet.evaluate(validation.X, validation.y, x, y, accuracy_operation)
        logger.info("EPOCH {} ...".format(i + 1))
        logger.info("Training Accuracy = {:.3f}".format(training_accuracy))
        logger.info("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './data/model/lenet')
    print("Model saved")
