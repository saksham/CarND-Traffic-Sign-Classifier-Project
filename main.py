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

logger.info('Augmenting training data...')
TRAINING_DATA_AUGMENTERS = [
    augmentation.GaussianBlurAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
    augmentation.AffineTransformAugmenter(),
]
d_train = augmentation.Augmenter.apply(training, TRAINING_DATA_AUGMENTERS)
print('Augmented training data: ', utils.get_summary([d_train]))


logger.info('Pre-processing training and validation data sets...')
PRE_PROCESSORS = [
    preprocessing.GrayScaleConverter(),
    preprocessing.ZNormaliser(),
]

# Perform pre-processing on validation and augmented data sets
d_train = preprocessing.PreProcessor.apply(d_train, PRE_PROCESSORS)
d_validation = preprocessing.PreProcessor.apply(validation, PRE_PROCESSORS)

tf.reset_default_graph()
placeholders, operations = lenet.setup_graph()

# TODO: remove this override
HYPER_PARAMETERS['EPOCHS'] = 10

logger.info('Hyper-parameters: %s', HYPER_PARAMETERS)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    logger.info("Training...")
    for i in range(HYPER_PARAMETERS['EPOCHS']):
        lenet.train_one_epoch(d_train, placeholders, operations)

        training_accuracy = lenet.evaluate(d_train, placeholders, operations)
        validation_accuracy = lenet.evaluate(d_validation, placeholders, operations)

        logger.info("EPOCH {} ...".format(i + 1))
        logger.info("Accuracy on training dataset = {:.3f}".format(training_accuracy))
        logger.info("Validation Accuracy = {:.3f}".format(validation_accuracy))
        logger.info()

    saver.save(sess, './data/model/lenet')
    logger.info("Model saved")
