#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import cv2
import numpy as np
import json

from src.loading import DataSet
from src.utils import get_logger, ParameterizedProcessor

logger = get_logger('augmentation')


class Augmenter(ParameterizedProcessor, ABC):
    def __init__(self, name, parameters=None):
        super().__init__(name, parameters)

    @abstractmethod
    def process_single_image(self, image):
        raise Exception('Unimplemented')

    def process(self, data_set):
        m, h, w, c = data_set.X.shape
        x = np.zeros((m, h, w, c), dtype=np.int)
        y = np.zeros(m, dtype=np.int)
        for i in range(m):
            processed = self.process_single_image(data_set.X[i])
            x[i] = processed.reshape(h, w, c)
            y[i] = data_set.y[i]
        return DataSet(data_set.name, x, y, m)

    @staticmethod
    def apply(data_set, steps):
        x = data_set.X
        y = data_set.y
        for s in steps:
            logger.info('Running {} on {} dataset...'.format(s.name, data_set.name))
            logger.info('\tParameters: {}'.format(json.dumps(s.parameters)))
            processed = s.process(data_set)
            x = np.concatenate([x, processed.X], axis=0)
            y = np.concatenate([y, processed.y], axis=0)
        return DataSet(data_set.name, x, y, len(x))


class GaussianBlurAugmenter(Augmenter):
    PARAMETERS = {
        'ksize': (3, 3),
    }

    def __init__(self):
        super().__init__('GAUSSIAN_BLUR', GaussianBlurAugmenter.PARAMETERS)

    def process_single_image(self, image):
        return cv2.blur(image, ksize=self.parameters['ksize'])


class AffineTransformAugmenter(Augmenter):
    # Parameters from http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
    PARAMETERS = {
        'TRANSLATION': [-2, 2],
        'SCALE': [0.9, 1.1],
        'ROTATION_IN_DEGREES': [-15, +15]
    }

    def __init__(self):
        super().__init__('AFFINE_TRANSFORM_AUGMENTER', AffineTransformAugmenter.PARAMETERS)

    def process_single_image(self, image):
        h, w, _ = image.shape
        assert h == w, "Only works with square images"

        scale = np.random.uniform(*AffineTransformAugmenter.PARAMETERS['SCALE'])
        theta = np.random.uniform(*AffineTransformAugmenter.PARAMETERS['ROTATION_IN_DEGREES'])
        center = np.random.uniform(*AffineTransformAugmenter.PARAMETERS['TRANSLATION'], 2) + h / 2

        m = cv2.getRotationMatrix2D(tuple(center), theta, scale)
        return cv2.warpAffine(image, m, (h, w))
