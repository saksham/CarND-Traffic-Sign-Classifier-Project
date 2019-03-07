#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from src.loading import DataSet
from src.utils import get_logger, ParameterizedProcessor

logger = get_logger('augmentation')


def augment(data_set, transform_func):
    m, h, w, c = data_set.X.shape
    x = np.zeros((2 * m, h, w, c))
    y = np.zeros((2 * m))
    for i in range(m):
        original_image = data_set.X[i, :]
        x[i, :] = original_image
        transformed = transform_func(original_image)
        x[m + i, :] = transformed.reshape((h, w, c))
        y[i] = data_set.y[i]
        y[m + i] = data_set.y[i]
    return DataSet(data_set.name, x, y, data_set.count)


class GaussianBlurAugmenter(ParameterizedProcessor):
    PARAMETERS = {
        'ksize': (3, 3),
        'sigma': 0
    }

    def __init__(self):
        super().__init__('GAUSSIAN_BLUR', GaussianBlurAugmenter.PARAMETERS)

    def process(self, data_set):
        return augment(data_set, lambda x: cv2.blur(x.squeeze(), self.parameters['ksize'], self.parameters['sigma']))


class AffineTransformAugmenter(ParameterizedProcessor):
    PARAMETERS = {
        'PX': 2
    }

    def __init__(self):
        super().__init__('AFFINE_TRANSFORM_AUGMENTER', AffineTransformAugmenter.PARAMETERS)

    @staticmethod
    def _affine_transform(image):
        h, w, _ = image.shape
        px = AffineTransformAugmenter.PARAMETERS['PX']
        dx, dy = np.random.randint(-px, px, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image.squeeze(), M, (h, w))

    def process(self, data_set):
        return augment(data_set, AffineTransformAugmenter._affine_transform)
