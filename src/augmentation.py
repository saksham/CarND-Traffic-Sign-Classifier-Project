#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from src.loading import DataSet
from src.utils import get_logger, ParameterizedProcessor

logger = get_logger('augmentation')


def augment_data_set(data_set, augmenters):
    result = DataSet(data_set.name, data_set.X, data_set.y, data_set.count)
    for augmenter in augmenters:
        result = augmenter.process(result)
    return result


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
    return DataSet(data_set.name, x, y, data_set.count * 2)


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
        'PIXELS': 4
    }

    def __init__(self):
        super().__init__('AFFINE_TRANSFORM_AUGMENTER', AffineTransformAugmenter.PARAMETERS)

    @staticmethod
    def _affine_transform(image):
        h, w, _ = image.shape
        px = AffineTransformAugmenter.PARAMETERS['PIXELS']
        dx, dy = np.random.randint(-px, px, 2)
        augmented_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image.squeeze(), augmented_matrix, (h, w))

    def process(self, data_set):
        return augment(data_set, AffineTransformAugmenter._affine_transform)


class HorizontalFlipper(ParameterizedProcessor):
    def __init__(self):
        super().__init__('HORIZONTAL_FLIPPER')

    def process(self, data_set):
        return augment(data_set, lambda img: cv2.flip(img, 0))


class RandomScalerAugmenter(ParameterizedProcessor):
    PARAMETERS = {
        'max-width': 40,
        'interpolation': cv2.INTER_LANCZOS4
    }

    def __init__(self):
        super().__init__('RANDOM_SCALER', RandomScalerAugmenter.PARAMETERS)

    def _scale_randomly(self, image):
        target_size = np.random.randint(32, RandomScalerAugmenter.PARAMETERS['max-width'], 2)
        new_image = cv2.resize(image, (target_size[0], target_size[1],),
                               interpolation=RandomScalerAugmenter.PARAMETERS['interpolation'])
        h, w, *_ = np.array(new_image.shape)
        start = np.array([h // 2 - 16, w // 2 - 16])
        end = start + 32
        new_indices = np.hstack((start, end)).astype(int)
        cropped = new_image[new_indices[0]:(new_indices[0] + 32), new_indices[1]:(new_indices[1] + 32), :]
        return cropped

    def process(self, data_set):
        return augment(data_set, lambda img: self._scale_randomly(img))
