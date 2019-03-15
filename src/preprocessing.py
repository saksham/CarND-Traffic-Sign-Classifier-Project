#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import ABC

import numpy as np

from src.loading import DataSet
from src.utils import get_logger, ParameterizedProcessor

logger = get_logger('Pre-processor')


class PreProcessor(ParameterizedProcessor, ABC):
    def __init__(self, name, parameters):
        super().__init__(name, parameters)

    @staticmethod
    def apply(data_set, steps):
        for s in steps:
            logger.info('Running {} on {} dataset...'.format(s.name, data_set.name))
            logger.info('\tParameters: {}'.format(json.dumps(s.parameters)))
            data_set = s.process(data_set)
        return data_set


class GrayScaleConverter(ParameterizedProcessor):
    # OpenCV uses the following weights to convert to grayscale
    # https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
    PARAMETERS = {
        'channel-weights': [0.299, 0.587, 0.114]
    }

    def __init__(self):
        super().__init__('TO_GRAY_SCALE', GrayScaleConverter.PARAMETERS)

    def process(self, data_set):
        """
        Convert example images to gray-scale

        Source: https://medium.com/@REInvestor/converting-color-images-to-grayscale-ab0120ea2c1e
        :param data_set:
        :return:
        """

        m, w, h, c = data_set.X.shape
        x = np.zeros((m, w, h, 1))
        for i in range(m):
            w_mean = np.tensordot(data_set.X[i], self._parameters['channel-weights'], axes=(-1, -1))[..., None]
            gray = w_mean.astype(data_set.X[i].dtype)
            x[i] = gray

        return DataSet(data_set.name, x, data_set.y, data_set.count)


class ZNormaliser(ParameterizedProcessor):
    def __init__(self, mean=None, sigma=None):
        super().__init__('Z_NORMALISATION')
        self._mean = mean
        self._sigma = sigma

    def process(self, data_set):
        if not self._mean:
            logger.info('No means were calculated yet. Calculating score and mean from {}...'.format(data_set.name))
            self._mean = np.mean(data_set.X)
            self._sigma = np.std(data_set.X)
        logger.info('Normalising {} with mean: {} and sigma: {}...'.format(data_set.name, self._mean, self._sigma))
        x = (data_set.X - self._mean) / self._sigma
        return DataSet(data_set.name, x, data_set.y, data_set.count)
