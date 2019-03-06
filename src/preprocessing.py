#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np
from sklearn import utils

from src.loading import DataSet
from src.utils import get_logger

logger = get_logger('Pre-processor')


class PreProcessingStep(ABC):
    def __init__(self, name, parameters=None):
        self._name = name
        self._parameters = parameters

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def process(self, data_set):
        raise Exception('Unimplemented')

    @property
    def info(self):
        return {'name': self._name, 'parameters': self._parameters}

    @staticmethod
    def apply(data_set, steps):
        for s in steps:
            logger.info('Running preprocessing step {} on {} dataset...'.format(s.name, data_set.name))
            logger.info('Parameters: {}'.format(json.dumps(s.parameters)))
            data_set = s.process(data_set)
        return data_set


class GrayScaleConverter(PreProcessingStep):
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


class MinMaxNormaliser(PreProcessingStep):
    def __init__(self):
        super().__init__('NORMALISE')

    def process(self, data_set):
        x = (data_set.X - 128) / 128
        return DataSet(data_set.name, x, data_set.y, data_set.count)


class GaussianBlur(PreProcessingStep):
    PARAMETERS = {
        'ksize': (3, 3),
        'sigma': 0
    }

    def __init__(self):
        super().__init__('GAUSSIAN_BLUR', GaussianBlur.PARAMETERS)

    def process(self, data_set):
        m, w, h, c = data_set.X.shape
        x = np.zeros((2 * m, w, h, c))
        y = np.zeros((2 * m))
        for i in range(m):
            single_image = data_set.X[i, :]
            x[i, :] = single_image
            blurred = cv2.blur(single_image.squeeze(), self.parameters['ksize'], self.parameters['sigma'])
            x[m + i, :] = blurred.reshape((w, h, c))
            y[i] = data_set.y[i]
            y[m + i] = data_set.y[i]
        return DataSet(data_set.name, x, y, data_set.count)


class DataShuffler(PreProcessingStep):
    def __init__(self):
        super().__init__('DATA_SHUFFLER')

    def process(self, data_set):
        x, y = utils.shuffle(data_set.X, data_set.y)
        return DataSet(data_set.name, x, y, data_set.count)
