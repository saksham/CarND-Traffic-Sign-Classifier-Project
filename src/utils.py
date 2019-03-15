#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import logging.handlers
import math
import os
import sys
from abc import ABC, abstractmethod

import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import utils

from src import loading
from src.loading import DataSet

LOG_FILE_PATH = 'data/logs/runs.txt'
FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

_SIGN_LABELS_CACHE = {}


def get_logger(source, level=LOG_LEVEL):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.handlers.RotatingFileHandler(LOG_FILE_PATH, 'a', 10 * 1024 * 1024)
    file_handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(source)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)
    return logger


class ParameterizedProcessor(ABC):
    _logger = get_logger('Parameterized processor')

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


def get_sign_labels_map():
    global _SIGN_LABELS_CACHE
    if not _SIGN_LABELS_CACHE:
        _SIGN_LABELS_CACHE = loading.read_sign_names_csv()
    return _SIGN_LABELS_CACHE


def to_sign_label(code):
    return get_sign_labels_map().get(code)


def get_sign_labels_dataframe():
    as_list = list(get_sign_labels_map().items())
    df = pd.DataFrame.from_records(as_list)
    df.columns = ['code', 'label']
    return df


def group_labels_by_counts(data_set):
    y = pd.DataFrame(data_set.y)
    counts = pd.DataFrame(y[0].value_counts())
    counts.columns = ['counts']
    counts['code'] = counts.index.astype('int64')

    sign_labels = get_sign_labels_dataframe()
    return sign_labels.merge(counts, on='code')


def get_summary(data_sets):
    summary = {}
    all_classes = set()

    for data_set in data_sets:
        summary[data_set.name] = {
            'number-of-examples': data_set.count,
            'image-shape': data_set.X[0].shape,
            'no-of-classes': len(set(data_set.y))
        }
        all_classes = all_classes.union(set(data_set.y))
    summary['total-no-of-classes'] = len(all_classes)
    return summary


def read_image_for_lenet(image_path):
    """
    https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imread
    Apparently
    - cv2.imread() => BGR
    - matplotlib.image.imread() => RGB

    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_CUBIC)


def plot_and_save(images, labels, filepath, n_cols=3):
    n_rows = int(math.ceil(len(images) / float(n_cols)))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    fig.subplots_adjust(hspace=.2, wspace=.01)
    axs = axs.ravel()
    for i in range(n_rows * n_cols):
        axs[i].axis('off')
        if i < len(images):
            c_map = 'gray' if images[i].ndim == 2 else None
            axs[i].imshow(images[i], cmap=c_map)
            axs[i].set_title(labels[i])

    plt.savefig(filepath)


def shuffle(data_set):
    x, y = utils.shuffle(data_set.X, data_set.y)
    return DataSet(data_set.name, x, y, data_set.count)
