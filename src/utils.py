#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

import pandas as pd

from src import loading

FORMAT = '%(levelname)s %(message)s'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

_SIGN_LABELS_CACHE = {}


def get_logger(source, level=LOG_LEVEL):
    logger = logging.getLogger(source)
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level)
    return logger


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
