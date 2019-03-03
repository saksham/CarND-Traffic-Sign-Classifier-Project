#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import utils
from collections import namedtuple

SIGN_NAMES_CSV = './data/signnames.csv'
_SIGN_LABELS_MAP = {}

DataSet = namedtuple('DataSet', ['name', 'X', 'y', 'count'])


def read_sign_names_csv():
    with open(SIGN_NAMES_CSV, 'r') as f:
        line = f.readline()
        while line != '':
            line = f.readline()
            tokens = line.split(',')
            if len(tokens) == 2:
                _SIGN_LABELS_MAP[int(tokens[0])] = tokens[1].strip()


def get_sign_labels_map():
    if not _SIGN_LABELS_MAP:
        read_sign_names_csv()
    return _SIGN_LABELS_MAP


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


def shuffle_data_set(data_set):
    x, y = utils.shuffle(data_set.X, data_set.y)
    return DataSet(data_set.name, x, y, data_set.count)


read_sign_names_csv()
