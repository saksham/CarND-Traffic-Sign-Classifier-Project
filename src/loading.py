#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections import namedtuple

DataSet = namedtuple('DataSet', ['name', 'X', 'y', 'count'])

TRAINING_DATA_SET_FILE = './data/traffic-signs-data/train.p'
VALIDATION_DATA_SET_FILE = './data/traffic-signs-data/valid.p'
TEST_DATA_SET_FILE = './data/traffic-signs-data/test.p'
SIGN_NAMES_CSV = './data/signnames.csv'


def load_all():
    """
    Loads training, validation and test datasets

    The dataset contains the following fields
    * 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height,
       channels).
    * 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id ->
       name mappings for each id.
    * 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
    * 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign
      in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32)
      OF THESE IMAGES

    :return: tuple of training, validation, test datasets
    """
    with open(TRAINING_DATA_SET_FILE, mode='rb') as f:
        train = pickle.load(f)
    with open(VALIDATION_DATA_SET_FILE, mode='rb') as f:
        valid = pickle.load(f)
    with open(TEST_DATA_SET_FILE, mode='rb') as f:
        test = pickle.load(f)
    read_sign_names_csv()

    training = DataSet('TRAINING', train['features'], train['labels'], len(train['labels']))
    validation = DataSet('VALIDATION', valid['features'], valid['labels'], len(valid['labels']))
    test = DataSet('TEST', test['features'], test['labels'], len(test['labels']))

    return training, validation, test


def read_sign_names_csv():
    result = {}
    with open(SIGN_NAMES_CSV, 'r') as f:
        line = f.readline()
        while line != '':
            line = f.readline()
            tokens = line.split(',')
            if len(tokens) == 2:
                result[int(tokens[0])] = tokens[1].strip()
    return result
