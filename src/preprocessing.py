#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from src.utils import DataSet


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    print(np.squeeze(gray).shape)
    print(rgb.shape)
    print(gray.shape)
    return gray


def normalise(data_set):
    x = data_set.X / 255 - 0.5
    return DataSet(data_set.name, x, data_set.y, data_set.count)
