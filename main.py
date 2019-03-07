#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src import loading, preprocessing

training, validation, test = loading.load_all()

training, validation, test = loading.load_all()
preprocessors = [
    preprocessing.DataShuffler(),
    preprocessing.MinMaxNormaliser(),
    preprocessing.GrayScaleConverter(),
    preprocessing.GaussianBlurAugmenter()
]
training, validation, test = tuple(
    preprocessing.Processor.apply(d, preprocessors) for d in [training, validation, test])
print(training.X[0].shape)
from matplotlib import pyplot as plt

plt.imshow(training.X[0].squeeze())