import numpy as np
from oversampling.GeneratorBase import *


def reduce_imbalance(dataX, dataY, examples_generator, num_examples=6000):
    resultX = []
    resultY = []

    for class_x, class_y in get_classes_examples(dataX, dataY):
        x = oversample_or_undersample(
            class_x, class_y, examples_generator, num_examples)
        resultX.append(x)
        y = np.ones((num_examples,)) * class_y[0]
        resultY.append(y)

    return np.vstack(resultX), np.hstack(resultY)


def oversample_or_undersample(class_x, class_y, examples_generator, num_examples):
    num_class_samples = class_x.shape[0]
    if num_class_samples >= num_examples:  # undersampling
        x = GeneratorBase.choose_n_samples(class_x, num_examples)
    elif examples_generator is not None:  # oversampling
        x = examples_generator.generate(class_x, class_y, num_examples)
    else:
        x = class_x
    return x


def get_classes_examples(dataX, dataY):
    num_classes = 5
    for i in range(num_classes):
        yield get_class_examples(dataX, dataY, i)


def get_class_examples(dataX, dataY, class_index):
    indexes = get_class_indexes(dataY, class_index)
    return dataX[indexes], dataY[indexes]


def get_class_indexes(dataY, class_index):
    return np.argwhere(dataY.flatten() == class_index).flatten()
