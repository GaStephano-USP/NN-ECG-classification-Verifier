import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as ps
from scipy.signal import resample


def plot_first_4_samples(trainX):
    plt.figure(2, figsize=(10, 6))
    plt.plot(trainX[0], label='learning example 0')
    plt.plot(trainX[1], label='learning example 1')
    plt.plot(trainX[2], label='learning example 2')
    plt.plot(trainX[3], label='learning example 3')
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('ECG signal')


def plot_first_examples_from_each_class(trainX, trainY):
    plt.figure(3, figsize=(10, 6))
    plt.plot(trainX[0], label='class 1')

    train_counts = np.bincount(trainY.astype('int64').flatten())
    plt.plot(trainX[train_counts[0]+1], label='class 2')
    plt.plot(trainX[train_counts[1]+1], label='class 3')
    plt.plot(trainX[train_counts[2]+1], label='class 4')
    plt.plot(trainX[train_counts[3]+1], label='class 5')
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('ECG signal')


def plot_simple_ovesampling(trainX):
    plt.figure(4, figsize=(10, 6))
    plt.plot(trainX[0, :], label='original signal')
    plt.plot(amplify(trainX[0, :]), label='amplified signal')
    plt.plot(stretch(trainX[0, :]), label='stretched signal')
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('ECG signal')


def stretch(x):
    sum_samples = int(187 * (1 + (np.random.rand()-0.5)/3))
    y = resample(x, sum_samples)
    if sum_samples < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:sum_samples] = y
    else:
        y_ = y[:187]
    return y_


def amplify(x):
    alpha = (np.random.rand()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor


def augment(x):
    result = np.zeros(shape=(3, 187))
    result[0, :] = stretch(x)
    result[1, :] = amplify(x)
    result[2, :] = amplify(stretch(x))
    return result


if __name__ == '__main__':
    PACKAGE_PARENT = '../'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(
        os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from dataset_utils.read_MIT_dataset import *

    imbalance_analysis('dataset/mitbih_train.csv')

    trainX, trainY, testX, testY = load_testing_dataset()
    plot_first_4_samples(trainX)
    plot_first_examples_from_each_class(trainX, trainY)
    plot_simple_ovesampling(trainX)
    plt.show()
