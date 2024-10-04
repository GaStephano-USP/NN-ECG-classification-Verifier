import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from stl_bagging import *


def load_data(filename):
    dataframe = read_csv(filename, engine='python')
    dataset = dataframe.values
    return dataset.astype('float32')


def show_samples(samples):
    for i, sample in enumerate(samples):
        plt.plot(sample, label='example'+str(i))
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('ECG signal')


trainX = load_data('dataset/mitbih_train.csv')[:, :-1]
trainX = np.expand_dims(trainX, axis=2)

plt.figure(1, figsize=(10, 6))
show_samples(trainX[:4])

addtional_samples = stl_bagging(trainX[1])

plt.figure(2, figsize=(10, 6))
show_samples(addtional_samples)
plt.show()
