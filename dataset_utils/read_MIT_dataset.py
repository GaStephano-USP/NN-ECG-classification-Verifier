import numpy as np
from sklearn.model_selection import train_test_split

from dataset_utils.load_data import *


def load_testing_dataset():
    trainX, trainY = load_file('dataset/mitbih_train.csv')
    testX, testY = load_file('dataset/mitbih_test.csv')

    return trainX, trainY, testX, testY


def load_validation_dataset(split=0.3):
    trainX, trainY = load_file('dataset/mitbih_train.csv')
    trainX, validX, trainY, validY = train_test_split(
        trainX, trainY, test_size=split, random_state=42, stratify=trainY)

    return trainX, trainY, validX, validY


def load_whole_dataset():
    trainX, trainY, testX, testY = load_testing_dataset()
    return np.vstack([trainX, testX]), np.vstack([trainY, testY])
