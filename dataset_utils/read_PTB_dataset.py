import numpy as np
from sklearn.model_selection import train_test_split

from dataset_utils.load_data import *


def load_testing_dataset():
    normalX, normalY = load_file('dataset/ptbdb_normal.csv')
    abnormalX, abnormalY = load_file('dataset/ptbdb_abnormal.csv')

    trainX = np.concatenate([normalX, abnormalX], axis=0)
    trainY = np.concatenate([normalY, abnormalY], axis=0)

    trainX, testX, trainY, testY = train_test_split(
        trainX, trainY, test_size=0.3, random_state=42, stratify=trainY)

    return trainX, trainY, testX, testY


def load_validation_dataset(split=0.3):
    trainX, trainY, _, _ = load_testing_dataset()
    trainX, validX, trainY, validY = train_test_split(
        trainX, trainY, test_size=split, random_state=42, stratify=trainY)

    return trainX, trainY, validX, validY
