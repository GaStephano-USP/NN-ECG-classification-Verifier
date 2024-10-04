import pandas as pd

from dataset_utils.utils import *


def load_data(filename):
    dataframe = pd.read_csv(filename, engine='python')
    dataset = dataframe.values
    return dataset.astype('float32')


def load_file(filename):
    train = load_data(filename)
    input = train[:, :-1]
    target = train[:, -1]

    return expand_dims(input, target)
