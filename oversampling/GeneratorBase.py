from abc import abstractmethod

import numpy as np


class GeneratorBase(object):

    @abstractmethod
    def generate(self, dataX, dataY, examples_to_generate):
        raise AttributeError()

    @staticmethod
    def choose_n_samples(dataX, n):
        all_indexes = np.arange(dataX.shape[0])
        chosen = np.random.choice(all_indexes, n, replace=False)
        del all_indexes

        return dataX[chosen]
