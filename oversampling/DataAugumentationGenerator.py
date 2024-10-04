import numpy as np
from scipy.signal import resample

from oversampling.GeneratorBase import *


class DataAugumentationGenerator(GeneratorBase):

    def generate(self, dataX, dataY, examples_to_generate):
        while dataX.shape[0] < examples_to_generate:
            additional_samples = np.apply_along_axis(
                augment, axis=1, arr=dataX).reshape(-1, 187, 1)
            dataX = np.vstack([dataX, additional_samples])
        return self.choose_n_samples(dataX, examples_to_generate)

    @staticmethod
    def augment(self, x):
        result = np.zeros(shape=(3, 187))
        result[0, :] = stretch(x)
        result[1, :] = amplify(x)
        result[2, :] = amplify(stretch(x))
        return result

    @staticmethod
    def stretch(x):
        signal_len = len(x)
        sum_samples = int(signal_len * (1 + (np.random.rand()-0.5)/3))
        y = resample(x, sum_samples)
        if sum_samples < signal_len:
            y_ = np.zeros(shape=(signal_len, ))
            y_[:sum_samples] = y
        else:
            y_ = y[:signal_len]
        return y_

    @staticmethod
    def amplify(x):
        alpha = (np.random.rand()-0.5)
        factor = -alpha*x + (1+alpha)
        return x*factor
