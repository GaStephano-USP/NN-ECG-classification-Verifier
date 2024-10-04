import numpy as np

from oversampling.GeneratorBase import *
from oversampling.stl_bagging import *


class StlBaggingGenerator(GeneratorBase):

    def generate(self, dataX, dataY, num_examples_we_want):
        num_examples = dataX.shape[0]
        examples_to_generate = (num_examples_we_want - num_examples)
        examples_per_signal = examples_to_generate // num_examples
        rest = examples_to_generate - num_examples * examples_per_signal
        assert examples_per_signal * num_examples + \
            num_examples + rest == num_examples_we_want

        result = np.copy(dataX)
        for i in range(num_examples):
            if i < rest:
                generate = examples_per_signal+1
            else:
                generate = examples_per_signal
            if generate > 0:
                additional_samples = stl_bagging(
                    dataX[i], num_samples=generate)
                additional_samples = np.expand_dims(additional_samples, axis=2)
                result = np.vstack([result, additional_samples])
        return self.sanitize(result)

    def sanitize(self, x):
        indexes = np.argwhere(np.isnan(x) == True)
        for i in indexes:
            if i[1] == 0:  # left-most point
                p1 = (1, x[i[0], i[1]+1, 0])
                p2 = (2, x[i[0], i[1]+2, 0])
            else:
                p1 = (i[1]-1, x[i[0], i[1]-1, 0])
                p2 = (i[1]+1, x[i[0], i[1]+1, 0])
            padding = self.regression_for_two_points_at(p1, p2, i[1])
            x[i[0], i[1], 0] = padding
        return x

    def regression_for_two_points_at(self, p1, p2, x):
        if np.isnan(p1[1]) or np.isnan(p2[1]):
            return 0
        a = (p1[1] - p2[1])/(p1[0]-p2[0])
        b = p1[1] - a * p1[0]
        return a * x + b
