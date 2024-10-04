import numpy as np
import pandas as pd  # Missing statement - added during Safety ArtISt analysis.

def expand_dims(input, target):
    # input = np.expand_dims(input, axis=3)
    # Issue detected with Safety ArtISt - Since axis = 3 > input.ndim = 2, np.expand_dims() will not work anymore starting on NumPy 1.18.
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    # Workaround: Since axis = 3 > input.ndim = 2, np.expand_dims() would work equivalently to making axis = 2. Hence, the line of code has been rewritten to make it clear.
    input = np.expand_dims(input, axis=2)
    # target = np.expand_dims(target, axis=2)
    # Issue detected with Safety ArtISt - Since axis = 2 > target.ndim = 1, np.expand_dims() will not work anymore starting on NumPy 1.18.
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
    # Workaround: Since axis = 2 > target.ndim = 1, np.expand_dims() would work equivalently to making axis = 1. Hence, the line of code has been rewritten to make it clear.
    target = np.expand_dims(target, axis=1)
    return input, target


def imbalance_analysis(filename):
    df_train = pd.read_csv(filename)

    target_count = df_train.iloc[:, -1].value_counts()
    print("Number of traning examples in each class:")
    for i in range(len(target_count)):
        print('Class {}:'.format(i), target_count[i])

    print("Imbalance ratio for classes 1-4:")
    for i in range(1, len(target_count)):
        print('IR(class {}) ='.format(i), round(
            target_count[0] / target_count[i], 2), ': 1')

    target_count.plot(kind='bar', title='Count (target)')


def sets_shapes_report(x_train, y_train):
    print("train set shape: ", x_train.shape)
    print("train set shape: ", y_train.shape)

    print("train classes sample count:")
    print(np.bincount(y_train.astype('int32').flatten()))
