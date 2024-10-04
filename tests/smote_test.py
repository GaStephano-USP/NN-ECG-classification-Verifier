import pytest
from sklearn.datasets import load_iris
from oversampling.smote import *
import numpy as np


def test_after_smote_bincount_is_equal(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote(x, y)

    assert is_bincount_equal(y)


def test_after_smote_there_is_right_number_of_examples_in_dataset(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote(x, y)

    assert len(np.argwhere(y == 0)) == 30
    assert len(np.argwhere(y == 1)) == 30
    assert len(np.argwhere(y == 2)) == 30


def test_after_smote_shape_of_dataset_is_correct(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote(x, y)

    assert x.shape == (90, 4, 1)
    assert y.shape == (90,)


def test_after_smote_ndim(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote(x, y)
    assert x.ndim == 3
    assert y.ndim == 1


def test_after_smote_with_augmentation_bincount_is_equal(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote_with_data_agumentation(x, y)

    assert is_bincount_equal(y)


def test_after_smote_with_augmentation_there_is_right_number_of_examples_in_dataset(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote_with_data_agumentation(x, y)

    assert len(np.argwhere(y == 0)) == 30
    assert len(np.argwhere(y == 1)) == 30
    assert len(np.argwhere(y == 2)) == 30


def test_after_smote_with_augmentation_shape_of_dataset_is_correct(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote_with_data_agumentation(x, y)

    assert x.shape == (90, 4, 1)
    assert y.shape == (90,)


def test_after_smote_with_augmentation_ndim(imbalanced_iris):
    x, y = imbalanced_iris
    assert not is_bincount_equal(y)

    x, y = smote_with_data_agumentation(x, y)
    assert x.ndim == 3
    assert y.ndim == 1


@pytest.fixture
def imbalanced_iris():
    x, y = load_iris(return_X_y=True)
    x = np.vstack([x[:30], x[50:70], x[100:110]])
    y = np.hstack([y[:30], y[50:70], y[100:110]])
    x = np.expand_dims(x, axis=2)

    return x, y


def is_bincount_equal(y):
    bin = np.bincount(y)
    comparison = np.array([bin[i-1] == bin[i] for i in range(1, len(bin))])
    return comparison.all()
