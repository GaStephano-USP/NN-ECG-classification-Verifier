import numpy as np
import pytest
from ensamble.fusion_methods import *


def consistent_decisions():
    return [np.array([[0, 1, 0], [1, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0]])]


def test_avrg_fusion__consistent_decisions__returns_array_with_right_values():
    decisions = consistent_decisions()
    avrg = avrg_fusion(decisions)
    assert np.isclose(avrg[0], [0, 1, 0]).all()
    assert np.isclose(avrg[1], [1, 0, 0]).all()


def test_avrg_fusion__consistent_decisions__returns_array_with_valid_shape():
    decisions = consistent_decisions()
    avrg = avrg_fusion(decisions)
    assert avrg.shape == (2, 3)


def test_avrg_fusion__consistent_decisions__returns_array_with_valid_ndim():
    decisions = consistent_decisions()
    avrg = avrg_fusion(decisions)
    assert avrg.ndim == 2


def inconsistent_decisions():
    return [np.array([[0, 1, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 1]]),
            np.array([[0, 1, 0], [1, 0, 0]])]


def test_avrg_fusion__inconsistent_decisions__returns_array_with_right_values():
    decisions = inconsistent_decisions()
    avrg = avrg_fusion(decisions)
    assert np.isclose(avrg[0], [0, 0.66, 0.33], atol=0.1).all()
    assert np.isclose(avrg[1], [0.66, 0, 0.33], atol=0.1).all()


def test_avrg_fusion__inconsistent_decisions__returns_array_with_valid_shape():
    decisions = inconsistent_decisions()
    avrg = avrg_fusion(decisions)
    assert avrg.shape == (2, 3)


def test_avrg_fusion__inconsistent_decisions__returns_array_with_valid_ndim():
    decisions = inconsistent_decisions()
    avrg = avrg_fusion(decisions)
    assert avrg.ndim == 2
