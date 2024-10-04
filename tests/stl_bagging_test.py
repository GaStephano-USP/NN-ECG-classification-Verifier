import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists

from stl_bagging import remove_zero_padding


def get_padded_sequence(non_zero_len, zeros_len):
    return np.hstack([np.random.rand(non_zero_len), np.zeros((zeros_len,))])


@given(lists(integers(min_value=1, max_value=4096), min_size=10, max_size=10),
       lists(integers(min_value=1, max_value=4096), min_size=10, max_size=10))
def test_remove_padding_sequence_has_corect_shape_after_remove(non_zero_len_list, zeros_len_list):
    for non_zero_len, zeros_len in zip(non_zero_len_list, zeros_len_list):
        sequence = get_padded_sequence(non_zero_len, zeros_len)
        sequence_without_padding, _ = remove_zero_padding(sequence)
        assert len(sequence_without_padding) == non_zero_len


@given(lists(integers(min_value=1, max_value=4096), min_size=10, max_size=10),
       lists(integers(min_value=1, max_value=4096), min_size=10, max_size=10))
def test_remove_padding_2_zeros_are_removed(non_zero_len_list, zeros_len_list):
    for non_zero_len, zeros_len in zip(non_zero_len_list, zeros_len_list):
        sequence = get_padded_sequence(non_zero_len, zeros_len)
        _, num_zeros_removed = remove_zero_padding(sequence)
        assert num_zeros_removed == zeros_len
