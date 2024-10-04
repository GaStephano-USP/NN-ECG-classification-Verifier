import numpy as np
import pytest


class OversamplingTestBase(object):

    @classmethod
    def setup(cls):
        try:
            cls.result_400
        except AttributeError:
            cls.result_400 = cls.get_dataset(400)
            cls.result_3000 = cls.get_dataset(3000)
            cls.result_6000 = cls.get_dataset(6000)

    def test_400_samples_number_of_learning_examples_for_all_classes_is_equal(self):
        _, y_aug = self.result_400
        self.check_if_bincount_is_equal(y_aug)

    def test_400_samples_number_of_learning_examples_is_rgiht(self):
        _, y_aug = self.result_400
        self.check_if_bincount_have_right_number_of_examples(y_aug, 400)

    def test_400_samples_number_of_learning_examples_is_the_same_for_x_and_y(self):
        x_aug, y_aug = self.result_400
        assert x_aug.shape[0] == y_aug.shape[0]

    def test_400_samples_no_nans_or_infs(self):
        x_aug, y_aug = self.result_400
        self.checks_for_nans_and_inf(x_aug)
        self.checks_for_nans_and_inf(y_aug)

    def test_3000_samples_number_of_learning_examples_for_all_classes_is_equal(self):
        _, y_aug = self.result_3000
        self.check_if_bincount_is_equal(y_aug)

    def test_3000_samples_number_of_learning_examples_is_rgiht(self):
        _, y_aug = self.result_3000
        self.check_if_bincount_have_right_number_of_examples(y_aug, 3000)

    def test_3000_samples_number_of_learning_examples_is_the_same_for_x_and_y(self):
        x_aug, y_aug = self.result_3000
        assert x_aug.shape[0] == y_aug.shape[0]

    def test_3000_samples_no_nans_or_infs(self):
        x_aug, y_aug = self.result_3000

        self.checks_for_nans_and_inf(x_aug)
        self.checks_for_nans_and_inf(y_aug)

    def test_6000_samples_number_of_learning_examples_for_all_classes_is_equal(self):
        _, y_aug = self.result_6000
        self.check_if_bincount_is_equal(y_aug)

    def test_6000_samples_number_of_learning_examples_is_rgiht(self):
        _, y_aug = self.result_6000
        self.check_if_bincount_have_right_number_of_examples(y_aug, 6000)

    def test_6000_samples_number_of_learning_examples_is_the_same_for_x_and_y(self):
        x_aug, y_aug = self.result_6000
        assert x_aug.shape[0] == y_aug.shape[0]

    def print_bincount(self, labels):
        print("bincount = {}".format(self.bincount(labels)))

    def bincount(self, labels):
        return np.bincount(labels.astype('int64').flatten())

    def check_if_bincount_is_equal(self, labels):
        bin = self.bincount(labels)
        for i in range(1, len(bin)):
            assert bin[i-1] == bin[i]

    def check_if_bincount_have_right_number_of_examples(self, labels, num_examples):
        bin = self.bincount(labels)
        for b in bin:
            assert b == num_examples

    def checks_for_nans_and_inf(self, x):
        assert not np.isnan(x).any()
        assert not np.isinf(x).any()
        assert np.isfinite(x).all()
