from hypothesis import given
from hypothesis.strategies import integers, lists
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from ensamble.umce import *
from ensamble.fusion_methods import *


class TestUMCE(object):

    class ModelMock(object):

        def __init__(self):
            self.dummy = DummyClassifier()
            self.how_many_times_fit_was_called = 0
            self.how_many_times_predict_was_called = 0

        def fit(self, x_train, y_train):
            self.how_many_times_fit_was_called += 1
            self.dummy.fit(x_train, y_train)

        def predict(self, x_test):
            self.how_many_times_predict_was_called += 1
            return self.dummy.predict(x_test)

    def setup(cls):
        cls.mock = TestUMCE.ModelMock()

    def simple_dataset(self):
        all_samples = 30
        x = np.arange(all_samples)
        x = x.reshape((all_samples, 1))

        class_a = np.full((5,), 0)
        class_b = np.full((10,), 1)
        class_c = np.full((15,), 2)
        y = np.hstack([class_a, class_b, class_c])

        assert x.shape == (all_samples, 1)
        assert y.shape == (all_samples,)

        return x, y

    def custom_dataset(self, num_samples_in_classes, num_features):
        all_samples = sum(num_samples_in_classes)
        x = np.arange(all_samples*num_features)
        x = x.reshape((all_samples, num_features))

        lables = [np.full((num_samples,), label)
                  for label, num_samples in enumerate(num_samples_in_classes)]
        y = np.hstack(lables)

        assert x.shape == (all_samples, num_features)
        assert y.shape == (all_samples,)

        return x, y

    def sample_dummy_classifier(self):
        self.dummy = DummyClassifier()
        return self.dummy

    def sample_mock(self):
        return self.mock

    def test_get_samples_in_classes_for_simple_dataset_all_returned_y_arrays_contain_only_one_label(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        for y in y_classes:
            array_from_first_label = np.full(y.shape, y[0])
            assert np.array_equal(y, array_from_first_label)

    def test_get_samples_in_classes_for_simple_dataset_all_returned_arrays_have_valid_ndim(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_classes, y_classes = sut.get_separate_classes(x_train, y_train)
        for x in x_classes:
            assert x.ndim == 2
        for y in y_classes:
            assert y.ndim == 1

    def test_get_samples_in_classes_for_simple_dataset_all_returned_x_y_have_same_number_of_examples(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_train, y_classes = sut.get_separate_classes(x_train, y_train)
        for x, y in zip(x_train, y_classes):
            assert x.shape[0] == y.shape[0]

    def test_get_samples_in_classes_for_simple_dataset_all_classes_found(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        assert len(y_classes) == 3

    def test_get_num_samples_in_each_class_returns_right_number(self):
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        arg = [np.array([0, 0, 0]), np.array([1, 1])]
        num_samples = sut.get_num_samples_in_each_class(arg)
        assert num_samples == [3, 2]

    def test_after_fit_simple_dataset_num_classifiers_is_3(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        sut.fit(x_train, y_train)
        assert sut.num_classifiers == 3

    def test_k_fold_for_each_class_returns_right_number_of_folds(self):
        arg_x = [np.array([0, 1, 2, 3]).reshape(4, 1),
                 np.array([4, 5]).reshape(2, 1)]
        arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
        arg_k_i = [2, 1]

        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)
        class_0_folds_x, class_1_folds_x = folds_x
        class_0_folds_y, class_1_folds_y = folds_y

        assert len(class_0_folds_x) == 2
        assert len(class_0_folds_y) == 2
        assert len(class_1_folds_x) == 1
        assert len(class_1_folds_y) == 1

    def test_k_fold_for_each_class_number_samples_in_each_fold_close_to_minimal_num_of_samples(self):
        arg_x = [np.array([0, 1, 2, 3]).reshape(4, 1),
                 np.array([4, 5]).reshape(2, 1)]
        arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
        arg_k_i = [2, 1]
        minimal = min(list(map(lambda x: len(x), arg_y)))

        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)

        for class_folds_lists in folds_x:
            for fold in class_folds_lists:
                # atol due to floor in during calculation of k_i
                assert np.isclose(len(fold), minimal, atol=1)
        for class_folds_lists in folds_y:
            for fold in class_folds_lists:
                assert np.isclose(len(fold), minimal, atol=1)

    def test_k_fold_for_each_class_number_each_fold_for_x_and_y_have_the_same_len(self):
        arg_x = [np.array([0, 1, 2, 3]).reshape(4, 1),
                 np.array([4, 5]).reshape(2, 1)]
        arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
        arg_k_i = [2, 1]

        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)

        for class_folds_x_list, class_folds_y_list in zip(folds_x, folds_y):
            for fold_x, fold_y in zip(class_folds_x_list, class_folds_y_list):
                assert len(fold_x) == len(fold_y)

    def test_k_fold_for_each_class_ndim_of_folds_is_correct(self):
        arg_x = [np.array([0, 1, 2, 3]).reshape(4, 1),
                 np.array([4, 5]).reshape(2, 1)]
        arg_y = [np.array([0, 0, 0, 0]), np.array([1, 1])]
        arg_k_i = [2, 1]

        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        folds_x, folds_y = sut.k_fold_for_each_class(arg_k_i, arg_x, arg_y)

        for class_folds_lists in folds_x:
            for fold in class_folds_lists:
                assert fold.ndim == 2
        for class_folds_lists in folds_y:
            for fold in class_folds_lists:
                assert fold.ndim == 1

    def test_simple_dataset_after_fit_dummy_classifiers_are_fitted(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        sut.fit(x_train, y_train)
        check_is_fitted(self.dummy, 'classes_')

    def test_simple_dataset_after_predict_reutrns_array_with_valid_size(self):
        x_train, y_train = self.simple_dataset()
        x_test, _ = self.simple_dataset()

        sut = MuticlassUMCE(self.sample_dummy_classifier, avrg_fusion)
        sut.fit(x_train, y_train)
        y_pred = sut.predict(x_test)
        assert y_pred.shape == (30, 3)

    def test_ModelMock_fit_called_3_times(self):
        x_train, y_train = self.simple_dataset()
        sut = MuticlassUMCE(self.sample_mock, None)
        sut.fit(x_train, y_train)

        assert self.mock.how_many_times_fit_was_called == 3

    def test_ModelMock_predict_called_3_times(self):
        x_train, y_train = self.simple_dataset()
        x_test, _ = self.simple_dataset()

        sut = MuticlassUMCE(self.sample_mock, avrg_fusion)
        sut.fit(x_train, y_train)
        sut.predict(x_test)

        assert self.mock.how_many_times_predict_was_called == 3

    def test_get_samples_in_classes_all_returned_y_arrays_contain_only_one_label(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        for y in y_classes:
            array_from_first_label = np.full(y.shape, y[0])
            assert np.array_equal(y, array_from_first_label)

    def test_get_samples_in_classes_all_returned_arrays_have_valid_ndim(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_classes, y_classes = sut.get_separate_classes(x_train, y_train)
        for x in x_classes:
            assert x.ndim == 2
        for y in y_classes:
            assert y.ndim == 1

    def test_get_samples_in_classes_all_returned_x_y_have_same_number_of_examples(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_train, y_classes = sut.get_separate_classes(x_train, y_train)
        for x, y in zip(x_train, y_classes):
            assert x.shape[0] == y.shape[0]

    def test_get_samples_in_classes_all_classes_found(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        num_clases = len(class_counts)
        assert len(y_classes) == num_clases

    def test_after_fit_dummy_classifiers_are_fitted(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        sut.fit(x_train, y_train)
        check_is_fitted(self.dummy, 'classes_')

    def test_after_predict_reutrns_array_with_valid_size(self):
        class_counts = (5, 10, 15)
        num_features = 4
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        x_test, _ = self.simple_dataset()

        sut = MuticlassUMCE(self.sample_dummy_classifier, avrg_fusion)
        sut.fit(x_train, y_train)
        y_pred = sut.predict(x_test)
        assert y_pred.shape == (30, 3)

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_get_samples_in_classes_fuzzer_all_returned_y_arrays_contain_only_one_label(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        for y in y_classes:
            array_from_first_label = np.full(y.shape, y[0])
            assert np.array_equal(y, array_from_first_label)

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_get_samples_in_classes_fuzzer_all_returned_arrays_have_valid_ndim(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_classes, y_classes = sut.get_separate_classes(x_train, y_train)
        for x in x_classes:
            assert x.ndim == 2
        for y in y_classes:
            assert y.ndim == 1

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_get_samples_in_classes_fuzzer_all_returned_x_y_have_same_number_of_examples(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        x_train, y_classes = sut.get_separate_classes(x_train, y_train)
        for x, y in zip(x_train, y_classes):
            assert x.shape[0] == y.shape[0]

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_get_samples_in_classes_fuzzer_all_classes_found(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        _, y_classes = sut.get_separate_classes(x_train, y_train)
        num_clases = len(class_counts)
        assert len(y_classes) == num_clases

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_fuzzer_after_fit_dummy_classifiers_are_fitted(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        sut.fit(x_train, y_train)
        check_is_fitted(self.dummy, 'classes_')

    @given(lists(integers(min_value=1, max_value=512), min_size=2, max_size=10),
           integers(min_value=1, max_value=100))
    def test_fuzzer_after_predict_reutrns_array_with_valid_size(self, class_counts, num_features):
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        x_test, y_test = self.custom_dataset(class_counts, num_features)

        sut = MuticlassUMCE(self.sample_dummy_classifier, avrg_fusion)
        sut.fit(x_train, y_train)
        y_pred = sut.predict(x_test)
        assert y_pred.shape == to_categorical(y_test).shape

    # failed test cases found by fuzzer

    def test_single_samples_in_two_classes_after_predict_reutrns_array_with_valid_size(self):
        class_counts = [1, 1]
        num_features = 1
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        x_test, y_test = self.custom_dataset(class_counts, num_features)

        sut = MuticlassUMCE(self.sample_dummy_classifier, avrg_fusion)
        sut.fit(x_train, y_train)
        y_pred = sut.predict(x_test)
        assert y_pred.shape == to_categorical(y_test).shape

    # deadline exceded
    def test_many_samples_after_fit_dummy_classifiers_are_fitted(self):
        class_counts = [2557, 4061, 4061, 4061, 2305, 2305, 1]
        num_features = 2
        x_train, y_train = self.custom_dataset(class_counts, num_features)
        sut = MuticlassUMCE(self.sample_dummy_classifier, None)
        sut.fit(x_train, y_train)
        check_is_fitted(self.dummy, 'classes_')
