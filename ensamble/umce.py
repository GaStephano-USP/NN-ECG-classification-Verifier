from typing import List, Tuple, Callable

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import KFold


class MuticlassUMCE(object):

    def __init__(self, base_model_sampling: Callable, fusion_method: Callable):
        self.base_model_sampling = base_model_sampling
        self.fusion_method = fusion_method

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_classes, y_classes = self.get_separate_classes(x, y)
        num_samples_in_each_class = self.get_num_samples_in_each_class(
            y_classes)

        min_samples = min(num_samples_in_each_class)
        max_samples = max(num_samples_in_each_class)
        ir = max_samples // min_samples
        self.num_classifiers = min(10, ir)

        k_i = self.__imbalance_ratio_for_each_class(min_samples,
                                                    num_samples_in_each_class)

        folds_x, folds_y = self.k_fold_for_each_class(
            k_i, x_classes, y_classes)

        self.__create_models()
        self.__fit_all_models(k_i, folds_x, folds_y)

    def get_separate_classes(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        y = self.__convert_targets_if_needed(y)
        self.num_classes = y.max()+1  # labels usualy start from 0

        classes_x, classes_y = [], []
        for i in range(self.num_classes):
            class_indexes = np.argwhere(y == i).flatten()
            classes_x.append(x[class_indexes])
            classes_y.append(y[class_indexes])
        return classes_x, classes_y

    def __convert_targets_if_needed(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 2 and y.shape[1] > 1:
            y = self.__one_hot_to_labels(y)
        else:
            y = y.flatten()
        return y

    def __one_hot_to_labels(self, y: np.ndarray) -> np.ndarray:
        return y.argmax(axis=1)

    def get_num_samples_in_each_class(self, y: np.ndarray) -> List[int]:
        return list(map(lambda x: x.size, y))

    def __imbalance_ratio_for_each_class(self, min_samples: int, num_samples_in_each_class: List[int]) -> List[int]:
        return list(map(lambda x: x // min_samples,
                        num_samples_in_each_class))

    def k_fold_for_each_class(self, k_i: List[int], x_classes: List[np.ndarray], y_classes: List[np.ndarray]) -> Tuple[Tuple[List[np.ndarray]], Tuple[List[np.ndarray]]]:
        folds_x, folds_y = [], []
        for k, x_class, y_class in zip(k_i, x_classes, y_classes):
            self.__add_folds_for_class(k, x_class, y_class, folds_x, folds_y)

        return tuple(folds_x), tuple(folds_y)

    def __add_folds_for_class(self, k: int, x_class: np.ndarray, y_class: np.ndarray, folds_x: list, folds_y: list) -> None:
        folds_x.append([])
        folds_y.append([])
        if k > 1:
            kf = KFold(n_splits=k)
            for _, fold_index in kf.split(x_class):
                folds_x[-1].append(x_class[fold_index])
                folds_y[-1].append(y_class[fold_index])
        else:  # minority class
            folds_x[-1].append(x_class)
            folds_y[-1].append(y_class)

    def __create_models(self) -> None:
        self.models = [self.base_model_sampling()
                       for i in range(self.num_classifiers)]

    def __fit_all_models(self, k_i: List[int], folds_x: Tuple[List[np.ndarray]], folds_y: Tuple[List[np.ndarray]]) -> None:
        for i, base_model in enumerate(self.models):
            print("fitting model {}/{}".format(i+1, len(self.models)))
            x_train, y_train = self.__get_trainigset(k_i, folds_x, folds_y)
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            base_model.fit(x_train, y_train)

    def __get_trainigset(self, k_i: List[int], folds_x: Tuple[List[np.ndarray]], folds_y: Tuple[List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        x_train, y_train = [], []
        for k, folds_for_class_x, folds_for_class_y in zip(k_i, folds_x, folds_y):
            fold_index = self.__random_fold_index(k)
            x_train.append(folds_for_class_x[fold_index])
            y_train.append(folds_for_class_y[fold_index].flatten())
        return np.vstack(x_train), np.hstack(y_train)

    def __random_fold_index(self, k: int) -> int:
        if k == 1:
            return 0  # only one fold
        return np.random.randint(0, k-1)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        decisions = self.__get_decisions(x_test)
        return self.fusion_method(decisions)

    def __get_decisions(self, x_test: np.ndarray) -> List[np.ndarray]:
        return [base_model.predict(x_test) for base_model in self.models]
