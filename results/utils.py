import numpy as np


def save_all(folds: list, folder: str):
    for i, fold in enumerate(folds):
        filename = folder + "/fold" + str(i) + ".npy"
        np.save(filename, fold)
        print("saved to ", filename)
