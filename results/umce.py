import numpy as np
from utils import *

"""
learning for umce was stoped, but we can
calculate avrg metrics using results from folds

data is in format:
precision    recall  f1-score
in columns for classes 0-4 in rows
"""

fold1 = np.array([
    [1.00,      0.92,      0.95],
    [0.37,      0.92,      0.53],
    [0.88,      0.95,      0.91],
    [0.23,      0.91,      0.37],
    [0.96,      0.99,      0.97],
])

fold2 = np.array([
    [1.00,      0.92,      0.96],
    [0.37,      0.91,      0.52],
    [0.91,      0.95,      0.93],
    [0.25,      0.95,      0.39],
    [0.96,      0.99,      0.97],
])

fold3 = np.array([
    [1.00,      0.91,      0.95],
    [0.31,      0.92,      0.47],
    [0.85,      0.94,      0.90],
    [0.29,      0.95,      0.44],
    [0.98,      0.99,      0.99],
])

fold4 = np.array([
    [1.00,      0.92,      0.96],
    [0.38,      0.94,      0.54],
    [0.88,      0.95,      0.91],
    [0.26,      0.93,      0.41],
    [0.96,      0.99,      0.98],
])

fold5 = np.array([
    [1.00,      0.92,      0.96],
    [0.39,      0.95,      0.55],
    [0.88,      0.94,      0.91],
    [0.26,      0.95,      0.41],
    [0.97,      1.00,      0.98],
])

fold6 = np.array([
    [1.00,      0.92,      0.95],
    [0.39,      0.92,      0.54],
    [0.87,      0.95,      0.91],
    [0.23,      0.95,      0.37],
    [0.96,      0.99,      0.98],
])

fold7 = np.array([
    [1.00,      0.92,      0.96],
    [0.38,      0.88,      0.53],
    [0.88,      0.96,      0.91],
    [0.25,      0.93,      0.39],
    [0.96,      0.99,      0.97],
])

fold8 = np.array([
    [1.00,      0.92,      0.96],
    [0.38,      0.94,      0.54],
    [0.88,      0.96,      0.92],
    [0.24,      0.90,      0.38],
    [0.98,      0.99,      0.98],
])

fold9 = np.array([
    [1.00,      0.91,      0.95],
    [0.36,      0.90,      0.51],
    [0.88,      0.97,      0.92],
    [0.23,      0.97,      0.37],
    [0.97,      0.98,      0.97],
])

fold10 = np.array([
    [1.00,      0.92,      0.95],
    [0.36,      0.91,      0.51],
    [0.87,      0.94,      0.91],
    [0.26,      0.96,      0.41],
    [0.97,      0.98,      0.98],
])


all_folds = [
    fold1,
    fold2,
    fold3,
    fold4,
    fold5,
    fold6,
    fold7,
    fold8,
    fold9,
    fold10,
]

save_all(all_folds, 'umce')


fold1_avrg = np.mean(fold1, axis=0)
fold2_avrg = np.mean(fold2, axis=0)
fold3_avrg = np.mean(fold3, axis=0)
fold4_avrg = np.mean(fold4, axis=0)
fold5_avrg = np.mean(fold5, axis=0)
fold6_avrg = np.mean(fold6, axis=0)
fold7_avrg = np.mean(fold7, axis=0)
fold8_avrg = np.mean(fold8, axis=0)
fold9_avrg = np.mean(fold9, axis=0)
fold10_avrg = np.mean(fold10, axis=0)


print("fold1 avrg")
print(fold1_avrg)
print("fold2 avrg")
print(fold2_avrg)
print("fold3 avrg")
print(fold3_avrg)
print("fold4 avrg")
print(fold4_avrg)
print("fold5 avrg")
print(fold5_avrg)
print("fold6 avrg")
print(fold6_avrg)
print("fold7 avrg")
print(fold7_avrg)
print("fold8 avrg")
print(fold8_avrg)
print("fold9 avrg")
print(fold9_avrg)
print("fold10 avrg")
print(fold10_avrg)


all_avrg = np.vstack(
    [fold1_avrg,
     fold2_avrg,
     fold3_avrg,
     fold4_avrg,
     fold5_avrg,
     fold6_avrg,
     fold7_avrg,
     fold8_avrg,
     fold9_avrg,
     fold10_avrg])

print("average value of metrics:")
print("precision    recall  f1-score")
print(np.mean(all_avrg, axis=0))

all_folds = [fold1, fold2, fold3, fold4, fold5,
             fold6, fold7, fold8, fold9, fold10]
all_folds = list(map(lambda x: np.expand_dims(x, axis=2), all_folds))

all_folds = np.concatenate(all_folds, axis=2)

avrg_label_metrics = np.mean(all_folds, axis=2)
print("\navrg_label_metrics:")
print(avrg_label_metrics)
print("avrg acros labels:")
print(np.mean(avrg_label_metrics, axis=0))
