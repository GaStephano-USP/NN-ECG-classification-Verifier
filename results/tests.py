import numpy as np
from scipy.stats import kruskal
from scikit_posthocs import posthoc_conover
from save_tex_table import *


def load_all(folder: str):
    result = []
    for i in range(0, 10):
        filename = folder + "/fold" + str(i) + ".npy"
        fold = np.load(filename)
        result.append(fold)
    return result


def folds_avrg(all_results):
    return list(map(lambda x: np.mean(x, axis=0), all_results))


def get_precision(results):
    return list(map(lambda x: x[0], results))


def get_recall(results):
    return list(map(lambda x: x[1], results))


def get_f1(results):
    return list(map(lambda x: x[2], results))


baseline = folds_avrg(load_all('baseline'))
umce = folds_avrg(load_all('umce'))
smote = folds_avrg(load_all('smote'))
smote_data_aug = folds_avrg(load_all('smote_data_aug'))


print("kruskal")
precision_stat, precision_p_val = kruskal(get_precision(
    baseline), get_precision(umce), get_precision(smote), get_precision(smote_data_aug))
recall_stat, recall_p_val = kruskal(get_recall(
    baseline), get_recall(umce), get_recall(smote), get_recall(smote_data_aug))
f1_stat, f1_p_val = kruskal(get_f1(
    baseline), get_f1(umce), get_f1(smote), get_f1(smote_data_aug))


print("precision: ", precision_stat, precision_p_val)
print("recall: ", recall_stat, recall_p_val)
print("f1: ", f1_stat, f1_p_val)

print("precision: ")
posthoc_precison = posthoc_conover(
    [get_precision(baseline), get_precision(umce), get_precision(smote), get_precision(smote_data_aug)])
print(posthoc_precison)

print("recall: ")
posthoc_recall = posthoc_conover(
    [get_recall(baseline), get_recall(umce), get_recall(smote), get_recall(smote_data_aug)])
print(posthoc_recall)

print("f1: ")
posthoc_f1 = posthoc_conover(
    [get_f1(baseline), get_f1(umce), get_f1(smote), get_f1(smote_data_aug)])
print(posthoc_f1)


kruskal_table = [['metric', 'H-value', 'p-value'],
                 ['precision', precision_stat, precision_p_val],
                 ['recall', recall_stat, recall_p_val],
                 ['f1-score', f1_stat, f1_p_val]]
save_tex_table(kruskal_table, 'kruskal_table')

posthoc_precision_table = [["\\", "undersampling", "UMCE", "SMOTE", "SMOTE with augmentation"],
                           ["undersampling", "-", posthoc_precison[0, 1],
                               posthoc_precison[0, 2], posthoc_precison[0, 3]],
                           ["UMCE", posthoc_precison[1, 0], "-",
                            posthoc_precison[1, 2], posthoc_precison[1, 3]],
                           ["SMOTE", posthoc_precison[2, 0],
                               posthoc_precison[2, 1], "-", posthoc_precison[2, 3]],
                           ["SMOTE with augmentation", posthoc_precison[3, 0],
                            posthoc_precison[3, 1], posthoc_precison[3, 2], "-"],
                           ]
save_tex_table(posthoc_precision_table, 'posthoc_precision_table')

posthoc_recall_table = [["\\", "undersampling", "UMCE", "SMOTE", "SMOTE with augmentation"],
                        ["undersampling", "-", posthoc_recall[0, 1],
                         posthoc_recall[0, 2], posthoc_recall[0, 3]],
                        ["UMCE", posthoc_recall[1, 0], "-",
                            posthoc_recall[1, 2], posthoc_recall[1, 3]],
                        ["SMOTE", posthoc_recall[2, 0],
                         posthoc_recall[2, 1], "-", posthoc_recall[2, 3]],
                        ["SMOTE with augmentation", posthoc_recall[3, 0],
                            posthoc_recall[3, 1], posthoc_recall[3, 2], "-"],
                        ]
save_tex_table(posthoc_recall_table, 'posthoc_recall_table')

posthoc_f1_table = [["\\", "undersampling", "UMCE", "SMOTE", "SMOTE with augmentation"],
                    ["undersampling", "-", posthoc_f1[0, 1],
                     posthoc_f1[0, 2], posthoc_f1[0, 3]],
                    ["UMCE", posthoc_f1[1, 0], "-",
                     posthoc_f1[1, 2], posthoc_f1[1, 3]],
                    ["SMOTE", posthoc_f1[2, 0],
                     posthoc_f1[2, 1], "-", posthoc_f1[2, 3]],
                    ["SMOTE with augmentation", posthoc_f1[3, 0],
                     posthoc_f1[3, 1], posthoc_f1[3, 2], "-"],
                    ]
save_tex_table(posthoc_f1_table, 'posthoc_f1_table')
