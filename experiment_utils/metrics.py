from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import numpy as np


def metrics_values(y_pred, y_true):
    y_pred_labels = y_pred.argmax(axis=1)
    y_true_labels = y_true.argmax(axis=1)

    classif_report = classification_report(y_true_labels, y_pred_labels)
    print(classif_report)

    acc = accuracy_score(y_true_labels, y_pred_labels)

    labels = unique_labels(y_true_labels, y_pred_labels)
    p, r, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=None)
    return acc, p, r, f1


def metrics_report(acc, precision, recall, f1):
    print("avrg results for all folds:")
    print("acc")
    print(np.mean(acc))
    print("precision")
    avrg_precision_all_labels = np.mean(precision, axis=0)
    print(avrg_precision_all_labels)
    print(np.mean(avrg_precision_all_labels))
    print("recall")
    avrg_recall_all_labels = np.mean(recall, axis=0)
    print(avrg_recall_all_labels)
    print(np.mean(avrg_recall_all_labels))
    print("f1")
    avrg_f1_all_labels = np.mean(f1, axis=0)
    print(avrg_f1_all_labels)
    print(np.mean(avrg_f1_all_labels))
