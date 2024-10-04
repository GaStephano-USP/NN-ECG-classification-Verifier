from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from base_model import *
from dataset_utils.read_MIT_dataset import *
from ensamble.fusion_methods import *
from ensamble.umce import *
from experiment_utils.metrics import *

if __name__ == '__main__':
    num_classes = 5
    x, y = load_whole_dataset()

    sets_shapes_report(x, y)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    input_shape = x.shape[1:]

    acc, precision, recall, f1 = [], [], [], []

    for fold_number, (train_index, test_index) in enumerate(kf.split(x, y)):
        print("fold ", fold_number+1)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # prepare data for traning
        #sets_shapes_report(x_train, y_train)
        #sets_shapes_report(x_test, y_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # sample and train model

        def sample_base_model():
            return BaseModel(input_shape, batch_size=32)
        umce = MuticlassUMCE(sample_base_model, avrg_fusion)
        umce.fit(x_train, y_train)

        # evaluate metrics
        y_pred = umce.predict(x_test)
        a, p, r, f = metrics_values(y_pred, y_test)
        acc.append(a)
        precision.append(p)
        recall.append(r)
        f1.append(f)

        print("\n\n")

    precision = np.vstack(precision)
    recall = np.vstack(recall)
    f1 = np.vstack(f1)
    metrics_report(acc, precision, recall, f1)
