import onnxruntime as ort
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataset_utils.read_MIT_dataset import *
from base_model import *
from dataset_utils.read_MIT_dataset import *
from ensamble.fusion_methods import *
from ensamble.umce import *
from experiment_utils.metrics import *
from oversampling.reduce_imbalance import *

# Caminho para o modelo ONNX
model_path = "baseline.onnx"

# Crie uma sessão para o modelo ONNX
session = ort.InferenceSession(model_path)

# Obtenha o nome das entradas e saídas do modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Dados de entrada de exemplo (substitua com seus dados reais)

x, y = load_whole_dataset()
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
input_shape = x.shape[1:]
print(input_shape)
acc, precision, recall, f1 = [], [], [], []
for fold_number, (train_index, test_index) in enumerate(kf.split(x, y)):
        print("fold ", fold_number+1)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # prepare data for traning
        num_undersample = np.min(np.bincount(
            y_train.astype('int16').flatten()))
        x_train, y_train = reduce_imbalance(
            x_train, y_train, None, num_examples=num_undersample)  # No oversampling technique
        #sets_shapes_report(x_train, y_train)
        #sets_shapes_report(x_test, y_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

input_data = x_test
predictions = []
correct = []
# Faça a inferência
output = session.run([output_name], {input_name: input_data})

# Mostre a saída

pred_class = np.argmax(output[0], axis=1)
predictions.append(pred_class[0])
print("Saída do modelo:", pred_class)
print("Saídas corretas:", y_test)
correct_class = np.argmax(y_test, axis=1)
print("Saídas corretas:", correct_class)
correct.append(correct_class[0])
# Calcular a acurácia
accuracy = accuracy_score(correct, predictions)
print("Acurácia no conjunto de teste:", accuracy)