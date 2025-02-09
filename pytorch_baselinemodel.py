import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_utils.read_MIT_dataset import *
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ensamble.fusion_methods import *
from ensamble.umce import *
from experiment_utils.metrics import *
from oversampling.reduce_imbalance import *

num_classes = 5
x, y = load_whole_dataset()

sets_shapes_report(x, y)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
input_shape = x.shape[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(input_shape)
acc, precision, recall, f1 = [], [], [], []

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, input_shape, num_classes=5):
        kernel_size = 5
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[0], 32, kernel_size=kernel_size, stride=1)
        
        self.resblock1 = ResidualBlock1D(32, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        
        self.resblock2 = ResidualBlock1D(32, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        
        self.resblock3 = ResidualBlock1D(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        
        self.resblock4 = ResidualBlock1D(64, 64)
        self.pool4 = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        
        self.resblock5 = ResidualBlock1D(64, 128)
        self.pool5 = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 512)  # Corrigido com o tamanho dinâmico
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
    
    # def _forward_features(self, x):
    #     x = self.conv1(x)
    #     x = self.resblock1(x)
    #     x = self.pool1(x)
    #     x = self.resblock2(x)
    #     x = self.pool2(x)
    #     x = self.resblock3(x)
    #     x = self.pool3(x)
    #     x = self.resblock4(x)
    #     x = self.pool4(x)
    #     x = self.resblock5(x)
    #     x = self.pool5(x)
    #     x = self.flatten(x)
    #     return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        x = self.resblock4(x)
        x = self.pool4(x)
        x = self.resblock5(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return 


# # Definição do modelo e hiperparâmetros
model = ResNet1D(input_shape)
model_scripted = torch.jit.script(model)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Treinamento

# for fold_number, (train_index, test_index) in enumerate(kf.split(x, y)):
#     print("fold ", fold_number+1)
#     x_train, x_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#         # prepare data for traning
#     num_undersample = np.min(np.bincount(
#         y_train.astype('int16').flatten()))
#     x_train, y_train = reduce_imbalance(
#         x_train, y_train, None, num_examples=num_undersample)  # No oversampling technique
#         #sets_shapes_report(x_train, y_train)
#         #sets_shapes_report(x_test, y_test)
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)

#     x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
#     y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
#     x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
#     y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#     train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
#     # Inicializa o modelo e envia para o dispositivo
#     model.to(device)
#     model.train()
#         # sample and train model
#     model.train()
#     # Treinamento do modelo
#     epochs = 20  # Número de épocas
#     for epoch in range(epochs):
#         total_loss = 0.0  # Reinicia a perda a cada época

#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move para GPU/CPU

#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
model_scripted.save('model_scripted.pt')