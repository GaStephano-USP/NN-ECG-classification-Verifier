# inspired by https://www.kaggle.com/code/justuser/mnist-with-pytorch-fully-connected-network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import random
from numpy.random import RandomState
from torch.utils.data import Subset
from torch.autograd import Variable
import copy

class FullyConnected(nn.Module):  # inherits nn.Module

    def __init__(self, input_size, num_classes, hidden_size):  # input size = 28x28 = 784 for mnist
        super(FullyConnected, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, mode="max", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif (self.mode == "max" and score < self.best_score + self.delta) or \
             (self.mode == "min" and score > self.best_score - self.delta):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


# hyperparameters
input_size = 784
output_size = 4
hidden_size = 50

epochs = 1000
batch_size = 50
learning_rate = 0.01
loss_func = nn.CrossEntropyLoss()
early_stopper = EarlyStopping(patience=10, mode="max")

def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        optimizer.zero_grad()
        target = target.squeeze().long()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './trained_models/Fracture_FC_Net/Fracture_FC_Net.pth')
    if display:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()
def test(model, device, test_loader, name="\nVal"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.squeeze().long()
            test_loss += loss_func(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(name, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
# preprocessing
data_flag = 'octmnist'

download = True

info = INFO[data_flag]
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[.5], std=[.5]),
      ])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from random import randint

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

loss_val = []
accs_val = []
seed = randint(0,50)

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)

print(f'Num Samples For Training: {len(train_dataset)}, Validation: {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = FullyConnected(input_size, output_size, hidden_size)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)

for epoch in range(epochs):
    l = train(model, device, train_loader, optimizer, epoch, display=epoch%5==0)
    loss_val.append(l)
    acc = test(model, device, val_loader)
    accs_val.append(acc)
    early_stopper(acc, model)
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(early_stopper.best_model_state)

torch.save(model.state_dict(), './trained_models/Fracture_FC_Net/Fracture_FC_Net.pth')