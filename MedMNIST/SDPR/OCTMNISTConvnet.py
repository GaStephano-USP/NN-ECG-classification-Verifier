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

import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
input_size = 784
output_size = 4
hidden_size = 50

epochs = 1000
batch_size = 50
learning_rate = 0.01
loss_func = nn.CrossEntropyLoss()


class RetinaMNISTCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # (1) Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1  # keeps 28x28
        )

        # (2) MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # (3–5) Convolutional layers
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 14x14
        self.conv3 = nn.Conv2d(32, 63, kernel_size=3, padding=1)   # 14x14
        self.conv4 = nn.Conv2d(63, 128, kernel_size=3, padding=1)  # 14x14

        # Second pooling to reach 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # (6–9) Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 6272)
        self.fc2 = nn.Linear(6272, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 50)

        # (10) Output layer
        self.out = nn.Linear(50, num_classes)

    def forward(self, x):
        # Conv + Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Conv stack
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Softmax output
        x = F.softmax(self.out(x), dim=1)
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
    torch.save(model.state_dict(), './trained_models/OCT_FC_Net/OCT_FC_Net.pth')
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
      # transforms.Normalize(mean=[.5], std=[.5]),
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
test_dataset = DataClass(split='test', transform=data_transform, download=download)

print(f'Num Samples For Training: {len(train_dataset)}, Validation: {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = RetinaMNISTCNN()
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

acc_test = test(model, device, test_loader)
print(acc_test*100)

torch.save(model.state_dict(), './trained_models/OCT_ConvNet/OCT_ConvNet.pth')

t_epochs, t_loss = list(zip(*train_losses))
t_epochs, t_acc = list(zip(*train_accuracy))
t_epochs, v_loss = list(zip(*train_vlosses))
t_epochs, v_acc = list(zip(*val_accuracy))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(t_epochs, t_loss) 
ax[0].plot(t_epochs, v_loss)
ax[0].set_title(f"Loss Curve (batch_size={batch_size}, lr={lr})")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend(["Trainamento", "Validação"])

ax[1].plot(t_epochs, t_acc) 
ax[1].plot(t_epochs, v_acc)
ax[1].set_title(f"Accuracy Curve (batch_size={batch_size}, lr={lr})")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend(["Trainamento", "Validação"])
fig.savefig("MedMNIST/SDCM/BreastMNISTResNet18/grafico.png")