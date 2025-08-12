# inspired by https://www.kaggle.com/code/justuser/mnist-with-pytorch-fully-connected-network
import torch
import torch.nn as nn  # all neural network modules
import torch.optim as optim  # optimization algo
import torch.nn.functional as F  # Functions with no parameters -> activation functions
from torch.utils.data import DataLoader  # easier dataset management, helps create mini batches
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import random
from numpy.random import RandomState
from torch.utils.data import Subset
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


# hyperparameters
input_size = 784
output_size = 1
hidden_size = 50

epochs = 20
batch_size = 50

def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target.float())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './trained_models/PneumoniaMNIST/PnuemoniaMNISTFCNet.pth')
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
            test_loss += F.binary_cross_entropy_with_logits(output, target.float(), size_average=False).item() # sum up batch loss
            pred = torch.sigmoid(output) > 0.5
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(name, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
# preprocessing
data_flag = 'pneumoniamnist'

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
val_dataset = DataClass(split='train', transform=data_transform, download=download)
prng = RandomState(seed)
random_permute = prng.permutation(np.arange(0, 1000))
train_top = 10//n_classes
val_top = 1000//n_classes
indx_train = np.concatenate([np.where(train_dataset.labels == label)[0][random_permute[0:train_top]] for label in range(0, n_classes)])
indx_val = np.concatenate([np.where(train_dataset.labels == label)[0][random_permute[train_top:train_top + val_top]] for label in range(0, n_classes)])

train_data = Subset(train_dataset, indx_train)
val_data = Subset(val_dataset, indx_val)



print('Num Samples For Training %d Num Samples For Val %d'%(train_data.indices.shape[0],val_data.indices.shape[0]))

train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=32, 
                                                shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

model = FullyConnected(input_size, output_size, hidden_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(50):
    l = train(model, device, train_loader, optimizer, epoch, display=epoch%5==0)
    loss_val.append(l)

    accs_val.append(test(model, device, val_loader))