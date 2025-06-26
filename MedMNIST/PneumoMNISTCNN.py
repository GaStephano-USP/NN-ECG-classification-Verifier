# Code with little modifications from https://www.kaggle.com/code/prathambarua/pneumonia-cnn
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn 
import torch.nn.functional as F
import medmnist
from medmnist import INFO, Evaluator
from numpy.random import RandomState
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset
import re
from torchvision import datasets, transforms
def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target.float())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'trained_models/CNN_Net/CNN_ResNet_{}.pth'.format(epoch))
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
            pred = output >= 0.5 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(name, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers+=[nn.Conv2d(1, 16,  kernel_size=3) , 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 16,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 32,  kernel_size=3), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(32, 32,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.fc = nn.Linear(32*4*4, 1)
    def forward(self, x):
        for i in range(len(self.layers)):
          x = self.layers[i](x)
        x = x.view(-1, 32*4*4)
        x = self.fc(x)
        return x
def resNet18():
    resNet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    resNet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet18.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return resNet18

def resNet34():
    resNet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    resNet34.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet34.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return resNet34

def resNet50():
    resNet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    resNet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet50.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    return resNet50

def resNet101():
    resNet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)
    resNet101.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet101.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    return resNet101

def resNet152():
    resNet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False)
    resNet152.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resNet152.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    return resNet152

resNets = [resNet18]

from random import randint

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

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

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='train', transform=data_transform, download=download)

loss18_val = []
loss34_val = []
loss50_val = []
loss101_val = []
loss152_val = []
accs_val = []
seed = randint(0,50)

for resNet in resNets:
    if resNet is resNet18:
        print("ResNet18: ")
    elif resNet is resNet34:
        print("ResNet34: ")
    elif resNet is resNet50:
        print("ResNet50: ")
    elif resNet is resNet101:
        print("ResNet101: ")
    elif resNet is resNet152:
        print("ResNet152: ")
    # for seed in  range(0, 50):
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

    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=128, 
                                                shuffle=False)

    model = resNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(50):
        l = train(model, device, train_loader, optimizer, epoch, display=epoch%5==0)
        if resNet is resNet18:
            loss18_val.append(l)
        elif resNet is resNet34:
            loss34_val.append(l)
        elif resNet is resNet50:
            loss50_val.append(l)
        elif resNet is resNet101:
            loss101_val.append(l)
        elif resNet is resNet152:
            loss152_val.append(l)

    accs_val.append(test(model, device, val_loader))