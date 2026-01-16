import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import random
from numpy.random import RandomState
from torch.utils.data import Subset

class PneumoniaMNISTCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # (1) Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=0   # 28 -> 26
        )

        # (2) MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 26 -> 13

        # (3–5) Convolutional layers
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, padding=0)   # 13 -> 11
        self.conv3 = nn.Conv2d(40, 80, kernel_size=3, padding=0)   # 11 -> 9
        self.conv4 = nn.Conv2d(80, 160, kernel_size=3, padding=0)  # 9 -> 7

        # (6–9) Fully connected layers
        self.fc1 = nn.Linear(160 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Softmax output (for training / inference)
        x = self.fc4(x)
        return x
    
def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target.float())
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), './trained_models/PneumoniaMNIST/PnuemoniaMNISTConvNet.pth')
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
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).long()
            correct += pred.eq(target).sum().item()

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
      ])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

loss_val = []
accs_val = []
seed = random.randint(0,50)

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

model = PneumoniaMNISTCNN()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(50):
    l = train(model, device, train_loader, optimizer, epoch, display=epoch%5==0)
    loss_val.append(l)

    accs_val.append(test(model, device, val_loader))
    print(accs_val)