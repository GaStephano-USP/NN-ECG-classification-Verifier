import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from medmnist import INFO
# source: https://www.sciencedirect.com/science/article/pii/S1877050924015424
class PneumoniaMNIST_CNN(nn.Module):
    def __init__(self):
        super(PneumoniaMNIST_CNN, self).__init__()
        
        # Layers 
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)        # Layer 1 - Conv Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Layer 2 - Maxpool Layer
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)       # Layer 3 - Conv Layer
        self.conv3 = nn.Conv2d(40, 80, kernel_size=3)       # Layer 4 - Conv Layer
        self.conv4 = nn.Conv2d(80, 160, kernel_size=3)      # Layer 5 - Conv Layer
        self.flatten = nn.Flatten()                         # Layer F - Flatten *in the original work it is ommited
        self.fc1 = nn.Linear(7840, 1000)         # Layer 6 - FC Layer
        self.fc2 = nn.Linear(1000, 500)          # Layer 7 - FC Layer
        self.fc3 = nn.Linear(500, 50)            # Layer 8 - FC Layer
        self.fc4 = nn.Linear(50, 1)              # Layer 9 - FC Layer
       #self.softmax = nn.Softmax()   #*in the original work it is mencioned a Softmax Layer, but it does not work in this architecture

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Layer 1
        x = self.pool1(x)              # Layer 2
        x = F.relu(self.conv2(x))      # Layer 3
        x = F.relu(self.conv3(x))      # Layer 4
        x = F.relu(self.conv4(x))      # Layer 5
        x = self.flatten(x)            # Layer F
        x = F.relu(self.fc1(x))        # Layer 6
        x = F.relu(self.fc2(x))        # Layer 7
        x = F.relu(self.fc3(x))        # Layer 8
        x = self.fc4(x)                # Layer 9
        #x = self.softmax(x)
        return x  

# Download PneumoniaMNIST
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = info['python_class']

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load training dataset
train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Initialize model
model = PneumoniaMNIST_CNN()
# Class weight for BCEWithLogitsLoss: use positive class weight (for label 1)
pos_weight = torch.tensor([40.0 / 6.0])  # pos_weight > 1 => penalize positive class more

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 30

# Example training loop structure
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:  # define train_loader appropriately
        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.float()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "pytorch_model.pth")  # Save the model state