import torch
import torch.nn as nn 
import torch.nn.functional as F
import copy
from typing import Type, Any, Callable, Union, List, Optional

def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()

        # Main Path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip Connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Residual Addition
        out += identity
        out = self.relu(out)

        return out


# hyperparameters
input_size = 784
output_size = 4
hidden_size = 50

epochs = 1000
batch_size = 50
learning_rate = 0.01
loss_func = nn.CrossEntropyLoss()


class BreastMNISTCNN(nn.Module):
    def __init__(self, num_classes=1):
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
        return self.out(x)


# --- Early Stopping Class ---

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