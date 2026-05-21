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


class PneumoniaMNISTCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # (1) Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # (2) MaxPooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 -> 7x7
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # (10) Output layer
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # Conv + Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        return self.out(x)

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