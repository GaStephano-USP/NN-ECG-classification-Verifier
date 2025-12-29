import torch
import torch.nn as nn 
import copy
from typing import Type, Any, Callable, Union, List, Optional

# --- Residual Block Implementation (Updated to enforce kernel % stride == 0) ---

class BasicBlock(nn.Module):
    """
    The standard Residual Block for ResNet-18/34, modified to ensure
    kernel_size is always divisible by the stride.
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        # Determine if we need to adjust the skip connection (Downsample)
        downsample_needed = stride != 1 or in_channels != out_channels

        # Configuration based on stride to satisfy kernel % stride == 0
        if stride == 1:
            # Stride 1: Use standard 3x3 kernel (3 % 1 == 0). Preserves size.
            kernel_main = 3
            padding_main = 1
            kernel_downsample = 1 # 1 % 1 == 0
            padding_downsample = 0
        elif stride == 2:
            # Stride 2: Use 4x4 kernel (4 % 2 == 0). Halves size.
            kernel_main = 4
            padding_main = 1
            # Skip connection: Use 2x2 kernel (2 % 2 == 0). Halves size.
            kernel_downsample = 2 
            padding_downsample = 0
        else:
            raise ValueError("Unsupported stride value in BasicBlock.")
        
        # 1. First Convolution: Uses the provided stride (1 or 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_main, stride=stride, padding=padding_main, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Second Convolution: Always uses stride 1 to preserve size. Uses 3x3 kernel (3 % 1 == 0).
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 3. Downsample (Skip Connection)
        self.downsample = None
        if downsample_needed:
            # Uses the configured downsample kernel and padding with the specified stride.
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_downsample, stride=stride, padding=padding_downsample, bias=False),
                nn.BatchNorm2d(out_channels),
            )

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


# --- ResNet-18 Model Definition (Structure unchanged, uses new BasicBlock) ---

class ResNet18(nn.Module):
    """The ResNet-18 Model adapted for 28x28 input (e.g., MedMNIST)"""

    def __init__(self, n_classes: int = 1) -> None:
        super().__init__()
        
        self.in_channels = 64 

        # 1. Initial Layers (3x3 conv, stride 1, 3 % 1 == 0)
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        # 2. Sequential Layers (Passing stride=2 only where spatial downsampling is required)
        
        # Layer 1: Stride 1 (64 -> 64 channels, no downsampling)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )
        
        # Layer 2: Stride 2 (64 -> 128 channels, spatial downsampling)
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2), 
            BasicBlock(128, 128, stride=1),
        )
        
        # Layer 3: Stride 2 (128 -> 256 channels, spatial downsampling)
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )
        
        # Layer 4: Stride 2 (256 -> 512 channels, spatial downsampling)
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )

        # 3. Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=256 * BasicBlock.expansion, out_features=n_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classifier
        x = self.avgpool(x)  # [bs, 512, 1, 1]
        x = torch.flatten(x, 1) # Flatten to [bs, 512]
        o = self.fc(x)

        return o

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