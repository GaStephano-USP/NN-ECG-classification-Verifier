import torch
from torchvision import transforms
from medmnist import PneumoniaMNIST
from medmnist import INFO
import numpy as np
import json
import os

# Load info
info = INFO['pneumoniamnist']
DataClass = PneumoniaMNIST

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to [0,1] range and shape [1,28,28]
])

# Load dataset (download=True the first time)
dataset = DataClass(split='test', transform=transform, download=True)

# Get one sample (image, label)
image_tensor, label = dataset[0]  # Change index to get other images

print(label)
# Flatten the image input to a 1D vector
flattened_input = image_tensor.view(-1).numpy()  # shape (784,)

# Convert to list (for serialization)
input_list = flattened_input.tolist()
output_path = "C:\\Users\\gaste\\Downloads\\input_vector.txt"
# Save to a .txt file with one value per line
try:
    with open(output_path, "w") as f:
        i = 0
        for val in flattened_input:
            f.write(f"(assert (<= X_{i} {val}))\n")
            f.write(f"(assert (>= X_{i} {val}))\n")
            i = i + 1
except Exception as e:
    print(f"❌ Error writing file: {e}")
print("Serialized input saved to input_vector.txt")
# Display some info
print(f"Label: {label}")
print(f"Serialized input vector (first 10 values): {input_list[:10]}")