import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import OCTMNIST
from medmnist import INFO
import numpy as np
import os
import glob
import argparse

class OCTMNISTFC(nn.Module):  # inherits nn.Module

    def __init__(self, input_size, num_classes, hidden_size):  # input size = 28x28 = 784 for mnist
        super(OCTMNISTFC, self).__init__()
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
output_size = 4
hidden_size = 50
def process_network(epsilon, mode):
    model_path = "./trained_models/OCT_FC_Net/OCT_FC_Net.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OCTMNISTFC(input_size, output_size, hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    info = INFO['octmnist']
    DataClass = OCTMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = DataClass(split='test', transform=transform, download=True)
    iterator = 0
    folder_path_delete = "./safety_benchmarks/benchmarks/OCTMNIST/vnnlib/OCTMNIST"
    compiled_files = glob.glob(os.path.join(folder_path_delete, "*.vnnlib.compiled"))
    for file_path in compiled_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Erro ao deletar {file_path}: {e}")
    for i in range(len(dataset)):
        image_tensor, label_tensor = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)  # shape [1,1,28,28]
        label = int(label_tensor.item())
        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()
        if predicted == label:
            flattened_input = image_tensor.view(-1).cpu().numpy()
            output_path_string = f"safety_benchmarks/benchmarks/OCTMNIST/vnnlib/OCTMNIST/Property_" + str(iterator) + ".vnnlib"
            output_path = os.path.abspath(output_path_string)
            iterator = iterator + 1
            try:
                with open(output_path, "w") as f:
                    n = 0
                    for j in range(784):
                        f.write(f"(declare-const X_{j} Real)\n")
                    for k in range(4):
                        f.write(f"(declare-const Y_{k} Real)\n")
                    for val in flattened_input:
                        if mode == 'rel':
                            f.write(f"(assert (<= X_{n} {val+(epsilon*val)}))\n")
                            f.write(f"(assert (>= X_{n} {val-(epsilon*val)}))\n")
                        elif mode == 'abs':
                            f.write(f"(assert (<= X_{n} {val+epsilon}))\n")
                            f.write(f"(assert (>= X_{n} {val-epsilon}))\n")
                        n = n + 1
                    for m in range(4):
                        if m != label:
                            f.write(f"(assert (<= Y_{label} Y_{m}))\n")
                #print(f"Serialized input saved to: {output_path}")
            except Exception as e:
                print(f"Error writing file: {e}")
    for g in range(iterator):
        output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/OCTMNIST/instances_{g}.csv")
        try:
            with open(output_path_instances, "w") as f:
               f.write(f"vnnlib/OCTMNIST/Property_{g}.vnnlib\n")         
        except Exception as e:
           print(f"Error writing file: {e}")
    output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/OCTMNIST/all_instances.csv")
    with open(output_path_instances, "w") as f:
        try:
            for j in range(iterator):
                f.write(f"vnnlib/OCTMNIST/Property_{j}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")
def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Dimentional of epsilon used in the perturbation')
    parser.add_argument('--mode', type=str, default='abs',
                        help='Peturbation Mode')
    args = parser.parse_args()

    process_network(args.epsilon, args.mode)

if __name__ == "__main__":
    main()
