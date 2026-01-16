import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import BreastMNIST
from medmnist import INFO
import numpy as np
import os
import glob
from PIL import Image
from BreastMNISTmodel import FullyConnected

default_epsilon = 0.10

def process_network(epsilon, mode):
    model_path = "./BreastMNISTResNet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparameters
    model = FullyConnected(784, 1, 100).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    DataClass = BreastMNIST

    transform = transforms.Compose([ 
        transforms.ToTensor(),
    ])

    dataset = DataClass(split='test', transform=transform, download=True)
    iterator = 0
    folder_path_delete = "./safety_benchmarks/benchmarks/BreastMNIST/vnnlib"
    compiled_files = glob.glob(os.path.join(folder_path_delete, "*.vnnlib.compiled"))
    #print('tamanho', len(dataset))
    for file_path in compiled_files:
        try:
            os.remove(file_path)
            #print(f"Deletado: {file_path}")
        except Exception as e:
            print(f"Erro ao deletar {file_path}: {e}")
    for i in range(len(dataset)):
        image_tensor, label_tensor = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)  # shape [1,1,28,28]
        label = int(label_tensor.item())
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output)
            predicted = int(prob > 0.5)
            #print('predicted:', predicted, 'prob:', prob.item(), 'label:', label, 'sample:', str(i))
        if predicted == label:
            if epsilon == None:
                epsilon = default_epsilon
            flattened_input = image_tensor.view(-1).cpu().numpy()
            output_path_string = f"safety_benchmarks/benchmarks/BreastMNIST/vnnlib/Property_" + str(iterator) + ".vnnlib"
            output_path = os.path.abspath(output_path_string)
            iterator = iterator + 1
            try:
                with open(output_path, "w") as f:
                    n = 0
                    for j in range(784):
                        f.write(f"(declare-const X_{j} Real)\n")
                    f.write(f"(declare-const Y_0 Real)\n")
                    for val in flattened_input:
                        if mode == 'rel':
                            f.write(f"(assert (<= X_{n} {val+(epsilon*val)}))\n")
                            f.write(f"(assert (>= X_{n} {val-(epsilon*val)}))\n")
                        elif mode == 'abs':
                            f.write(f"(assert (<= X_{n} {val+epsilon}))\n")
                            f.write(f"(assert (>= X_{n} {val-epsilon}))\n")
                        n = n + 1
                    if  label == 0:
                        f.write(f"(assert (>= Y_0 0))\n")
                    else:
                        f.write(f"(assert (<= Y_0 0))\n")
                # print(f"Serialized input saved to: {output_path}")
            except Exception as e:
                print(f"Error writing file: {e}")
    for g in range(iterator):
        output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/BreastMNIST/instances_{g}.csv")
        try:
            with open(output_path_instances, "w") as f:
                f.write(f"vnnlib/Property_{g}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")
    output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/BreastMNIST/all_instances.csv")
    with open(output_path_instances, "w") as f:
        try:
            for j in range(iterator):
                f.write(f"vnnlib/Property_{j}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")

def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Dimensao da perturbacao a ser adicionada')
    parser.add_argument('--mode', type=str, default='abs',
                        help='The epsilon for L_infinity perturbation')
    args = parser.parse_args()

    process_network(args.epsilon, args.mode)
 
if __name__ == "__main__":
    main()