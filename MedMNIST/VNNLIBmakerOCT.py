import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import OCTMNIST
from medmnist import INFO
import numpy as np
import os
import glob

# Testar variações de epsilons até encontrar "pontos chave" 100% - 99.9% - 95% - 70% - 50% - 20% - 5% - 0% (mais generalizado) (opcional: utilizar o epsilon multiplicativo)
# Verificar se é possível salvar algum "contraexemplo"
# Pegar as propriedades mais robustas e aplicar artefatos de imagem específicos
# OPCIONAL: Utilizar uma amostragem maior do dataset

epsilon = 0.010
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

model_path = "./trained_models/OCT_FC_Net/OCT_FC_Net.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OCTMNISTFC(input_size, output_size, hidden_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

info = INFO['octmnist']
DataClass = OCTMNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[.5], std=[.5])
])

dataset = DataClass(split='test', transform=transform, download=True)
iterator = 0
folder_path_delete = "./safety_benchmarks/benchmarks/FC_Net/vnnlib/OCTMNIST"
compiled_files = glob.glob(os.path.join(folder_path_delete, "*.vnnlib.compiled"))
for file_path in compiled_files:
    try:
        os.remove(file_path)
        print(f"Deletado: {file_path}")
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
        output_path_string = f"safety_benchmarks/benchmarks/FC_Net/vnnlib/OCTMNIST/Property_" + str(iterator) + ".vnnlib"
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
    output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/FC_Net/instances_{g}.csv")
    try:
        with open(output_path_instances, "w") as f:
           f.write(f"vnnlib/OCTMNIST/Property_{g}.vnnlib\n")         
    except Exception as e:
       print(f"Error writing file: {e}")
output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/FC_Net/all_instances.csv")
with open(output_path_instances, "w") as f:
    try:
        for j in range(iterator):
            f.write(f"vnnlib/OCTMNIST/Property_{j}.vnnlib\n")         
    except Exception as e:
        print(f"Error writing file: {e}")