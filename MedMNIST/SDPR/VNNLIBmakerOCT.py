import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import OCTMNIST
from medmnist import INFO
import numpy as np
import cv2
import os
import glob
import argparse

default_epsilon = 0.00
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

class OCTMNISTCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # (1) Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # (2) MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # (10) Output layer
        self.out = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        # Conv + Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        return self.out(x)

# hyperparameters
input_size = 784
output_size = 4
hidden_size = 50
def process_network(epsilon, mode, k, p, altura, largura, P0, seed, pixels, angle, model, model_path):
    model_path = model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (model == "FC"):
        model = OCTMNISTFC(input_size, output_size, hidden_size).to(device)
    elif (model == "CNN"):
        model = OCTMNISTCNN().to(device)
    else:
        print(f"Modelo {model} inválido")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    info = INFO['octmnist']
    DataClass = OCTMNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = DataClass(split='test', transform=transform, download=True)
    iterator = 0
    folder_path_delete = "./safety_benchmarks/benchmarks/OCTMNIST/vnnlib/"
    compiled_files = glob.glob(os.path.join(folder_path_delete, "*.vnnlib.compiled"))
    #print('tamanho', len(dataset))
    for file_path in compiled_files:
        try:
            os.remove(file_path)
            #print(f"Deletado: {file_path}")
        except Exception as e:
            print(f"Erro ao deletar {file_path}: {e}")
    
    if (altura != None and largura != None and P0 != None):
        region = [P0[1]+1, P0[1]+altura, P0[0]+1, P0[0]+largura]
        #print (region)
    else: region = None

    rng = np.random.default_rng(seed)
    if (pixels == None):
        delimit = []
        if (region != None):
            for i in range(region[0], region[1]+1):
                for j in range(28*(i-1)+region[2], 28*(i-1)+region[3]+1):
                    delimit.append(j)
            #print (delimit)
            pixel = rng.permutation(delimit)
            pixel = pixel [:k]

        else:
            pixel = rng.permutation(784)
            pixel = pixel [:k]
        pixel = pixel.tolist()

    else:
        pixel = pixels

    if (k == None and pixel != None): 
        k = len(pixel)
    x = int(784*p/100+0.5)
    values = [1.0]*x + [0.0]*(784 - x)
    values = rng.permutation(values)
    values = values [:k]
    values = values.tolist()
 
    print(f"pixels = {pixel} e valores = {values}")
    a = 0    #pra iterar o values
    
    for i in range(len(dataset)):
        temp = len(dataset)
        image_tensor, label_tensor = dataset[i]
        #print(temp)
        image_tensor = image_tensor.unsqueeze(0).to(device)  # shape [1,1,28,28]
        label = int(label_tensor.item())
        image_tensor, label_tensor = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)  # shape [1,1,28,28]
        label = int(label_tensor.item())
        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()
        if predicted == label:
            #print (label)
            if epsilon == None:
                epsilon = default_epsilon

            if mode == "Rot":
                img = image_tensor.squeeze(0).cpu().numpy()
                img = img.squeeze(0)

                if (i==2):
                    print(f"testes {img[0]}")
                    print(img.shape)

                #print(f"array: {image_tensor}")
                #print(img.dtype)
                #print(angle)
                M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                image_tensor = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue = 0)
                #print(f"{i} rot: {image_tensor.shape}")
                if (i==2):
                    print(f"testes {image_tensor[0]}")
                    #print(image_tensor.shape)

                image_tensor = torch.from_numpy(image_tensor)
               
                #print (f"tensor: {image_tensor}")

            #print(image_tensor.shape)
            flattened_input = image_tensor.view(-1).cpu().numpy()
            #print (len(flattened_input))
            output_path_string = f"safety_benchmarks/benchmarks/OCTMNIST/vnnlib/Property_" + str(iterator) + ".vnnlib"
            output_path = os.path.abspath(output_path_string)
            a = 0
            print (iterator, label)
            iterator = iterator + 1
            try:
                with open(output_path, "w") as f:
                    n = 0
                    for j in range(784):
                        f.write(f"(declare-const X_{j} Real)\n")
                    for j in range(4):
                        f.write(f"(declare-const Y_{j} Real)\n")
                    for val in flattened_input:
                        if mode == 'SnP':
                            if n in pixel and a < len(values):
                                val = values[a]
                                #print(f"pixel = {n} e valor ficou {val}") 
                                a += 1      
                            f.write(f"(assert (<= X_{n} {val}))\n")
                            f.write(f"(assert (>= X_{n} {val}))\n")
                    
                        elif mode == 'rel':
                            f.write(f"(assert (<= X_{n} {val+(epsilon*val)}))\n")
                            f.write(f"(assert (>= X_{n} {val-(epsilon*val)}))\n")
                        elif mode == 'abs':
                            f.write(f"(assert (<= X_{n} {val+epsilon}))\n")
                            f.write(f"(assert (>= X_{n} {val-epsilon}))\n")

                        elif mode == 'Crop':
                            if n in delimit:    
                                val = 0.0
                                #print(f"pixel = {n} e valor ficou {val}")
                            f.write(f"(assert (<= X_{n} {val+epsilon}))\n")
                            f.write(f"(assert (>= X_{n} {val-epsilon}))\n")

                        elif mode == 'Rot':
                            f.write(f"(assert (<= X_{n} {val}))\n")
                            f.write(f"(assert (>= X_{n} {val}))\n")

                        n = n + 1
                    f.write("(assert (or\n")
                    for m in range(4):
                        if m != label:
                            f.write(f"(and (<= Y_{label} Y_{m}))\n")
                    f.write("))")

                    #print(iterator, label)
                # print(f"Serialized input saved to: {output_path}")
            except Exception as e:
                print(f"Error writing file: {e}")
    for g in range(iterator):
        output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/OCTMNIST/instances_{g}.csv")
        try:
            with open(output_path_instances, "w") as f:
                f.write(f"vnnlib/Property_{g}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")
    output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/OCTMNIST/all_instances.csv")
    with open(output_path_instances, "w") as f:
        try:
            for j in range(iterator):
                f.write(f"vnnlib/Property_{j}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")
    for file_path in compiled_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Erro ao deletar {file_path}: {e}")
        

def prop_0_100(proporcao):
    v = int(proporcao)
    if v < 0:
        return 0
    if v > 100:
        return 100
    return v  

def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="FC",
                        help='Modelo da rede')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Caminho do .pth da rede')

    parser.add_argument('--epsilon', type=float, default=None,
                        help='Dimensao da perturbacao a ser adicionada')
    parser.add_argument('--mode', type=str, default='rel',
                        help='Modo de operação')
    parser.add_argument('--k', type=int, default=0,
                        help='Quatidade de pixels perturbados')
    parser.add_argument('--p', type=prop_0_100, default=100,
                        help='Proporção de pixels com valor 1')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed para escolher os pixels perturbados')
    parser.add_argument('--pixels', nargs='+', type=int, default=None) 

    parser.add_argument('--angle', type=float, default=None,
                        help='angulo da rotação em graus')
    
    parser.add_argument('--altura', type=int, default=None,
                        help='Altura da delimitação ou Crop')
    parser.add_argument('--largura', type=int, default=None,
                        help='Largura da delimitação ou Crop') 
    parser.add_argument('--P0', nargs=2, type=int, default=None,
                        help='Ponto inicial (x0, y0) da delimitação ou Crop - ponto (0,0) é o pixel 1')  
    
    args = parser.parse_args()

    process_network(args.epsilon, args.mode, args.k, args.p, args.altura, args.largura, args.P0, args.seed, args.pixels, args.angle, args.model, args.model_path)
 
if __name__ == "__main__":
    main()