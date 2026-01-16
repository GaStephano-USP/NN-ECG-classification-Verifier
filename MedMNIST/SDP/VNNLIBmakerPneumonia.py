import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import PneumoniaMNIST
from medmnist import INFO
import numpy as np
import os
import glob
from PIL import Image

default_epsilon = 0.05
class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes, hidden_size):
        super(FullyConnected, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def process_network(epsilon, mode, k, p, altura, largura, P0, seed, pixels):
    model_path = "./trained_models/PneumoniaMNIST/PnuemoniaMNISTFCNet100.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparameters
    input_size = 784
    output_size = 1
    hidden_size = 150
    model = FullyConnected(input_size, output_size, hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    info = INFO['pneumoniamnist']
    DataClass = PneumoniaMNIST

    transform = transforms.Compose([ 
        transforms.ToTensor(),
    ])

    dataset = DataClass(split='test', transform=transform, download=True)
    iterator = 0
    folder_path_delete = "./safety_benchmarks/benchmarks/PneumoniaMNIST/vnnlib"
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
        print (region)
    else: region = None

    rng = np.random.default_rng(seed)
    if (pixels == None):
        delimit = []
        if (region != None):
            for i in range(region[0], region[1]+1):
                for j in range(28*(i-1)+region[2], 28*(i-1)+region[3]+1):
                    delimit.append(j)
            print (delimit)
            pixel = rng.choice(delimit, size = k, replace=False)


        else:
            pixel = rng.integers(0, 785, size = k)
        pixel = pixel.tolist()

    else:
        pixel = pixels

    if (k == None and pixels != None): 
        k = len(pixel)
        x = int(k*p/100+0.5)
        values = [1.0]*x + [0.0]*(k - x)
        rng.shuffle(values)
        #print(values)  
 
        print(f"pixels = {pixel} e valores = {values}")
    a = 0    #pra iterar o values
    
    for i in range(len(dataset)):
        temp = len(dataset)
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
            output_path_string = f"safety_benchmarks/benchmarks/PneumoniaMNIST/vnnlib/Property_" + str(iterator) + ".vnnlib"
            output_path = os.path.abspath(output_path_string)
            a = 0
            iterator = iterator + 1
            try:
                with open(output_path, "w") as f:
                    n = 0
                    for j in range(784):
                        f.write(f"(declare-const X_{j} Real)\n")
                    f.write(f"(declare-const Y_0 Real)\n")
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
                                val = 1.0
                                #print(f"pixel = {n} e valor ficou {val}")
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
    #print(iterator)
    for g in range(iterator):
        output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/PneumoniaMNIST/instances_{g}.csv")
        try:
            with open(output_path_instances, "w") as f:
                f.write(f"vnnlib/Property_{g}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")
    output_path_instances = os.path.abspath(f"safety_benchmarks/benchmarks/PneumoniaMNIST/all_instances.csv")
    with open(output_path_instances, "w") as f:
        try:
            for j in range(iterator):
                f.write(f"vnnlib/Property_{j}.vnnlib\n")         
        except Exception as e:
            print(f"Error writing file: {e}")

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
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Dimensao da perturbacao a ser adicionada')
    parser.add_argument('--mode', type=str, default='rel',
                        help='Modo de operação')
    parser.add_argument('--k', type=int, default=None,
                        help='Quatidade de pixels perturbados')
    parser.add_argument('--p', type=prop_0_100, default=50,
                        help='Proporção de pixels com valor 1')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed para escolher os pixels perturbados')
    parser.add_argument('--pixels', nargs='+', type=int, default=None) 
    
    parser.add_argument('--altura', type=int, default=None,
                        help='Altura da delimitação ou Crop')
    parser.add_argument('--largura', type=int, default=None,
                        help='Largura da delimitação ou Crop') 
    parser.add_argument('--P0', nargs=2, type=int, default=None,
                        help='Ponto inicial (x0, y0) da delimitação ou Crop - ponto (0,0) é o pixel 1')  
    
    args = parser.parse_args()

    process_network(args.epsilon, args.mode, args.k, args.p, args.altura, args.largura, args.P0, args.seed, args.pixels)
 
if __name__ == "__main__":
    main()

    