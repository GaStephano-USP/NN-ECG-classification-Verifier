import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader

class FullyConnected(nn.Module):  # herança corrigida para o seu modelo do PneumoMNIST
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta função imprime e plota a matriz de confusão.
    Normalização pode ser aplicada definindo `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    parser = argparse.ArgumentParser(description='Avaliação do modelo PneumoMNIST com matriz de confusão')
    parser.add_argument('--model_path', type=str, required=True, help='Caminho do arquivo .pth (Ex: ./trained_models/PneumoniaMNIST/PnuemoniaMNISTFCNet.pth)')
    parser.add_argument('--output_path', type=str, default='matriz_confusao_pneumo.png', help='Caminho para salvar a imagem da matriz')
    args = parser.parse_args()    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instancia o modelo conforme as configurações do PneumoMNISTFCnet (Input: 784, Output: 1, Hidden: 50)
    model = FullyConnected(input_size=784, num_classes=1, hidden_size=50).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Carrega as informações e dataset corretos do PneumoniaMNIST
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = DataClass(split='test', transform=transform, download=True)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Classificação binária (2 classes: normal e pneumonia)
    num_classes  = 2
    cm           = np.zeros((num_classes, num_classes), dtype=int)
    correct_test = 0
    total_test   = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            # Mantém o target float e no formato correto para comparação posterior
            labels  = labels.to(device).float()
            
            outputs = model(inputs)
            
            # Logica de classificação binária: Sigmoide + Limiar 0.5
            preds = (torch.sigmoid(outputs) > 0.5).float()

            correct_test += preds.eq(labels.view_as(preds)).sum().item()
            total_test   += labels.numel()

            # Converte para inteiro plano para indexar a matriz de confusão de forma segura
            labels_np = labels.squeeze().long().cpu().numpy()
            preds_np  = preds.squeeze().long().cpu().numpy()

            # Caso o batch tenha tamanho 1, força a ser array para o loop funcionar
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np])
                preds_np = np.array([preds_np])

            for i in range(len(labels_np)):
                real = labels_np[i]
                pred = preds_np[i]
                cm[real][pred] += 1

    acc_test = correct_test / total_test
    print(f"ACC Test: {acc_test:.4f}")

    # Mapeamento das classes do PneumoniaMNIST (0: normal, 1: pneumonia)
    classes = ['Normal', 'Pneumonia']
    
    plt.figure(figsize=(6, 5))
    plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão — PneumoniaMNIST')
    
    if args.output_path:
        plt.savefig(args.output_path)
        print(f"Matriz de confusão salva em: {args.output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()