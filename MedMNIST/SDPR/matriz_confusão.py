import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from medmnist import OCTMNIST
from medmnist import INFO
import numpy as np
from torch.utils.data import DataLoader

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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

    parser = argparse.ArgumentParser(description='Avaliação do modelo com matriz de confusão')
    parser.add_argument('--model', type=str, default='CNN', help='Tipo do modelo')
    parser.add_argument('--model_path', type=str, help='Caminho do .pth')
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'CNN':
        model = OCTMNISTCNN().to(device)
    elif args.model == 'FC':
        model = OCTMNISTFC(784, 4, 50).to(device)
    else:
        raise ValueError(f"Modelo '{args.model}' inválido. Use 'CNN' ou 'FC'.")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = OCTMNIST(split='test', transform=transform, download=True)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    num_classes  = 4
    cm           = np.zeros((num_classes, num_classes), dtype=int)
    correct_test = 0
    total_test   = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            labels  = labels.squeeze().long().to(device)
            outputs = model(inputs)
            preds   = torch.argmax(outputs, dim=1)

            correct_test += (preds == labels).sum().item()
            total_test   += labels.numel()

            for i in range(len(labels)):
                real = labels[i]
                pred = preds[i]
                cm[real][pred] += 1

    acc_test = correct_test/total_test
    print(f"ACC Test {acc_test}")

    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusão — OCTMNIST')
    plt.savefig(args.output_path)


if __name__ == "__main__":
    main()    
