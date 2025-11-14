from medmnist import PneumoniaMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Transformação básica (converte para tensor)
transform = transforms.Compose([transforms.ToTensor()])

# Carregar os splits
train_dataset = PneumoniaMNIST(split='train', download=True, transform=transform)
val_dataset = PneumoniaMNIST(split='val', download=True, transform=transform)
test_dataset = PneumoniaMNIST(split='test', download=True, transform=transform)

# Função auxiliar para contar as classes
def class_distribution(dataset):
    labels = np.array([label for _, label in dataset])
    # Flatten caso seja (N, 1)
    labels = labels.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

# Obter as distribuições
train_dist = class_distribution(train_dataset)
val_dist = class_distribution(val_dataset)
test_dist = class_distribution(test_dataset)

# Mostrar as contagens
print("Distribuição das classes:")
print(f"Treino: {train_dist}")
print(f"Validação: {val_dist}")
print(f"Teste: {test_dist}")

# Plotar em gráfico de barras
splits = ['Treino', 'Validação', 'Teste']
class0 = [train_dist.get(0, 0), val_dist.get(0, 0), test_dist.get(0, 0)]
class1 = [train_dist.get(1, 0), val_dist.get(1, 0), test_dist.get(1, 0)]

x = np.arange(len(splits))
width = 0.35

plt.figure(figsize=(7,5))
plt.bar(x - width/2, class0, width, label='Classe 0 (Normal)')
plt.bar(x + width/2, class1, width, label='Classe 1 (Pneumonia)')

plt.xlabel('Divisão do conjunto')
plt.ylabel('Número de amostras')
plt.title('Distribuição de classes no PneumoniaMNIST')
plt.xticks(x, splits)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("filename.png")
