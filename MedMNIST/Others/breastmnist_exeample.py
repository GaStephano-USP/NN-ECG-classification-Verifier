import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from medmnist import BreastMNIST

# =========================
# Transformação
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =========================
# Carregamento do dataset
# =========================
breast = BreastMNIST(split='train', download=True, transform=transform)

# =========================
# Função para pegar N exemplos de cada classe
# =========================
def get_multiple_examples_per_class(dataset, class_labels, num_per_class=4):
    images = []
    labels = []
    
    # Dicionário para contar quantas imagens já pegamos de cada classe
    counts = {label_idx: 0 for label_idx in class_labels.keys()}
    total_needed = len(class_labels) * num_per_class

    for img, label in dataset:
        label = int(label.item() if hasattr(label, 'item') else label)

        if counts[label] < num_per_class:
            images.append(img)
            labels.append(class_labels[label])
            counts[label] += 1

        if len(images) == total_needed:
            break

    return images, labels

# =========================
# Labels do BreastMNIST
# =========================
breast_labels = {
    0: "Maligno",
    1: "Benigno"
}

# =========================
# Coleta das imagens (4 de cada classe = 8 no total)
# =========================
all_images, all_lbls = get_multiple_examples_per_class(breast, breast_labels, num_per_class=4)
all_titles = [f"BreastMNIST\n{lbl}" for lbl in all_lbls]

# =========================
# Plotagem (Grid 2x4 Completo)
# =========================
rows = 2
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = all_images[i].squeeze().numpy()

    ax.imshow(img, cmap='gray')
    ax.set_title(all_titles[i], fontsize=20, fontweight='bold')
    ax.axis('off')

plt.tight_layout()

# =========================
# Salvar figura
# =========================
output_file = "breast_mnist_8_examples.png"

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Imagem salva em: {output_file}")