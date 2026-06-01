import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from medmnist import PneumoniaMNIST, OCTMNIST, BreastMNIST

# =========================
# Transformação
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

# =========================
# Carregamento dos datasets
# =========================
pneumonia = PneumoniaMNIST(split='train', download=True, transform=transform)
breast = BreastMNIST(split='train', download=True, transform=transform)
octmnist = OCTMNIST(split='train', download=True, transform=transform)

# =========================
# Função para pegar uma imagem de cada classe
# =========================
def get_one_example_per_class(dataset, class_labels):
    images = []
    labels = []

    found = set()

    for img, label in dataset:
        label = int(label)

        if label not in found:
            images.append(img)
            labels.append(class_labels[label])
            found.add(label)

        if len(found) == len(class_labels):
            break

    return images, labels

# =========================
# Labels dos datasets
# =========================

# PneumoniaMNIST
pneumonia_labels = {
    0: "Normal",
    1: "Pneumonia"
}

# BreastMNIST
breast_labels = {
    0: "Maligno",
    1: "Benigno"
}

# OCTMNIST
oct_labels = {
    0: "Neovascularização de Coroide",
    1: "Edema Macular Diabético",
    2: "Drusas",
    3: "Normal"
}

# =========================
# Coleta das imagens
# =========================
p_imgs, p_lbls = get_one_example_per_class(pneumonia, pneumonia_labels)
b_imgs, b_lbls = get_one_example_per_class(breast, breast_labels)
o_imgs, o_lbls = get_one_example_per_class(octmnist, oct_labels)

# Junta tudo
all_images = p_imgs + b_imgs + o_imgs
all_titles = (
    [f"PneumoniaMNIST\n{lbl}" for lbl in p_lbls] +
    [f"BreastMNIST\n{lbl}" for lbl in b_lbls] +
    [f"OCTMNIST\n{lbl}" for lbl in o_lbls]
)

# =========================
# Plotagem
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
output_file = "medical_mnist_examples.png"

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Imagem salva em: {output_file}")