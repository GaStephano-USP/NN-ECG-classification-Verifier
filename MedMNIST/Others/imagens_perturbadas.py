import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from medmnist import PneumoniaMNIST, OCTMNIST, BreastMNIST
import numpy as np
import cv2
import os

transform = transforms.Compose([transforms.ToTensor()])

pneumonia = PneumoniaMNIST(split='train', download=True, transform=transform)
breast    = BreastMNIST(split='train', download=True, transform=transform)
octmnist  = OCTMNIST(split='train', download=True, transform=transform)

# =========================
# Função SnP
# =========================
def snp(img_tensor, k, p, seed):
    rng = np.random.default_rng(seed)
    pixel = rng.permutation(784)[:k].tolist()
    x = int(784 * p / 100 + 0.5)
    values = [1.0] * x + [0.0] * (784 - x)
    values = rng.permutation(values)[:k].tolist()
    flat = img_tensor.view(-1).clone().numpy()
    for idx, pix in enumerate(pixel):
        flat[pix] = values[idx]
    return torch.tensor(flat).view(1, 28, 28)

# =========================
# Função Rotação
# =========================
def rotation(img_tensor, angle):
    img = img_tensor.squeeze(0).numpy()
    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (28, 28),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return torch.tensor(rotated).unsqueeze(0)

# =========================
# Função Recorte
# =========================
def crop(img_tensor, altura, largura, P0):
    flat = img_tensor.view(-1).clone().numpy()
    region_rows = range(P0[1], P0[1] + altura)
    for i in region_rows:
        for j in range(P0[0], P0[0] + largura):
            pixel_idx = i * 28 + j
            if pixel_idx < 784:
                flat[pixel_idx] = 0.0
    return torch.tensor(flat).view(1, 28, 28)

# =========================
# Função para pegar uma imagem de cada classe
# =========================
def get_one_example_per_class(dataset, class_labels, transform_fn=None):
    images = []
    labels = []
    found  = set()

    for img, label in dataset:
        label = int(label.item())
        if label not in found:
            if transform_fn is not None:
                img = transform_fn(img)
            images.append(img)
            labels.append(class_labels[label])
            found.add(label)
        if len(found) == len(class_labels):
            break

    return images, labels

# =========================
# Labels
# =========================
pneumonia_labels = {0: "Normal", 1: "Pneumonia"}
breast_labels    = {0: "Maligno", 1: "Benigno"}
oct_labels       = {0: "Neovascularização\nde Coroide",
                    1: "Edema Macular\nDiabético",
                    2: "Drusas",
                    3: "Normal"}

# =========================
# Coleta das imagens com artefatos
# =========================
p_imgs, p_lbls = get_one_example_per_class(
    pneumonia, pneumonia_labels,
    transform_fn=lambda img: crop(img, altura=3, largura=3, P0=(12, 12))
)

b_imgs, b_lbls = get_one_example_per_class(
    breast, breast_labels,
    transform_fn=lambda img: rotation(img, angle=45)
)

o_imgs, o_lbls = get_one_example_per_class(
    octmnist, oct_labels,
    transform_fn=lambda img: snp(img, k=200, p=50, seed=1950)
)
#o_lbls = ["Choroidal Neo.", "Normal", "Macular Ed.", "Drusen"]
o_lbls = ["Neo. Coroidal", "Normal", "Ed. Macular", "Drusas"]
#b_lbls = ["Benign", "Malignant"]

all_images = p_imgs + b_imgs + o_imgs
all_titles = (
    [f"PneumoniaMNIST {lbl}\n(Recorte 3×3)" for lbl in p_lbls] +
    [f"BreastMNIST {lbl}\n(Rotação 45°)"    for lbl in b_lbls] +
    [f"OCTMNIST {lbl}\n(Salt and Pepper)"    for lbl in o_lbls]
)

# =========================
# Plotagem
# =========================
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = all_images[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(all_titles[i], fontsize=18, fontweight='bold')
    ax.axis('off')

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig("./MedMNIST/Others/imagens_perturbadas.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Imagem salva")