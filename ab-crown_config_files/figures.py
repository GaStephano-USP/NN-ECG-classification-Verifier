import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST, OCTMNIST
from collections import defaultdict

# Transformação
data_transform = transforms.Compose([
    transforms.ToTensor()
])

def show_and_save_grid(images, labels, title, filename, rows=2):
    n = len(images)
    cols = (n + rows - 1) // rows  # número de colunas necessário

    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()

    for i in range(n):
        img = images[i].squeeze().numpy()

        if img.ndim == 2:  # grayscale
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(img.transpose(1,2,0))

        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis("off")

    # Esconde quadros extras caso existam
    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✅ Imagem salva: {filename}")

# -------- PneumoniaMNIST (simples: pega primeiras 8 imagens) --------
p_data = PneumoniaMNIST(split="test", transform=data_transform, download=True)

p_loader = torch.utils.data.DataLoader(p_data, batch_size=8, shuffle=True)
p_images, p_labels = next(iter(p_loader))

show_and_save_grid(
    p_images,
    [int(l) for l in p_labels],
    "Amostras do conjunto de dados PneumoniaMNIST",
    "pneumonia_samples_2rows.png"
)

# -------- OCTMNIST (garante 1 exemplo por classe) --------
oct_data = OCTMNIST(split="train", transform=data_transform, download=True)

class_examples = defaultdict(list)
max_classes = 4  # 4 classes no OCTMNIST

for img, label in oct_data:
    label = int(label)
    if len(class_examples[label]) < 1:
        class_examples[label].append(img)
    if all(len(class_examples[c]) >= 1 for c in range(max_classes)):
        break

# Concatena imagens e labels
oct_images = torch.stack([class_examples[c][0] for c in range(max_classes)])
oct_labels = list(range(max_classes))

# Se quiser mais imagens, adiciona mais aleatórias
extra_needed = 8 - len(oct_images)
if extra_needed > 0:
    loader_extra = torch.utils.data.DataLoader(oct_data, batch_size=extra_needed, shuffle=True)
    extra_imgs, extra_lbls = next(iter(loader_extra))
    oct_images = torch.cat((oct_images, extra_imgs), dim=0)
    oct_labels = oct_labels + [int(l) for l in extra_lbls]

show_and_save_grid(
    oct_images,
    oct_labels,
    "Amostras do conjunto de dados OCTMNIST",
    "octmnist_samples_2rows.png"
)
