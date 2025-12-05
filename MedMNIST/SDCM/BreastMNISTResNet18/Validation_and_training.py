import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import ResNet18
from medmnist import BreastMNIST
from medmnist import INFO
from model import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 100
lr = 0.00001
early_stopper = EarlyStopping(patience=10, mode="min")


# Download PneumoniaMNIST
data_flag = 'breastmnist'
info = INFO[data_flag]
DataClass = info['python_class']

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load training dataset
train_dataset = BreastMNIST(split='train', transform=transform, download=True)

test_dataset = BreastMNIST(split='test', transform=transform, download=True)

val_dataset = BreastMNIST(split='val', transform=transform, download=True)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = ResNet18(kernel_size=1)

# Class weight for BCEWithLogitsLoss: use positive class weight (for label 1)
pos_weight = torch.tensor([40.0 / 6.0])  # pos_weight > 1 => penalize positive class more

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)
correct = 0
total = 0
correct_v = 0
total_v = 0
# Training parameters
num_epochs = epochs 

train_losses = []
train_accuracy = []
check_val_every_n_epoch = 1
train_vlosses = []
val_accuracy = []

# Example training loop structure
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:  # define train_loader appropriately
        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()  # threshold at 0.5
        correct += (preds == labels).sum().item()
        total += labels.numel()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}")
    
    avg_loss = running_loss/len(train_loader)
    train_losses.append((epoch, avg_loss))

    acc = correct/total
    train_accuracy.append((epoch, acc))

    if epoch % check_val_every_n_epoch == 0:
        with torch.no_grad():
            model.eval()
            running_vloss = 0.0    

            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.float()
                loss = criterion(outputs, labels)

                running_vloss += loss.item()
                preds_v = (torch.sigmoid(outputs) > 0.5).float()
                correct_v += (preds_v == labels).sum().item()
                total_v += labels.numel()

        avg_vloss = running_vloss/len(val_loader)
        train_vlosses.append((epoch, avg_vloss))

        acc_val = correct_v/total_v
        val_accuracy.append((epoch, acc_val)) 

    early_stopper(avg_vloss, model)
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

t_epochs, t_loss = list(zip(*train_losses))
t_epochs, t_acc = list(zip(*train_accuracy))
t_epochs, v_loss = list(zip(*train_vlosses))
t_epochs, v_acc = list(zip(*val_accuracy))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(t_epochs, t_loss) 
ax[0].plot(t_epochs, v_loss)
ax[0].set_title(f"Loss Curve (batch_size={batch_size}, lr={lr})")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend(["Trainamento", "Validação"])

ax[1].plot(t_epochs, t_acc) 
ax[1].plot(t_epochs, v_acc)
ax[1].set_title(f"Accuracy Curve (batch_size={batch_size}, lr={lr})")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend(["Trainamento", "Validação"])
fig.savefig("MedMNIST/BreastMNISTResNet18/grafico.png")


torch.save(model.state_dict(), "./BreastMNISTResNet.pth")  # Save the model state
print(correct/total)