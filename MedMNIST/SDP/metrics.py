import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from PneumoMNISTFCnet import FullyConnected

def evaluate_saved_model(model_path="./trained_models/PneumoniaMNIST/PnuemoniaMNISTFCNet.pth", batch_size=128):
    # 1. Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Define the exact same transform used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 3. Load the test dataset and dataloader
    print("Loading PneumoniaMNIST test dataset...")
    test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 4. Initialize the model and load weights
    print(f"Loading model weights from {model_path}...")
    model = FullyConnected(784, 1, 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    # 5. Tracking variables
    correct_test = 0
    total_test = 0
    test_tp = 0
    test_actual_pos = 0

    # 6. Evaluation Loop
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            outputs = model(inputs)
            preds_test = (torch.sigmoid(outputs) > 0.5).float()
            
            # Global tracking
            correct_test += (preds_test == labels).sum().item()
            total_test += labels.numel()
            
            # Sensitivity pieces (True Positives and Actual Positives)
            test_tp += ((preds_test == 1) & (labels == 1)).sum().item()
            test_actual_pos += (labels == 1).sum().item()

    # 7. Calculate Metrics
    acc_test = correct_test / total_test if total_test > 0 else 0.0
    sens_test = test_tp / test_actual_pos if test_actual_pos > 0 else 0.0

    # 8. Print Summary
    print("\n" + "="*30)
    print("       EVALUATION RESULTS       ")
    print("="*30)
    print(f"Total Samples Tested: {total_test}")
    print(f"Accuracy (ACC):       {acc_test:.4f} ({acc_test * 100:.2f}%)")
    print(f"Sensitivity (Recall): {sens_test:.4f} ({sens_test * 100:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    evaluate_saved_model()