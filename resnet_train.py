import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader.gtsrb_loader import GTSRB_load
from models.resnet import ResNetWithSTN 
from src.cfg import NUM_CLASSES, BATCH, EPOCHS, LR, WEIGHT_DECAY, IMAGE_SIZE, DEVICE, GTSRB_TRAINING_PATH, RESNET_CHECKPOINT_PATH, RESNET_FIGURE_PATH
from src.earlystopping import EarlyStopping
from src.cfg import EARLY_STOPPING_PARAMS

def train_resnet_with_stn():
    if not os.path.exists(GTSRB_TRAINING_PATH):
        raise FileNotFoundError(f"Training directory not found: {GTSRB_TRAINING_PATH}")
    
    train_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='train')
    val_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=3)

    # Use ResNetWithSTN model
    model = ResNetWithSTN(num_classes=NUM_CLASSES, stn_filters=(16, 32), stn_fc_units=128, input_size=IMAGE_SIZE)
    model = model.to(DEVICE)

    # stage 2
    if os.path.exists(RESNET_CHECKPOINT_PATH):
        checkpoint = torch.load(RESNET_CHECKPOINT_PATH, map_location=DEVICE)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")  # Debug: Show checkpoint structure
        
        # Load model state
        try:
            model.load_state_dict(checkpoint, strict=False)  # Use strict=False to handle fc mismatch
            print(f"Loaded model weights from {RESNET_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise RuntimeError("Failed to load checkpoint weights")
    else:
        print(f"No checkpoint found at {RESNET_CHECKPOINT_PATH}. Starting with pretrained ImageNet weights.")
    
    model.unfreeze_layers(layer_names=['layer4', 'fc'])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    
    optimizer = optim.Adam([
        {'params': model.stn.parameters(), 'lr': LR},
        {'params': model.resnet.fc.parameters(), 'lr': LR * 0.1},
        {'params': model.resnet.layer4.parameters(), 'lr': LR * 0.1}
    ], lr=LR, weight_decay=WEIGHT_DECAY)
    
    if os.path.exists(RESNET_CHECKPOINT_PATH):
        checkpoint = torch.load(RESNET_CHECKPOINT_PATH, map_location=DEVICE)
        try:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
            else:
                print("Optimizer state not found in checkpoint. Initializing new optimizer.")
        except Exception as e:
            print(f"Error loading optimizer state: {e}")
            print("Initializing new optimizer.")
    
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PARAMS['patience'],
        delta=EARLY_STOPPING_PARAMS['delta'],
        verbose=EARLY_STOPPING_PARAMS['verbose']
    )

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            progress_bar.set_postfix(loss=running_loss/len(train_loader), accuracy=f"{accuracy:.2f}%")

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        early_stopping(val_loss, {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, RESNET_CHECKPOINT_PATH)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

        scheduler.step()

    print(f"Training complete. Model saved to: {RESNET_CHECKPOINT_PATH}")


    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', color='blue')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='x', color='green')
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='x', color='red')
    ax2.set_ylabel('Accuracy (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    loss_png_path = os.path.join(RESNET_FIGURE_PATH, "resnet_loss_accuracy_stage2.png")
    plt.title('Training and Validation Loss and Accuracy')
    plt.savefig(loss_png_path)
    plt.close()
    
if __name__ == "__main__":
    train_resnet_with_stn()