import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.gtsrb_loader import GTSRB_load
from models.resnet import ResNetWithSTN 
from src.cfg import NUM_CLASSES, BATCH, DEVICE, RESNET_CHECKPOINT_PATH, GTSRB_TRAINING_PATH, RESNET_FIGURE_PATH

def evaluate_resnet_with_stn():
    if not os.path.exists(GTSRB_TRAINING_PATH):
        raise FileNotFoundError(f"Training directory not found: {GTSRB_TRAINING_PATH}")
    
    val_dataset = GTSRB_load(training_dir=GTSRB_TRAINING_PATH, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=3)

    model = ResNetWithSTN(num_classes=NUM_CLASSES, stn_filters=(16, 32), stn_fc_units=128, input_size=(64, 64))
    model = model.to(DEVICE)
    if os.path.exists(RESNET_CHECKPOINT_PATH):
        checkpoint = torch.load(RESNET_CHECKPOINT_PATH, map_location=DEVICE)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model weights from {RESNET_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise RuntimeError("Failed to load checkpoint weights")
    else:
        raise FileNotFoundError(f"No checkpoint found at {RESNET_CHECKPOINT_PATH}")
    model.eval()

    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

            accuracy = 100 * correct / total
            progress_bar.set_description("Evaluating")
            progress_bar.set_postfix(accuracy=f"{accuracy:.2f}%")

    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(NUM_CLASSES)), yticklabels=list(range(NUM_CLASSES)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    cm_png_path = os.path.join(RESNET_FIGURE_PATH, "confusion_matrix.png")
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
    plt.savefig(cm_png_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_png_path}")

    low_accuracy_classes = []
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {class_accuracy:.2f}%")
            if class_accuracy < 100.0:
                low_accuracy_classes.append(i)
    
    if low_accuracy_classes:
        print(f"Classes with accuracy < 100%: {low_accuracy_classes}")
        filtered_cm = confusion_matrix(all_labels, all_predictions, labels=low_accuracy_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(filtered_cm, annot=True, fmt='d', cmap='Blues', xticklabels=low_accuracy_classes, yticklabels=low_accuracy_classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Filtered Confusion Matrix (Classes with Accuracy < 100%)')

        filtered_cm_png_path = os.path.join(RESNET_FIGURE_PATH, "filtered_confusion_matrix.png")
        os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)
        plt.savefig(filtered_cm_png_path)
        plt.close()
        print(f"Filtered confusion matrix saved to {filtered_cm_png_path}")

if __name__ == "__main__":
    evaluate_resnet_with_stn()
