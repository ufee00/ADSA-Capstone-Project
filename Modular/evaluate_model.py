import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, testloader, test_datasets, device):
    """
    Evaluate the model on the test set and print metrics.

    Args:
        model: Trained model.
        testloader: DataLoader for the test set.
        test_datasets: ImageFolder dataset for the test set (for class names).
        device: Device to run evaluation on.
    """

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs.logits, dim=1)

            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    class_names = test_datasets.classes

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # ROC-AUC Score
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        print(f"ROC-AUC Score (OvR): {roc_auc:.4f}")
    except ValueError:
        print("ROC-AUC could not be calculated (might need more than one class in test batch).")
