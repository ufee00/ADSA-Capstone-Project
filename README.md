# ADSA-Capstone-Project

# 🌽 Maize Leaf Disease Identification using Vision Transformer (ViT)

This project trains a Vision Transformer (ViT) model to identify diseases in maize leaves. The pipeline includes preprocessing, dataset splitting, model training, evaluation, and experiment tracking with Weights and Biases (WandB).

---

## 📦 Requirements

Install the required dependencies using the following command:

```bash
pip install wandb torchvision matplotlib seaborn tensorboard scikit-learn opencv-python grad-cam transformers
```

---

## 📂 Dataset

The dataset used is the [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) from Kaggle.

Download the dataset using:

```python
path = kagglehub.dataset_download("smaranjitghose/corn-or-maize-leaf-disease-dataset")
```

---

## 🧹 Preprocessing

### 🔄 Transformations

Images are normalized using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Other augmentations include:

* Resize: `256 x 256`
* RandomResizedCrop: `224`
* Brightness adjustment: `0.2`
* Contrast adjustment: `0.2`

---

## 🔀 Dataset Splitting

Use the `split_dataset_by_class()` function to split the dataset into:

* **Training**: 90%
* **Validation**: 5%
* **Test**: 5%

Function signature:

```python
split_dataset_by_class(
    source_dir: str,
    output_base_dir: str,
    split_ratios: Tuple[float, float, float],
    seed: int
)
```

---

## 🧠 Model Architecture

We use the **Vision Transformer (ViT)** for image classification, leveraging Hugging Face Transformers.

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_datasets.classes)
)
```

---

## ⚙️ Training Configuration

### Loss and Optimizer

* **Loss Function**: `CrossEntropyLoss()`
* **Optimizer**: `AdamW()`

### Weights and Biases Setup

```python
import wandb

wandb.init(
    project="maize-leaf-disease-identification",
    name="vit-training-run",
    config={
        "epochs": 20,
        "batch_size": 16,
        "optimizer": "Adam",
        "architecture": "ViT-base-patch16-224"
    }
)
```

---

## 🏋️‍♂️ Training the Model

Train the model using the `train_and_validate()` function:

```python
train_and_validate(model, trainloader, validloader, criterion, optimizer, device, epochs)
```

---

## 💾 Save the Model

After training, save the model using:

```python
save_model(model, save_path)
```

---

## 🧪 Evaluation

Evaluate the model using the `evaluate_model()` function:

```python
evaluate_model(model, testloader, test_datasets, device)
```

---

## 📊 Experiment Tracking

Training metrics including loss, accuracy, and learning curves are recorded using **Weights and Biases (WandB)** for better visibility and experiment management.

---

## 📁 Summary

| Component       | Value                                          |
| --------------- | ---------------------------------------------- |
| Model           | ViT Base Patch16 (google/vit-base-patch16-224) |
| Dataset         | Corn or Maize Leaf Disease Dataset             |
| Frameworks      | PyTorch, Hugging Face Transformers             |
| Monitoring Tool | Weights and Biases (wandb)                     |
| Optimizer       | AdamW                                          |
| Loss Function   | CrossEntropyLoss                               |


