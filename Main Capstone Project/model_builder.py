# Step 7: Load the Pre-trained Model, Modify the Classifier Head and Freeze the Base Layers

from transformers import ViTForImageClassification

def load_model(num_labels: int):
    """
    Load a pre-trained Vision Transformer model with a custom classifier head.

    Args:
        num_labels (int): Number of output classes for the classifier head.

    Returns:
        ViTForImageClassification: The modified model.
    """
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
