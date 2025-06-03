import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import os
import torch

def predict_and_plot_random_images(
    model,
    test_dir: str,
    class_names: list,
    transform: transforms.Compose,
    device: torch.device = torch.device("cpu"),
    num_images: int = 5
) -> None:
    """
    Picks random images from the test set, performs predictions, and plots the images with predicted and actual labels.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_dir (str): Path to test dataset directory.
        class_names (list): List of class names.
        transform (transforms.Compose): Transform to apply to images.
        device (torch.device): Torch device to use (CPU or CUDA).
        num_images (int): Number of random images to predict and show.
    """
    model.eval()

    for _ in range(num_images):
        # Choose a random class and a random image within that class
        class_name = random.choice(class_names)
        class_folder = os.path.join(test_dir, class_name)
        image_name = random.choice(os.listdir(class_folder))
        image_path = os.path.join(class_folder, image_name)

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.inference_mode():
            output = model(image_tensor)
            pred_label = torch.argmax(output.logits, dim=1).item()

        # Plot
        plt.figure()
        plt.imshow(image)
        plt.title(f"Actual: {class_name} | Predicted: {class_names[pred_label]}")
        plt.axis("off")
        plt.show()
