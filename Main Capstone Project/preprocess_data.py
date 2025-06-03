# Step 1: Create a directory structure for the dataset

import os
import shutil
import random
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from PIL import Image

def split_dataset_by_class(
    source_dir: str,
    output_base_dir: str,
    split_ratios: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42
) -> None:
    """
    Splits a dataset organized in class folders into train, valid, and test sets.

    Args:
        source_dir (str): Path to the original dataset with class subfolders.
        output_base_dir (str): Path to the output folder where splits will be saved.
        split_ratios (Tuple[float, float, float], optional): Train, valid, test ratios. Defaults to (0.9, 0.05, 0.05).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    splits = ['Train', 'Valid', 'Test']
    random.seed(seed)

    # Create split folders
    for split in splits:
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if os.path.isdir(class_dir):
                split_class_dir = os.path.join(output_base_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)

    # Process each class
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', 'jpeg', '.png'))]
        total = len(images)
        random.shuffle(images)

        train_end = int(split_ratios[0] * total)
        valid_end = train_end + int(split_ratios[1] * total)

        split_data = {
            'Train': images[:train_end],
            'Valid': images[train_end:valid_end],
            'Test': images[valid_end:]
        }

        for split in splits:
            for img_name in tqdm(split_data[split], desc=f"Copying {split} images for '{class_name}'"):
                src = os.path.join(class_path, img_name)
                dst = os.path.join(output_base_dir, split, class_name, img_name)
                shutil.copyfile(src, dst)

    print("\n Dataset split completed!") 

# Step 2: Check the Distribution of Images Across Classes
def get_class_distribution(path):
    """
    Get the distribution of images across different classes.

    Args:
        path (str): Path to the dataset folder (train/valid/test).

    Returns:
        dict: Dictionary with class names as keys and number of images as values.
    """
    class_counts = {}
    for class_folder in os.listdir(path):
        class_folder_path = os.path.join(path, class_folder)
        if os.path.isdir(class_folder_path):
            num_images = len(os.listdir(class_folder_path))
            class_counts[class_folder] = num_images
    return class_counts

