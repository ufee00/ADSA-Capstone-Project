# Step 3: Visualize Samples from Each Class

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_samples(path, num_samples=3):
    """
    Visualize a few sample images from each class.

    Args:
        path (str): Path to the dataset folder (train/test).
        num_samples (int): Number of samples to visualize per class.
    """
    for class_folder in os.listdir(path):
        class_folder_path = os.path.join(path, class_folder)
        if os.path.isdir(class_folder_path):
            print(f"\nClass: {class_folder}")
            images = os.listdir(class_folder_path)
            available_samples = min(len(images), num_samples)
            if available_samples == 0:
                print("No images found in this class.")
                continue

            sample_images = random.sample(images, available_samples)

            # Handle single or multiple subplots
            if available_samples == 1:
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                img_path = os.path.join(class_folder_path, sample_images[0])
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(class_folder)
            else:
                fig, axes = plt.subplots(1, available_samples, figsize=(15, 5))
                for i, img_name in enumerate(sample_images):
                    img_path = os.path.join(class_folder_path, img_name)
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    axes[i].set_title(class_folder)
            plt.show()

# Step 4: Plot Class Distribution

def plot_class_distribution(distribution, title):
    """
    Plot a bar chart for class distribution.

    Args:
        distribution (dict): Dictionary with class names as keys and image counts as values.
        title (str): Title of the chart.
    """
    classes = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Step 5: Drop Images to Balance Classes: Downsample to 515 Images per Class

def drop_images(class_folder, keep_count):
    images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)

    if current_count <= keep_count:
        print(f"Class '{os.path.basename(class_folder)}' already has {current_count} images (<= {keep_count}). No images dropped.")
        return

    drop_count = current_count - keep_count
    images_to_drop = random.sample(images, drop_count)

    for img_name in images_to_drop:
        img_path = os.path.join(class_folder, img_name)
        os.remove(img_path)

    print(f"Dropped {drop_count} images from class: '{os.path.basename(class_folder)}'. Retained {keep_count} images.")

# Run this to balance each class in the training set to 515 images
train_path = 'maize_leaf_split_dataset/Train'
keep_count = 515

for class_name in os.listdir(train_path):
    class_folder = os.path.join(train_path, class_name)
    if os.path.isdir(class_folder):
        drop_images(class_folder, keep_count)
