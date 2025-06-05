import torch
from typing import Tuple, Dict
from timeit import default_timer as timer
import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor

# Set the number of labels for classification
num_labels = 4

model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
# Load the pre-trained model weights
model.load_state_dict(torch.load("saved model/maize_leaf_disease_model.pth", map_location='cpu'))

labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

model.eval()

def predict(img) -> Tuple[Dict[str, float], float]:
    start_time = timer()

    try:
        # Preprocess the image using your processor
        inputs = processor(images=img, return_tensors="pt")

        # Model prediction
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_probs = torch.softmax(logits, dim=1)

        # Return class probabilities as decimals (0.0 to 1.0)
        pred_dict = {
            labels[i]: round(float(pred_probs[0][i]), 4) for i in range(len(labels))
        }

        pred_time = round(timer() - start_time, 5)

        return pred_dict, pred_time

    except Exception as e:
        print(f"[ERROR] {e}")
        return {"Error": 0.0}, 0.0

title = "Maize Leaf Disease Detector 🌽"
description = "Upload a maize leaf image to detect disease. Output shows prediction confidence (%) for each class."
article = "Deep Learning - Capstone Project: Arewa Data Science Academy"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=4, label="Predictions (%)"),
        gr.Number(label="Prediction time (s)")
    ],
    title=title,
    description=description,
    article=article
)

if __name__ == "__main__":
    demo.launch()