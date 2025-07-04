import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import IntegratedGradients, Saliency
from .inference import model, CLASS_NAMES  # Import your loaded model from inference.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.interpret.captum_visualizer import plot_attributions


# 🔁 Consistent transform for attribution
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def explain_image(pil_image, target_idx, method="Integrated Gradients"):
    try:
        # Convert input image to tensor
        tensor = transform(pil_image).unsqueeze(0)

        # # Apply Integrated Gradients
        # ig = IntegratedGradients(model)
        # attributions = ig.attribute(tensor, target=target_idx, n_steps=200)
        if method == "Saliency":
            explainer = Saliency(model)
            attributions = explainer.attribute(tensor, target=target_idx)
        else:
            explainer = IntegratedGradients(model)
            attributions = explainer.attribute(tensor, target=target_idx, n_steps=50)
            

        # attributions = explainer.attribute(tensor, target=target_idx, n_steps=200 if method == "Integrated Gradients" else None)
        
            

        # Generate attribution visualization using your custom Captum utility
        fig = plot_attributions(
            attributions=attributions,
            original_image=tensor,
            target_label=CLASS_NAMES[target_idx],
            save_path=None
        )

        # Convert Matplotlib figure to NumPy array for Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        heatmap = np.array(Image.open(buf))
        plt.close(fig)

        return heatmap

    except Exception as e:
        print("❌ explain_image error:", e)
        raise RuntimeError(f"Captum explanation failed: {e}")