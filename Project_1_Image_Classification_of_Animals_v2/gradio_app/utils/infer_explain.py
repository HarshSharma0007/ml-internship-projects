import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import os
import sys
import gradio as gr
from captum.attr import IntegratedGradients, Saliency

# 🧠 Local imports
sys.path.append(os.path.abspath(".."))
from src.interpret.captum_visualizer import plot_attributions

# 📦 Load model
model = models.resnet18(pretrained=False, num_classes=15)
state_dict = torch.load("models/resnet18_20250702_155546.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 🧼 Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 🐾 Class names
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin",
    "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda",
    "Tiger", "Zebra"
]

def classify_and_explain(pil_image, method="Integrated Gradients"):
    try:
        if pil_image is None:
            return None, None

        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image.astype("uint8")).convert("RGB")

        tensor = transform(pil_image).unsqueeze(0)

        # 🔮 Predict class
        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, 3)
            label_probs = {
                CLASS_NAMES[i]: float(top_probs[j])
                for j, i in enumerate(top_indices)
            }

        predicted_class = torch.argmax(output, dim=1).item()

        # 🔍 Attribution method selection
        if method == "Saliency":
            explainer = Saliency(model)
            attributions = explainer.attribute(tensor, target=predicted_class)
        else:
            explainer = IntegratedGradients(model)
            attributions = explainer.attribute(tensor, target=predicted_class, n_steps=50)

        # 🎨 Generate heatmap
        fig = plot_attributions(
            attributions=attributions,
            original_image=tensor,
            target_label=CLASS_NAMES[predicted_class],
            save_path=None
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        heatmap = np.array(Image.open(buf))

        return label_probs, heatmap

    except Exception as e:
        print("❌ classify_and_explain error:", e)
        raise gr.Error(f"Something went wrong: {e}")