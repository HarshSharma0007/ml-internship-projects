import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import torch

# Build the absolute model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "resnet18_20250702_155546.pt")

# MODEL_PATH = "models/resnet18_20250702_155546.pt"
# 🧠 Load model
# model = models.resnet18(weights=False, num_classes=15)
# # state_dict = torch.load(MODEL_PATH, map_location="cpu")
# state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = models.resnet18(weights=None, num_classes=15)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)


model.load_state_dict(state_dict)
model.eval()

# 🏷️ Class names
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin",
    "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda",
    "Tiger", "Zebra"
]

# 🧼 Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_class(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
    top = {CLASS_NAMES[i]: float(probs[i]) for i in probs.argsort(descending=True)[:5]}
    return CLASS_NAMES[torch.argmax(probs)], top
