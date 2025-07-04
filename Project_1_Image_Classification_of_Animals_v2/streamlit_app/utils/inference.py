import torch
from torchvision import models, transforms
from src.interpret.captum_visualizer import get_attributions
import numpy as np

model = models.resnet18(pretrained=False, num_classes=15)
model.load_state_dict(torch.load("models/resnet18_20250702_155546.pt", map_location="cpu"))
model.eval()

CLASS_NAMES = [...]

transform = transforms.Compose([...])

def predict_class(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
    top = {CLASS_NAMES[i]: float(probs[i]) for i in probs.argsort(descending=True)[:5]}
    return CLASS_NAMES[torch.argmax(probs)], top
