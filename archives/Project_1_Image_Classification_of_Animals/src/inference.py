# src/inference.py

import cv2
import torch
import mlflow.pytorch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def load_transform():
    return Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_image(image_path, model_uri, class_names):
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = load_transform()
    tensor = transform(image=image)["image"].unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()

    return class_names[pred]
