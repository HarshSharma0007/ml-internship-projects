from torchvision import models
import torch.nn as nn
import torch
from src.config import NUM_CLASSES, DEVICE

def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)
