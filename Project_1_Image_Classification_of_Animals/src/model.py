# src/model.py

import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=14, pretrained=True, freeze=True):
    model = models.resnet18(pretrained=pretrained)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
