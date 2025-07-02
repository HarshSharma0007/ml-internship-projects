# src/data_loader.py

import os
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize

class AlbumentationsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label



def get_test_loader(data_dir, batch_size=32, num_workers=2):
    transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = AlbumentationsDataset(root_dir=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
