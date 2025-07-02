import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = glob(os.path.join(img_dir, "*", "*"))
        self.transform = transform
        self.classes = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.classes.index(os.path.basename(os.path.dirname(img_path)))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(),
            ToTensorV2()
        ])

def get_dataloaders():
    train_ds = ImageDataset(TRAIN_DIR, transform=get_transforms(train=True))
    val_ds = ImageDataset(VAL_DIR, transform=get_transforms(train=False))
    test_ds = ImageDataset(TEST_DIR, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader