import os
import torch

# === Directory Paths === #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

# === Device === #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Image + Training Settings === #
IMAGE_SIZE = 224  # Standard input size for ResNet/EfficientNet
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

# === Model Settings === #
NUM_CLASSES = 15        # Update this based on your actual number of classes
USE_ENSEMBLE = True     # Toggle ensemble training

# === Misc Settings === #
SAVE_BEST_ONLY = True
USE_MLFLOW = True       # Enable if tracking