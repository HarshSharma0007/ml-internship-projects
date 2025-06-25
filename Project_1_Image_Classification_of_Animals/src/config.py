# src/config.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BEST_MODEL_PATH = "checkpoints/best_model.pth"
TEST_DATA_DIR = "data/processed/test"
MLFLOW_TRACKING_URI = "file:./mlruns"

# Hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 15
