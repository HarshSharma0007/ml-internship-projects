# src/utils.py

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_class_names():
    return sorted([
        "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
        "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
    ])
