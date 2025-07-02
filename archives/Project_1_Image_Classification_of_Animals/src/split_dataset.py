# Split_dataset

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    class_names = os.listdir(source_dir)

    for cls in tqdm(class_names, desc="Processing classes"):
        src_folder = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        for split, split_files in splits.items():
            split_cls_dir = os.path.join(dest_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for f in split_files:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(split_cls_dir, f)
                shutil.copy2(src_path, dst_path)

    print("✅ Dataset split complete. Your data is ready in:", dest_dir)


source_path = r'./data/raw'  # this is where you copied your original folders
dest_path = r'./data/processed'  # the folders train/ val/ test/ will be created here

split_dataset(source_path, dest_path)
