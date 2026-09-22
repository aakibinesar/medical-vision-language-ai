from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size, train):
    if train:
        # No horizontal flip: chest X-ray anatomy isn't left-right symmetric
        # (heart, aortic arch), and flipping mirrors embedded laterality
        # markers ("L"/"R" stickers) into unreadable nonsense - plausibly why
        # Grad-CAM was found attending to an "L" marker in one case.
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CXRDataset(Dataset):
    """Reads a CSV with columns: path,<label1>,<label2>,... (values 0/1).

    Works for both single-label (binary) and multilabel classification —
    `n_labels` is just `len(label_columns)`.
    """

    def __init__(self, csv_path, img_root, img_size=224, train=False, label_columns=None):
        df = pd.read_csv(csv_path)
        self.label_columns = label_columns or [c for c in df.columns if c != "path"]
        self.paths = df["path"].tolist()
        self.labels = df[self.label_columns].values.astype(np.float32)
        self.img_root = Path(img_root)
        self.tf = build_transforms(img_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.img_root / self.paths[i]).convert("RGB")
        x = self.tf(img)
        y = torch.from_numpy(self.labels[i])
        return x, y
