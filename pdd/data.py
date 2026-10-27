"""Datasets and image transforms."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def plantvillage_classes():
    """The 38 PlantVillage class names in a fixed (sorted) order."""
    df = pd.read_csv(ROOT / "data/splits/plantvillage_grouped.csv", usecols=["label"])
    return sorted(df["label"].unique())


class ImageList(Dataset):
    """Images listed as (path relative to root, label name)."""

    def __init__(self, root, paths, labels, classes, transform):
        self.root = Path(root)
        self.paths = list(paths)
        self.index = {c: i for i, c in enumerate(classes)}
        self.targets = [self.index[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.targets[i]


def train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def strong_train_transform(size=224):
    """Heavier changes to how a photo looks (framing, colour, contrast, blur,
    occlusion), meant to stop the model relying on PlantVillage's single
    capture setting. Fixed before any step-4 result was seen."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.25, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.TrivialAugmentWide(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05),
        T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25),
    ])


def eval_transform(size=224):
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def plantvillage_split(split, transform, split_col="split"):
    df = pd.read_csv(ROOT / "data/splits/plantvillage_grouped.csv", keep_default_na=False)
    df = df[df[split_col] == split]
    return ImageList(ROOT / "data/raw/plantvillage", df["path"], df["label"],
                     plantvillage_classes(), transform)


def plantdoc_split(split, transform):
    """PlantDoc images of one split of data/splits/plantdoc_grouped.csv, labelled
    with their PlantVillage class."""
    df = pd.read_csv(ROOT / "data/splits/plantdoc_grouped.csv", keep_default_na=False)
    df = df[df["split"] == split]
    return ImageList(ROOT / "data/raw/plantdoc", df["path"], df["label"],
                     plantvillage_classes(), transform)


def seed_worker(worker_id):
    torch.manual_seed(torch.initial_seed() % 2**31 + worker_id)
