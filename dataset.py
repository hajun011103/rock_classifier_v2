# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import config

# Transforms with explicit interpolation and normalization
def get_transforms(train: bool = True) -> transforms.Compose:
    base_size = (224, 224)
    transform_list = [
        transforms.Resize(base_size, interpolation=transforms.InterpolationMode.BILINEAR)
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ]
    return transforms.Compose(transform_list)

# Stratified K-Fold dataloaders
def stratified_kfold_dataloaders():
    # Enable cuDNN autotuner for fixed-size inputs
    torch.backends.cudnn.benchmark = True

    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)

    # Load base dataset once for targets
    base_ds = datasets.ImageFolder(root=config.TRAIN_DIR, transform=None)
    labels = np.array(base_ds.targets)

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=42
    )

    loaders = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        # Wrap subsets with desired transforms
        train_ds = Subset(
            datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_tf),
            train_idx
        )
        val_ds = Subset(
            datasets.ImageFolder(root=config.TRAIN_DIR, transform=val_tf),
            val_idx
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=getattr(config, 'NUM_WORKERS', 8),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 2),
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=getattr(config, 'NUM_WORKERS', 8),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=getattr(config, 'PREFETCH_FACTOR', 2)
        )

        loaders.append((train_loader, val_loader))

    return loaders

# Test dataloader
class TestDataset(TorchDataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['filename']
        path = os.path.join(self.img_dir, fname)
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def make_test_dataloader():
    test_tf = get_transforms(train=False)
    test_ds = TestDataset(
        csv_file=config.TEST_CSV,
        img_dir=config.TEST_DIR,
        transform=test_tf
    )
    return DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(config, 'NUM_WORKERS', 8),
        pin_memory=True
    )
