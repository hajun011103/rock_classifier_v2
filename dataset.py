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

# Transforms
class TransformedSubset(Subset):
    """
    Subset wrapper that applies a given transform to each sample.
    Avoids re-scanning the directory for each fold by reusing the base dataset.
    """
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return self.transform(img), label

def get_transforms(train: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    if train:
        transform_list = [
            # 1) 원본에서 곧바로 랜덤 크롭 + 리사이즈 (데이터 증강)
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
    else:
        transform_list = [
            # 1) 검증/테스트 시: 해상도 유지 + 중앙 크롭
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

    transform_list += [
        transforms.ToTensor(),
        normalize()
    ]
    return transforms.Compose(transform_list)

# Stratified K-Fold dataloaders
def stratified_kfold_dataloaders():

    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)

    # Load base dataset once for targets
    base_ds = datasets.ImageFolder(root=config.TRAIN_DIR, transform=None)
    labels = np.array(base_ds.targets)

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.SEED
    )

    dummy_X = np.zeros((len(labels), 1), dtype=np.float32)

    loaders = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        # Wrap subsets with desired transforms
        train_ds = TransformedSubset(base_ds, train_idx, train_tf)
        val_ds   = TransformedSubset(base_ds, val_idx,   val_tf)

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
