# dataset.py (DatasetLoader, transform, augmentation, K-Fold)
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold
import pandas as pd
import config

# One Hot Encoding
class OneHotDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, idxs):
        self.base = base_ds
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        img, label = self.base[self.idxs[i]]
        # One-hot encoding
        target = torch.zeros(config.NUM_CLASSES, dtype=torch.float)
        target[label] = 1.0
        return img, target

# Transform
def get_transforms(train=True):
    aug = [transforms.Resize((224, 224))]
    if train:
        aug.append(transforms.RandomHorizontalFlip())
    aug += [transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    return transforms.Compose(aug)

# K-Fold
def make_kfold_dataloaders():
    # ImageFolder expects TRAIN_DIR with subfolders per class
    full_ds = datasets.ImageFolder(root=config.TRAIN_DIR, transform=get_transforms(train=True))
    labels = np.array(full_ds.targets)

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    folds = []
    etc_label = full_ds.class_to_idx[config.ETC_NAME]

    for train_idx, val_idx in kf.split(labels):
        # etc 클래스 제외하고 가중치 계산
        train_labels = labels[train_idx]
        counts = np.bincount(train_labels[train_labels != etc_label], minlength=config.NUM_CLASSES)
        weights = 1.0 / np.where(counts > 0, counts, 1)
        sample_weights = [weights[train_labels[i]] if train_labels[i] != etc_label else 0.0 for i in range(len(train_idx))]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_ds = OneHotDataset(full_ds, train_idx)
        val_ds   = OneHotDataset(full_ds, val_idx)

        train_loader = DataLoader(train_ds,
                                  batch_size=config.BATCH_SIZE,
                                  sampler=sampler,
                                  num_workers=8,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  prefetch_factor=6)
        val_loader = DataLoader(val_ds,
                                batch_size=config.BATCH_SIZE,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=6)
        folds.append((train_loader, val_loader))
    return folds


def make_test_dataloader():
    # 테스트용 DataLoader (레이블 없음)
    test_files = pd.read_csv(config.TEST_CSV)
    dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=get_transforms(train=False))
    loader = DataLoader(dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True)
    return loader