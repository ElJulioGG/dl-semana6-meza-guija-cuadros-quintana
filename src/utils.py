"""Shared utilities: dataset loading, transforms, metrics, and plotting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
MEAN = [0.8010, 0.6530, 0.7354]
STD = [0.1379, 0.1844, 0.1247]


def get_transforms(img_size=64, augment=True):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]) if augment else get_transforms(img_size, augment=False).transforms[0]

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def get_dataloaders(data_dir, img_size=64, batch_size=32, num_workers=0):
    data_dir = Path(data_dir)
    train_tf, val_tf = get_transforms(img_size)
    train_ds = datasets.ImageFolder(data_dir / "TRAIN", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "TEST", transform=val_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl


def plot_history(history, title="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()
