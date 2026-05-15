"""Model definitions: LeNet, VGG-11 (reduced), and transfer learning wrappers."""
import torch.nn as nn
from torchvision import models


class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class VGG11Small(nn.Module):
    """VGG-11 with reduced filters, suitable for 64x64 input."""
    def __init__(self, num_classes=4):
        super().__init__()
        def block(in_ch, out_ch):
            return [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(
            *block(3, 32), *block(32, 64), *block(64, 128), *block(128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_resnet18(num_classes=4, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_vgg16(num_classes=4, freeze_backbone=True):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
