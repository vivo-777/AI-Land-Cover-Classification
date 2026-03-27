"""
Model Definition
=================
Builds a transfer-learning-based land cover classifier
using a pretrained ResNet-18 backbone.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-18 model with a custom classification head.

    Args:
        num_classes: number of output land cover classes (default: 5)
        pretrained:  whether to load ImageNet pretrained weights

    Returns:
        model: PyTorch model ready for training
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final fully-connected layer for our classification task
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_device() -> torch.device:
    """Select GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    device = get_device()
    model = build_model().to(device)
    print(f"Model loaded on: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
