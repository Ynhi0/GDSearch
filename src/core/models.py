"""
Neural network models for MNIST and CIFAR-10 using PyTorch.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    A simple 2-layer MLP for MNIST classification.
    Architecture: Flatten -> Linear(hidden) -> ReLU -> Linear(num_classes)
    """

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 28, 28)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10.
    Architecture: Conv(32)->ReLU->MaxPool -> Conv(64)->ReLU->MaxPool -> FC(128)->ReLU -> FC(10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ConvNet(nn.Module):
    """
    Stronger ConvNet for CIFAR-10 with batch normalization and dropout.
    Architecture:
        Conv(64)->BN->ReLU->Conv(64)->BN->ReLU->MaxPool->Dropout
        Conv(128)->BN->ReLU->Conv(128)->BN->ReLU->MaxPool->Dropout
        FC(256)->BN->ReLU->Dropout->FC(10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        
        # Block 1: 2 conv layers with 64 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x16x16
            nn.Dropout2d(p=dropout)
        )
        
        # Block 2: 2 conv layers with 128 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x8x8
            nn.Dropout2d(p=dropout)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x
