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


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Architecture:
        Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> [+shortcut] -> ReLU
    """
    expansion = 1  # Output channels = input channels * expansion
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path (identity or projection)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).
    
    Architecture:
        - Conv1: 3->64, 3x3 (no pooling for CIFAR-10)
        - Layer1: 2 BasicBlocks, 64 channels
        - Layer2: 2 BasicBlocks, 128 channels, stride=2
        - Layer3: 2 BasicBlocks, 256 channels, stride=2
        - Layer4: 2 BasicBlocks, 512 channels, stride=2
        - AvgPool -> FC(num_classes)
    
    Total: 18 layers (1 conv + 8*2 conv in blocks + 1 fc)
    Parameters: ~11M for CIFAR-10
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        
        self.in_channels = 64
        self.dropout = dropout
        
        # Initial convolution (no pooling for small CIFAR-10 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)   # 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 4x4
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a layer with multiple residual blocks.
        
        Args:
            out_channels: Number of output channels
            num_blocks: Number of residual blocks in this layer
            stride: Stride for the first block (for downsampling)
        """
        downsample = None
        
        # If dimensions change, need projection shortcut
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
        layers = []
        # First block (may downsample)
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
