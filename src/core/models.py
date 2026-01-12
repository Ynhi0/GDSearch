"""
Neural network models for MNIST and CIFAR-10 using PyTorch.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    A universal MLP for MNIST/CIFAR classification.
    Architecture: Flatten -> [Linear -> [BN] -> ReLU -> [Dropout]]* -> Linear(num_classes)

    UNIVERSAL COMPATIBILITY: Accepts BOTH:
    - hidden_size (int): Single hidden layer [DEPRECATED for multi-layer]
    - hidden_dims (list): Multiple hidden layers [RECOMMENDED]

    This dual interface allows migration from run_all_kaggle.py without breaking changes.

    Added use_bn parameter to control Batch Normalization.
    This prevents confounding variables when comparing optimizers:
    - SGD benefits greatly from BN (stabilizes gradients)
    - Adam works well without BN (adaptive scaling handles it)
    Without this flag, SGD vs Adam comparisons confound optimizer with normalization.
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        input_dim: Optional[int] = None,  # Alias for backward compatibility
        hidden_size: Optional[int] = None,
        hidden_dims: Optional[list] = None,
        num_classes: int = 10,
        dropout: float = 0.0,
        use_bn: bool = False
    ):
        super().__init__()

        # Handle input_size vs input_dim alias
        if input_size is None and input_dim is None:
            input_size = 28 * 28  # Default for MNIST
        elif input_size is None:
            input_size = input_dim

        # Handle hidden_size (int) vs hidden_dims (list)
        # Priority: hidden_dims > hidden_size > default
        if hidden_dims is not None:
            self.hidden_layers = hidden_dims
        elif hidden_size is not None:
            self.hidden_layers = [hidden_size]
        else:
            self.hidden_layers = [256]  # Default single hidden layer

        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.use_bn = use_bn

        # Build network dynamically
        layers = []
        # Type safety: input_size is guaranteed to be int here
        assert input_size is not None, "input_size must be specified"
        prev_dim: int = input_size

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Optional Batch Normalization
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            # Optional Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 28, 28) or (N, input_size)
        # Flatten to (N, input_size)
        x = torch.flatten(x, 1)
        return self.network(x)


class SimpleCNN(nn.Module):
    """
    A simple, configurable CNN compatible with MNIST/CIFAR.
    Architecture: Conv(in_channels->32)->Act->MaxPool -> Conv(32->64)->Act->MaxPool -> FC(128)->Act -> FC(num_classes)

    Configurable options:
      - in_channels: number of input channels (1 for MNIST, 3 for CIFAR)
      - activation: 'relu' | 'tanh' | 'leaky_relu'

    Includes `apply_initialization()` helper used by tests to check initialization methods.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, activation: str = 'relu'):
        super().__init__()
        # Choose activation layer
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32xH/2 xW/2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 64xH/4 xW/4
        )

        # Use AdaptiveAvgPool2d to make classifier resolution-agnostic
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Always produces 64x1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            act,
            nn.Linear(128, num_classes),
        )

        # Expose commonly referenced layer attributes for tests and initialization helpers
        self.conv1 = self.features[0]
        self.conv2 = self.features[3]
        self.fc1 = self.classifier[1]
        self.fc2 = self.classifier[3]
        self.init_layers = [self.conv1, self.conv2, self.fc1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def apply_initialization(self, method: str) -> None:
        """Apply a named initialization method to Conv2d and Linear layers.

        Supported methods:
          - 'zero', 'uniform_small', 'normal_small', 'xavier_uniform',
            'xavier_normal', 'kaiming_uniform', 'kaiming_normal'
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if method == 'zero':
                    nn.init.constant_(m.weight, 0.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'uniform_small':
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'normal_small':
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    raise ValueError(f"Unknown initialization method: {method}")


class ConvNet(nn.Module):
    """
    Stronger ConvNet for CIFAR-10 with batch normalization and dropout.
    Architecture:
        Conv(64)->BN->ReLU->Conv(64)->BN->ReLU->MaxPool->Dropout
        Conv(128)->BN->ReLU->Conv(128)->BN->ReLU->MaxPool->Dropout
        AdaptiveAvgPool(1x1)->FC(256)->BN->ReLU->Dropout->FC(10)

    FIXED (Issue #23): Now uses AdaptiveAvgPool2d instead of hardcoded Linear(128*8*8, 256).
    This prevents crashes when using different input resolutions (e.g., MNIST 28x28).
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
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x16x16 for CIFAR, 64x14x14 for MNIST
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
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x8x8 for CIFAR, 128x7x7 for MNIST
            nn.Dropout2d(p=dropout)
        )

        # Use AdaptiveAvgPool2d instead of hardcoded shape
        # This allows the model to work with ANY input resolution (CIFAR 32x32, MNIST 28x28, etc.)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Always outputs 128x1x1

        # Classifier (now resolution-agnostic)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),  # Input is now always 128 (from adaptive pooling)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)  # Resolution-agnostic pooling
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Architecture:
        Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> [+shortcut] -> ReLU
    """
    expansion = 1  # Output channels = input channels * expansion

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
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

    EXTENDED (Issue #25): Now supports zero_init_residual for modern initialization.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.0, zero_init_residual: bool = False):
        super().__init__()

        self.in_channels = 64
        self.dropout = dropout
        self.zero_init_residual = zero_init_residual

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
        """
        Initialize weights using Kaiming initialization.

        EXTENDED (Issue #25): Implements Zero-Gamma Initialization when zero_init_residual=True.
        This initializes the last BatchNorm in each residual block to zero, making the block
        an identity function at initialization. This allows gradients to flow through the
        shortcut path unimpeded, improving trainability with SGD (not just Adam).

        Reference: "Bag of Tricks for Image Classification with CNNs" (He et al., 2018)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-Gamma Initialization for residual blocks
        # Initialize the last BN in each residual block to zero (γ=0)
        # This makes each block initially act as identity: out = x + 0 = x
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    # Zero-initialize the last BN in the residual block
                    nn.init.constant_(m.bn2.weight, 0)
                    logging.debug(f"Zero-initialized residual block: {m}")

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
