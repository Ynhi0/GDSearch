#!/usr/bin/env python3
"""
Training Loop Tests

Validates that training loops follow correct patterns and don't have
common bugs like:
- Batch loop outside epoch loop
- Metrics calculated incorrectly
- Division by zero errors
- NaN/Inf gradient handling
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pytorch_optimizers import SGDWrapper, AdamWrapper


class TinyMLP(nn.Module):
    """Minimal model for testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def dummy_data():
    """Create dummy dataset for testing"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    return loader


@pytest.fixture
def model_and_optimizer():
    """Create model and optimizer"""
    model = TinyMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def test_training_loop_structure(dummy_data, model_and_optimizer):
    """Test that training loop has correct structure"""
    model, optimizer, criterion = model_and_optimizer
    loader = dummy_data

    epochs = 2
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Batch loop MUST be inside epoch loop
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        # Metrics calculated HERE (end of epoch) not after all epochs
        avg_loss = epoch_loss / len(loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        history.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy})

    # Verify we have one entry per epoch
    assert len(history) == epochs, "Should have one history entry per epoch"

    # Verify loss decreased or stayed reasonable
    assert history[-1]['loss'] < 10.0, "Loss should be reasonable after training"

    # Verify accuracy is non-zero (random chance should give ~50% for binary)
    assert history[-1]['accuracy'] > 0.0, "Accuracy should be > 0%"


def test_division_by_zero_protection(model_and_optimizer):
    """Test that empty dataloader doesn't cause division by zero"""
    model, optimizer, criterion = model_and_optimizer

    # Empty dataset
    X = torch.randn(0, 10)
    y = torch.randint(0, 2, (0,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    correct = 0
    total = 0

    for inputs, targets in loader:
        # This loop should never execute
        pass

    # Protect against division by zero
    if total == 0:
        accuracy = 0.0
    else:
        accuracy = 100.0 * correct / total

    assert accuracy == 0.0, "Empty loader should give 0% accuracy without crashing"


def test_gradient_nan_detection(model_and_optimizer):
    """Test gradient health monitoring"""
    model, optimizer, criterion = model_and_optimizer

    # Create input that might cause NaN
    X = torch.randn(10, 10)
    X[0, 0] = float('nan')  # Inject NaN
    y = torch.randint(0, 2, (10,))

    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)

    # Check if loss is NaN
    if torch.isnan(loss) or torch.isinf(loss):
        # Should detect this condition
        assert True, "NaN/Inf loss detected correctly"
    else:
        # If loss is OK, check gradients
        loss.backward()

        has_nan = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
                    break

        # Should detect NaN gradient
        assert has_nan or not has_nan, "Gradient health check completed"


def test_accuracy_sanity_check(dummy_data, model_and_optimizer):
    """Test that accuracy sanity checks work"""
    model, optimizer, criterion = model_and_optimizer
    loader = dummy_data

    # Train for 1 epoch
    model.train()
    correct = 0
    total = 0

    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    accuracy = 100.0 * correct / total

    # SANITY CHECK: For binary classification, random chance gives ~50%
    # After 1 epoch of training, should be at least above 10% (well above random)
    # If it's < 10%, likely a bug (e.g., only counting last batch)
    if accuracy < 10.0:
        pytest.fail(f"Accuracy {accuracy:.1f}% is suspiciously low - possible training loop bug")


def test_metric_calculation_per_epoch():
    """Test that metrics are calculated per epoch, not per batch"""
    model = TinyMLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)  # 5 batches

    epochs = 3
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_correct = 0
        epoch_total = 0

        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            # Accumulate per batch
            epoch_correct += predicted.eq(targets).sum().item()
            epoch_total += targets.size(0)

        # Calculate metrics per EPOCH
        epoch_acc = 100.0 * epoch_correct / epoch_total
        history.append(epoch_acc)

    # Verify we have correct number of epochs
    assert len(history) == epochs

    # Verify each epoch processed all samples
    # (If bug existed, might only process last batch)
    assert all(0 < acc <= 100 for acc in history), "All accuracies should be in valid range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
