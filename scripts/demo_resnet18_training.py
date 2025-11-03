"""
Demo script for training ResNet-18 on CIFAR-10 with custom optimizers.

This demonstrates that custom optimizers work with deep networks that have
skip connections and complex architectures.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.core.models import ResNet18
from src.core.data_utils import get_cifar10_loaders
from src.core.pytorch_optimizers import SGDWrapper, SGDMomentumWrapper, AdamWrapper, RMSPropWrapper


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd_momentum',
                       choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop'],
                       help='Optimizer choice')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Print header
    print("=" * 80)
    print("ResNet-18 Training on CIFAR-10")
    print("=" * 80)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    print()
    
    # Create model
    print("Creating ResNet-18 model...")
    model = ResNet18(num_classes=10, dropout=args.dropout).to(device)
    num_params = model.get_num_parameters()
    print(f"✓ Parameters: {num_params:,}")
    print()
    
    # Create optimizer
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Learning rate: {args.lr}")
    print()
    
    if args.optimizer == 'sgd':
        optimizer = SGDWrapper(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd_momentum':
        optimizer = SGDMomentumWrapper(model.parameters(), lr=args.lr, beta=0.9)
    elif args.optimizer == 'adam':
        optimizer = AdamWrapper(model.parameters(), lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSPropWrapper(model.parameters(), lr=args.lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("=" * 80)
    print("Training...")
    print("=" * 80)
    print()
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Test
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"✓ New best test accuracy!")
        
        print()
    
    # Final summary
    elapsed_time = time.time() - start_time
    print("=" * 80)
    print("Training Complete!")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == '__main__':
    main()
