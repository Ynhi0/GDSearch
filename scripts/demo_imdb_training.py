"""
Demo: Train NLP model on IMDB sentiment analysis

Quick demonstration of training RNN/LSTM models on IMDB dataset
with our custom optimizers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.core.nlp_data_utils import get_imdb_loaders
from src.core.nlp_models import SimpleLSTM, BiLSTM, TextCNN
from src.core.pytorch_optimizers import AdamWrapper, SGDMomentumWrapper
import argparse
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        indices = batch['indices'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        
        # Forward pass
        outputs = model(indices, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update weights with optimizer
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            indices = batch['indices'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            outputs = model(indices, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train NLP model on IMDB')
    parser.add_argument('--model', type=str, default='SimpleLSTM',
                       choices=['SimpleLSTM', 'BiLSTM', 'TextCNN'],
                       help='Model architecture')
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['Adam', 'SGDMomentum'],
                       help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--train-size', type=int, default=5000,
                       help='Training set size (use subset for speed)')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Test set size')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--max-len', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set device and seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("IMDB Sentiment Analysis Training")
    print("="*80)
    train_loader, test_loader, vocab = get_imdb_loaders(
        batch_size=args.batch_size,
        max_vocab_size=args.vocab_size,
        max_len=args.max_len,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Create model
    vocab_size = len(vocab)
    if args.model == 'SimpleLSTM':
        model = SimpleLSTM(vocab_size, hidden_size=args.hidden_size)
    elif args.model == 'BiLSTM':
        model = BiLSTM(vocab_size, hidden_size=args.hidden_size)
    elif args.model == 'TextCNN':
        model = TextCNN(vocab_size)
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}")
    print(f"Parameters: {n_params:,}")
    
    # Create optimizer
    if args.optimizer == 'Adam':
        optimizer = AdamWrapper(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGDMomentum':
        optimizer = SGDMomentumWrapper(model.parameters(), lr=args.lr, momentum=0.9)
    
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "="*80)
    print("Training...")
    print("="*80)
    
    best_test_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"✓ New best test accuracy!")
    
    print("\n" + "="*80)
    print(f"Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
