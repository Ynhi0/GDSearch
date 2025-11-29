#!/usr/bin/env python3
"""
Simple GDSearch Kaggle Benchmark - Standalone Script
Run this directly on Kaggle without importing the full repository.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_simple_model():
    """Simple MLP for MNIST"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def run_quick_benchmark():
    """Run a quick benchmark comparing SGD and Adam on MNIST"""
    print("🚀 GDSearch Quick Kaggle Benchmark")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    results = []

    for optimizer_name in ['SGD', 'Adam']:
        print(f"\n🎯 Testing {optimizer_name}")

        for seed in [42, 123, 456]:
            set_seed(seed)

            model = create_simple_model().to(device)
            criterion = nn.CrossEntropyLoss()

            if optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Quick training (3 epochs)
            start_time = time.time()

            for epoch in range(3):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            accuracy = 100. * correct / total
            training_time = time.time() - start_time

            results.append({
                'optimizer': optimizer_name,
                'seed': seed,
                'test_accuracy': accuracy,
                'training_time': training_time
            })

            print(".2f")

    # Save results
    os.makedirs('/kaggle/working/results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('/kaggle/working/results/quick_benchmark_results.csv', index=False)

    print("\n💾 Results saved to /kaggle/working/results/quick_benchmark_results.csv")

    # Summary
    summary = df.groupby('optimizer')['test_accuracy'].agg(['mean', 'std']).round(2)
    print("\n📊 Summary:")
    print(summary)

    return df

if __name__ == "__main__":
    run_quick_benchmark()
    print("\n✅ Quick benchmark completed!")
