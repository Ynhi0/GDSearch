#!/usr/bin/env python3
"""
Minimal CIFAR-10 multi-seed runner with multiple optimizers.
Outputs per-run CSVs compatible with the project's result conventions.
Designed to be simple and Kaggle-friendly if needed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class SimpleCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(batch_size: int = 128):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = os.environ.get('DATA_ROOT', './data')
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def run_single(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    trainloader, testloader = get_loaders(batch_size)
    model = SimpleCIFARNet().to(device)

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    hist = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, trainloader, optimizer, device)
        te_loss, te_acc = evaluate(model, testloader, device)
        hist.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'test_loss': te_loss, 'test_acc': te_acc})
        print(f"seed={seed} {optimizer_name} [{epoch}/{epochs}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    df = pd.DataFrame(hist)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb

    out = results_dir / f"NN_SimpleCIFAR10_{optimizer_name}_lr{lr}_seed{seed}.csv"
    df.to_csv(out, index=False)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR-10 Multi-Seed Runner')
    parser.add_argument('--seeds', type=str, default='1,2,3')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true')
    args, _ = parser.parse_known_args()

    seeds = [1, 2, 3] if args.quick else [int(s) for s in args.seeds.split(',') if s]
    epochs = 2 if args.quick else args.epochs
    batch_size = args.batch_size

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    opt_config = [('SGD', 0.1), ('SGD_Momentum', 0.1), ('Adam', 1e-3), ('AdamW', 1e-3), ('RMSProp', 1e-3)]

    completed = 0
    for opt, lr in opt_config:
        for seed in seeds:
            try:
                run_single(opt, seed, lr, epochs, batch_size, results_dir)
                completed += 1
            except Exception as e:
                print('Error:', e)

    print(f"Completed {completed} runs")


if __name__ == '__main__':
    main()
