#!/usr/bin/env python3
"""
Self-contained MNIST publication experiments for Kaggle.
- Trains SimpleMLP on MNIST
- 5 optimizers × N seeds
- Saves per-run CSVs and statistical comparison CSV

This script is standalone (no external project imports) for easy Kaggle usage.
"""

import os
import math
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_data_loaders(batch_size: int, num_workers: int = 2, pin_memory: bool = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Kaggle note: Enable Internet in notebook settings to download MNIST.
    train_dataset = datasets.MNIST(root="/kaggle/working/data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="/kaggle/working/data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total


def _ckpt_path(ckpt_dir: Path, optimizer_name: str, seed: int, lr: float) -> Path:
    return ckpt_dir / f"MNIST_SimpleMLP_{optimizer_name}_lr{lr}_seed{seed}.pt"


def run_single_experiment(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path, *, resume: bool = False, ckpt_dir: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_loader, test_loader = get_data_loaders(batch_size)
    model = SimpleMLP().to(device)

    # Build optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Train
    history = []
    # Resume support
    if ckpt_dir is None:
        ckpt_dir = Path('checkpoints_mnist')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = _ckpt_path(ckpt_dir, optimizer_name, seed, lr)
    start_epoch = 1
    if resume and ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            if state.get('opt', '') == 'SGD' and isinstance(optimizer, optim.SGD):
                optimizer.load_state_dict(state['optimizer'])
            if state.get('opt', '') == 'Adam' and isinstance(optimizer, optim.Adam):
                optimizer.load_state_dict(state['optimizer'])
            if state.get('opt', '') == 'AdamW' and isinstance(optimizer, optim.AdamW):
                optimizer.load_state_dict(state['optimizer'])
            start_epoch = int(state.get('epoch', 0)) + 1
            history = state.get('history', [])
            tqdm.write(f"Resuming from epoch {start_epoch}: {ckpt_file}")
        except Exception as e:
            tqdm.write(f"Resume failed: {e}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        })
        tqdm.write(f"Seed {seed} | {optimizer_name} | Epoch {epoch}/{epochs} | "
                   f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, test_loss={test_loss:.4f}, test_acc={test_acc:.2%}")
        # Save checkpoint each epoch
        try:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'history': history,
                'opt': 'SGD' if isinstance(optimizer, optim.SGD) and optimizer.param_groups[0].get('momentum', 0.0) == 0.0 else (
                    'AdamW' if isinstance(optimizer, optim.AdamW) else ('Adam' if isinstance(optimizer, optim.Adam) else 'Other')
                ),
                'seed': seed,
                'lr': lr,
            }, ckpt_file)
        except Exception as e:
            tqdm.write(f"Warning: failed to save checkpoint: {e}")
    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    # Save per-run CSV
    df = pd.DataFrame(history)
    # Attach run-level telemetry (constant per-row for convenience)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    out_name = f"NN_SimpleMLP_MNIST_{optimizer_name}_lr{lr}_seed{seed}_publication.csv"
    out_path = results_dir / out_name
    df.to_csv(out_path, index=False)

    return df, elapsed


def run_suite(seeds, epochs, batch_size, results_dir: str, *, resume: bool = False, ckpt_dir: str | None = None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    optimizers = [
        ('SGD', 0.01),
        ('SGD_Momentum', 0.05),
        ('Adam', 0.001),
        ('AdamW', 0.001),
        ('AMSGrad', 0.001),
    ]

    total_runs = len(optimizers) * len(seeds)
    print(f"Total experiments to run: {total_runs}")

    completed = 0
    durations = []
    for opt_name, lr in optimizers:
        for seed in seeds:
            try:
                tqdm.write(f"\n=== Running: {opt_name} | seed={seed} | lr={lr} ===")
                _, dur = run_single_experiment(opt_name, seed, lr, epochs, batch_size, Path(results_dir), resume=resume, ckpt_dir=Path(ckpt_dir) if ckpt_dir else None)
                durations.append(dur)
                completed += 1
            except Exception as e:
                tqdm.write(f"❌ Error: {e}")

    print(f"\n✅ Completed {completed}/{total_runs} runs")
    if durations:
        avg_min = np.mean(durations) / 60.0
        print(f"Avg time per run: {avg_min:.2f} min")


def compute_statistics(results_dir: str):
    """Compute paired statistical comparisons and Holm-Bonferroni correction."""
    import glob

    # Collect files per optimizer
    patterns = {
        'SGD': f"{results_dir}/NN_SimpleMLP_MNIST_SGD_*_publication.csv",
        'SGD_Momentum': f"{results_dir}/NN_SimpleMLP_MNIST_SGD_Momentum_*_publication.csv",
        'Adam': f"{results_dir}/NN_SimpleMLP_MNIST_Adam_*_publication.csv",
        'AdamW': f"{results_dir}/NN_SimpleMLP_MNIST_AdamW_*_publication.csv",
        'AMSGrad': f"{results_dir}/NN_SimpleMLP_MNIST_AMSGrad_*_publication.csv",
    }

    # Extract final test_loss per seed
    def parse_seed(path):
        import re
        m = re.search(r"seed(\d+)", path)
        return int(m.group(1)) if m else None

    data = {}
    for opt, pattern in patterns.items():
        vals = {}
        for f in glob.glob(pattern):
            seed = parse_seed(f)
            if seed is None:
                continue
            df = pd.read_csv(f)
            final_row = df.iloc[-1]
            vals[seed] = final_row['test_loss']
        data[opt] = vals

    # Comparisons to perform
    comparisons = [
        ('Adam', 'SGD'),
        ('AdamW', 'Adam'),
        ('AMSGrad', 'Adam'),
        ('SGD_Momentum', 'SGD'),
        ('AdamW', 'SGD'),
        ('AMSGrad', 'SGD'),
        ('AMSGrad', 'AdamW'),
        ('SGD_Momentum', 'Adam'),
    ]

    rows = []
    for A, B in comparisons:
        seeds_A = set(data.get(A, {}).keys())
        seeds_B = set(data.get(B, {}).keys())
        common = sorted(list(seeds_A & seeds_B))
        if len(common) < 3:
            continue
        vals_A = np.array([data[A][s] for s in common])
        vals_B = np.array([data[B][s] for s in common])

        # Normality check
        _, pA = stats.shapiro(vals_A)
        _, pB = stats.shapiro(vals_B)
        if pA > 0.05 and pB > 0.05:
            stat_name = 'Paired t-test'
            t, p = stats.ttest_rel(vals_A, vals_B)
            d = (vals_A - vals_B).mean() / (vals_A - vals_B).std(ddof=1)
        else:
            stat_name = 'Wilcoxon'
            W, p = stats.wilcoxon(vals_A, vals_B)
            n = len(vals_A)
            d = 1 - (2 * W) / (n * (n + 1))  # rank-biserial correlation

        rows.append({
            'Optimizer A': A,
            'Optimizer B': B,
            'n_common_seeds': len(common),
            'Mean A': vals_A.mean(),
            'Std A': vals_A.std(ddof=1),
            'Mean B': vals_B.mean(),
            'Std B': vals_B.std(ddof=1),
            'Test': stat_name,
            'p-value': float(p),
            "Effect size (d or r)": float(d),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparisons could be computed (need >=3 common seeds per pair).")
        return None

    # Holm-Bonferroni correction
    m = len(df)
    order = np.argsort(df['p-value'].values)
    holm_sig = np.zeros(m, dtype=bool)
    alpha = 0.05
    for k, idx in enumerate(order):
        if df.loc[idx, 'p-value'] < alpha / (m - k):
            holm_sig[idx] = True
        else:
            break
    df['Significant (Holm-Bonferroni)'] = holm_sig

    out = Path(results_dir) / 'mnist_statistical_comparisons_publication.csv'
    df.to_csv(out, index=False)
    print(f"Saved statistical comparisons to: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description='MNIST Publication Experiments (Kaggle-ready)')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5,6,7,8,9,10', help='comma-separated seeds')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true', help='quick run: seeds=1..3, epochs=3')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoints if available')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_mnist')
    # Use parse_known_args to ignore Jupyter/Colab/Kaggle hidden args like '-f <kernel.json>'
    args, _unknown = parser.parse_known_args()

    if args.quick:
        seeds = [1, 2, 3]
        epochs = 3
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        epochs = args.epochs

    batch_size = args.batch_size
    results_dir = args.results_dir

    print("\n========================================")
    print(" MNIST Publication Experiments (Kaggle) ")
    print("========================================")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per run: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Results dir: {results_dir}")
    print("========================================\n")

    run_suite(seeds, epochs, batch_size, results_dir, resume=args.resume, ckpt_dir=args.ckpt_dir)
    compute_statistics(results_dir)


if __name__ == '__main__':
    main()
