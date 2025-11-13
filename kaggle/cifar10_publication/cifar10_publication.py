#!/usr/bin/env python3
"""
Self-contained CIFAR-10 publication experiments for Kaggle.
- Trains a small ConvNet on CIFAR-10
- 5 optimizers × N seeds
- Saves per-run CSVs and statistical comparison CSV

This script is standalone (no external project imports) for easy Kaggle usage.
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_data_loaders(batch_size: int, num_workers: int = 2, pin_memory: bool = True):
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

    # Kaggle note: Enable Internet to download CIFAR-10.
    root = "/kaggle/working/data"
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
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
    total_loss, correct, total = 0.0, 0, 0
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


def run_single_experiment(optimizer_name: str, seed: int, lr: float, epochs: int, batch_size: int, results_dir: Path, *, resume: bool = False, ckpt_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_loader, test_loader = get_data_loaders(batch_size)
    model = SimpleCIFARNet().to(device)

    # Build optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Resume setup
    if ckpt_dir is None:
        ckpt_dir = Path('checkpoints_cifar10')
    else:
        ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"CIFAR10_SimpleCIFARNet_{optimizer_name}_lr{lr}_seed{seed}.pt"

    history = []
    start_epoch = 1
    if resume and ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'], strict=False)
            # Only resume optimizer if types match
            opt_tag = state.get('opt', '')
            try:
                if opt_tag == 'SGD' and isinstance(optimizer, optim.SGD) and optimizer.param_groups[0].get('momentum', 0.0) == 0.0:
                    optimizer.load_state_dict(state['optimizer'])
                elif opt_tag == 'SGD_Momentum' and isinstance(optimizer, optim.SGD) and optimizer.param_groups[0].get('momentum', 0.0) != 0.0:
                    optimizer.load_state_dict(state['optimizer'])
                elif opt_tag == 'RMSProp' and isinstance(optimizer, optim.RMSprop):
                    optimizer.load_state_dict(state['optimizer'])
                elif opt_tag == 'AdamW' and isinstance(optimizer, optim.AdamW):
                    optimizer.load_state_dict(state['optimizer'])
                elif opt_tag == 'AMSGrad' and isinstance(optimizer, optim.Adam) and optimizer.defaults.get('amsgrad', False):
                    optimizer.load_state_dict(state['optimizer'])
                elif opt_tag == 'Adam' and isinstance(optimizer, optim.Adam) and not optimizer.defaults.get('amsgrad', False):
                    optimizer.load_state_dict(state['optimizer'])
            except Exception as _:
                pass
            start_epoch = int(state.get('epoch', 0)) + 1
            history = state.get('history', [])
            print(f"Resuming from epoch {start_epoch}: {ckpt_file}")
        except Exception as e:
            print('Resume failed, starting fresh:', e)

    # Train
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
                    'SGD_Momentum' if isinstance(optimizer, optim.SGD) and optimizer.param_groups[0].get('momentum', 0.0) != 0.0 else (
                        'RMSProp' if isinstance(optimizer, optim.RMSprop) else (
                            'AdamW' if isinstance(optimizer, optim.AdamW) else (
                                'AMSGrad' if isinstance(optimizer, optim.Adam) and optimizer.defaults.get('amsgrad', False) else 'Adam'
                            )
                        )
                    )
                ),
                'seed': seed,
                'lr': lr,
            }, ckpt_file)
        except Exception as e:
            print('Warning: failed to save checkpoint:', e)

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    # Save per-run CSV
    df = pd.DataFrame(history)
    df['elapsed_seconds'] = elapsed
    df['peak_gpu_mb'] = peak_mb
    out_name = f"NN_SimpleCIFAR10_{optimizer_name}_lr{lr}_seed{seed}_publication.csv"
    out_path = results_dir / out_name
    df.to_csv(out_path, index=False)

    return df, elapsed


def run_suite(seeds, epochs, batch_size, results_dir: str, *, resume: bool = False, ckpt_dir: str | None = None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    optimizers = [
        ('SGD', 0.1),
        ('SGD_Momentum', 0.1),
        ('RMSProp', 0.001),
        ('Adam', 0.001),
        ('AdamW', 0.001),
        ('AMSGrad', 0.001),
    ]

    total_runs = len(optimizers) * len(seeds)
    print(f"Total experiments to run: {total_runs}")

    durations = []
    completed = 0
    for opt_name, lr in optimizers:
        for seed in seeds:
            try:
                tqdm.write(f"\n=== Running: {opt_name} | seed={seed} | lr={lr} ===")
                _, dur = run_single_experiment(opt_name, seed, lr, epochs, batch_size, Path(results_dir), resume=resume, ckpt_dir=ckpt_dir)
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
        'SGD': f"{results_dir}/NN_SimpleCIFAR10_SGD_*_publication.csv",
        'SGD_Momentum': f"{results_dir}/NN_SimpleCIFAR10_SGD_Momentum_*_publication.csv",
        'RMSProp': f"{results_dir}/NN_SimpleCIFAR10_RMSProp_*_publication.csv",
        'Adam': f"{results_dir}/NN_SimpleCIFAR10_Adam_*_publication.csv",
        'AdamW': f"{results_dir}/NN_SimpleCIFAR10_AdamW_*_publication.csv",
        'AMSGrad': f"{results_dir}/NN_SimpleCIFAR10_AMSGrad_*_publication.csv",
    }

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
            vals[seed] = final_row['test_acc']
        data[opt] = vals

    comparisons = [
        ('Adam', 'SGD'),
        ('AdamW', 'Adam'),
        ('AMSGrad', 'Adam'),
        ('SGD_Momentum', 'SGD'),
        ('RMSProp', 'SGD'),
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

    out = Path(results_dir) / 'cifar10_statistical_comparisons_publication.csv'
    df.to_csv(out, index=False)
    print(f"Saved statistical comparisons to: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Publication Experiments (Kaggle-ready)')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true', help='quick run: seeds=1..3, epochs=3')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoints if available')
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_cifar10')
    args, _unknown = parser.parse_known_args()

    if args.quick:
        seeds = [1, 2, 3]
        epochs = 3
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        epochs = args.epochs

    batch_size = args.batch_size
    results_dir = args.results_dir

    print("\n==========================================")
    print(" CIFAR-10 Publication Experiments (Kaggle) ")
    print("==========================================")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per run: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Results dir: {results_dir}")
    print("==========================================\n")

    run_suite(seeds, epochs, batch_size, results_dir, resume=args.resume, ckpt_dir=args.ckpt_dir)
    compute_statistics(results_dir)


if __name__ == '__main__':
    main()
