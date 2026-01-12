#!/usr/bin/env python3
"""
Self-contained Fashion-MNIST benchmark experiments for Kaggle.
- Trains SimpleMLP on Fashion-MNIST
- 5 optimizers × N seeds
- Saves per-run CSVs and statistical comparison CSV

This script is standalone (no external project imports) for easy Kaggle usage.
Fashion-MNIST is a more challenging drop-in replacement for MNIST with the same format.
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

# Import gradient monitoring utility
try:
    from gradient_monitoring import check_gradient_health
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gradient_monitoring import check_gradient_health


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMLP(nn.Module):
    """Simple MLP for Fashion-MNIST: 784 -> 128 -> 64 -> 10."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============ SAM Optimizers (Same as MNIST) ============

class SAMSGD(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) with SGD base."""
    def __init__(self, params, lr=0.1, momentum=0.9, rho=0.05):
        defaults = dict(lr=lr, momentum=momentum, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute gradient in adversarial direction (+ ε)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # Store old params
                self.state[p]['old_p'] = p.data.clone()
                # Perturb parameters
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original parameters and update with sharpness-aware gradient."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'old_p' not in self.state[p]:
                    continue
                # Restore
                p.data = self.state[p]['old_p']

        # Now do actual SGD update
        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(d_p, alpha=-group['lr'])
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """SAM requires closure for adversarial gradient computation."""
        assert closure is not None, "SAM requires closure for gradient computation."

        # First forward-backward pass
        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)

        # Second forward-backward pass
        with torch.enable_grad():
            loss = closure()
        self.second_step(zero_grad=True)

        return loss

    def _grad_norm(self):
        """Compute L2 norm of gradients."""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class SAMAdam(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) with Adam base."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, rho=0.05):
        defaults = dict(lr=lr, betas=betas, eps=eps, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute gradient in adversarial direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore and do Adam update."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'old_p' not in self.state[p]:
                    continue
                p.data = self.state[p]['old_p']

        # Adam update
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """SAM requires closure for gradient computation."""
        assert closure is not None, "SAM requires closure."

        with torch.enable_grad():
            loss = closure()
        self.first_step(zero_grad=True)

        with torch.enable_grad():
            loss = closure()
        self.second_step(zero_grad=True)

        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def get_data_loaders(batch_size: int, num_workers: int = 2, pin_memory: bool = True):
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
    ])

    # Kaggle note: Enable Internet in notebook settings to download Fashion-MNIST.
    train_dataset = datasets.FashionMNIST(root="/kaggle/working/data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="/kaggle/working/data", train=False, download=True, transform=transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        # Check if optimizer is SAM (requires closure)
        is_sam = isinstance(optimizer, (SAMSGD, SAMAdam))

        if is_sam:
            # SAM requires closure for adversarial gradient computation
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                return loss

            # SAM handles zero_grad internally in step()
            loss = optimizer.step(closure)

            # Re-compute output for accuracy (since SAM modifies parameters during step)
            with torch.no_grad():
                output = model(data)
        else:
            # Standard optimizers (SGD, Adam, etc.)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Check gradient health
            check_gradient_health(model, context="Fashion-MNIST")

            optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def test(model, loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def run_single_experiment(optimizer_name, lr, batch_size, epochs, seed, device, results_dir):
    """Run one experiment: train + test, save CSV."""
    set_seed(seed)
    train_loader, test_loader = get_data_loaders(batch_size)
    model = SimpleMLP().to(device)

    # Create optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGDMomentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'SAMSGD':
        optimizer = SAMSGD(model.parameters(), lr=lr, rho=0.05)
    elif optimizer_name == 'SAMAdam':
        optimizer = SAMAdam(model.parameters(), lr=lr, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Train
    epoch_data = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = test(model, test_loader, device)

        epoch_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })

    # Save per-run CSV
    df = pd.DataFrame(epoch_data)
    csv_name = f"FashionMNIST_{optimizer_name}_lr{lr}_seed{seed}.csv"
    csv_path = results_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"✓ {optimizer_name} (seed={seed}) final test_acc={test_acc:.2f}% -> {csv_name}")

    return test_acc


def run_all_optimizers(optimizers, batch_size, epochs, seeds, device, results_dir):
    """Run all optimizers × seeds, return summary DataFrame."""
    results = []

    for opt_name, lr in optimizers:
        for seed in seeds:
            test_acc = run_single_experiment(opt_name, lr, batch_size, epochs, seed, device, results_dir)
            results.append({
                'optimizer': opt_name,
                'lr': lr,
                'seed': seed,
                'final_test_accuracy': test_acc
            })

    df = pd.DataFrame(results)
    return df


def compute_statistics(df, results_dir):
    """Compute mean, std, and t-tests for optimizers."""
    summary = df.groupby('optimizer')['final_test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ['optimizer', 'mean_acc', 'std_acc', 'n_seeds']

    # Pairwise t-tests vs. best optimizer
    best_opt = summary.loc[summary['mean_acc'].idxmax(), 'optimizer']
    best_scores = df[df['optimizer'] == best_opt]['final_test_accuracy'].values

    p_values = []
    for opt in summary['optimizer']:
        if opt == best_opt:
            p_values.append(1.0)
        else:
            scores = df[df['optimizer'] == opt]['final_test_accuracy'].values
            _, p = stats.ttest_ind(best_scores, scores)
            p_values.append(p)

    summary['p_value_vs_best'] = p_values
    summary['significant'] = summary['p_value_vs_best'] < 0.05

    # Save comparison CSV
    comp_path = results_dir / "FashionMNIST_comparison.csv"
    summary.to_csv(comp_path, index=False)
    print(f"\n✓ Statistical comparison saved to {comp_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Benchmark")
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Seeds (comma-separated)')
    parser.add_argument('--results_dir', type=str, default='/kaggle/working/fashion_mnist_results', help='Results folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(',')]
    print(f"Seeds: {seeds}")

    # Optimizers with tuned LRs (same as MNIST for consistency)
    optimizers = [
        ('SGD', 0.05),
        ('SGDMomentum', 0.05),
        ('Adam', 0.001),
        ('AdamW', 0.001),
        ('RMSProp', 0.001),
        # ('SAMSGD', 0.05),  # Optional: uncomment to test SAM
        # ('SAMAdam', 0.001),
    ]

    print(f"\nRunning {len(optimizers)} optimizers × {len(seeds)} seeds = {len(optimizers)*len(seeds)} experiments")
    print("="*60)

    results_df = run_all_optimizers(optimizers, args.batch_size, args.epochs, seeds, device, results_dir)
    summary = compute_statistics(results_df, results_dir)

    print("\n" + "="*60)
    print("SUMMARY:")
    print(summary.to_string(index=False))
    print("="*60)


if __name__ == '__main__':
    main()
