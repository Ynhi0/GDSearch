"""
Run neural network experiments on MNIST and CIFAR-10 with detailed logging.
"""
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleMLP, SimpleCNN
from data_utils import get_mnist_loaders, get_cifar10_loaders
from optimizer_wrappers import DelayedOptimizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flattened_grad_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        g = torch.cat(grads)
        return torch.linalg.norm(g, ord=2).item()


def _params_clone(model: torch.nn.Module) -> Tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return tuple(p.detach().clone() for p in model.parameters() if p.requires_grad)


def _update_norm(model: torch.nn.Module, before: Tuple[torch.Tensor, ...]) -> float:
    with torch.no_grad():
        sq = 0.0
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            diff = (p.detach() - before[idx]).view(-1)
            sq += torch.dot(diff, diff).item()
            idx += 1
        return float(np.sqrt(sq))


def build_model_and_data(dataset: str, model_name: str, batch_size: int, device: torch.device):
    if dataset.upper() == 'MNIST':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        if model_name == 'SimpleMLP':
            model = SimpleMLP()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for MNIST")
    elif dataset.upper() == 'CIFAR-10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
        if model_name == 'SimpleCNN':
            model = SimpleCNN()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for CIFAR-10")
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    model.to(device)
    return model, train_loader, test_loader


def build_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
    name = optimizer_name.upper()
    if name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in ('SGD_MOMENTUM', 'SGD-MOMENTUM'):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'ADAM':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'ADAMW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def train_and_evaluate(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Train a model with specified config and return a DataFrame log.

    Expected config keys:
      - model: 'SimpleMLP' | 'SimpleCNN'
      - dataset: 'MNIST' | 'CIFAR-10'
      - optimizer: 'SGD' | 'SGD_Momentum' | 'Adam' | 'AdamW'
      - lr: float
      - epochs: int
      - batch_size: int
      - seed: int
      - momentum: float (for SGD_Momentum)
      - weight_decay: float (optional)
      - use_delay_wrapper: bool (optional)
      - delay_steps: int (if wrapper is used)
    """
    seed = int(config.get('seed', 42))
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = config['model']
    dataset = config['dataset']
    batch_size = int(config.get('batch_size', 128))
    epochs = int(config.get('epochs', 5))

    model, train_loader, test_loader = build_model_and_data(dataset, model_name, batch_size, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        optimizer_name=config['optimizer'],
        model=model,
        lr=float(config.get('lr', 1e-3)),
        weight_decay=float(config.get('weight_decay', 0.0)),
        momentum=float(config.get('momentum', 0.0)),
    )

    use_delay = bool(config.get('use_delay_wrapper', False))
    delay_steps = int(config.get('delay_steps', 1))
    if use_delay:
        optimizer = DelayedOptimizer(optimizer, delay_steps=delay_steps)

    history = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            # grad norm before step (based on current grads)
            grad_norm = _flattened_grad_norm(model)

            # capture params before update to compute update_norm
            params_before = _params_clone(model)

            # step (possibly delayed optimizer)
            optimizer.step()

            update_norm = _update_norm(model, params_before)

            global_step += 1

            history.append({
                'phase': 'train',
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step,
                'train_loss': loss.item(),
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'lr': float(config.get('lr', 1e-3)),
            })

        # evaluation after each epoch
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history.append({
            'phase': 'eval',
            'epoch': epoch,
            'global_step': global_step,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        })

    df = pd.DataFrame(history)
    return df


def result_filename(config: Dict[str, Any]) -> str:
    model = config['model']
    dataset = config['dataset']
    optimizer = config['optimizer']
    lr = config.get('lr', 0.0)
    seed = config.get('seed', 0)
    parts = ["NN", model, dataset, optimizer, f"lr{lr}", f"seed{seed}"]
    if config.get('use_delay_wrapper', False):
        parts.append(f"delay{config.get('delay_steps', 1)}")
    return "_".join(parts) + ".csv"


def main():
    os.makedirs('results', exist_ok=True)

    experiments = [
        # MNIST with MLP, Adam vs AdamW
        {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'Adam',  'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42},
        {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'AdamW', 'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42},
        # CIFAR-10 with CNN, SGD Momentum
        {'model': 'SimpleCNN', 'dataset': 'CIFAR-10', 'optimizer': 'SGD_Momentum', 'lr': 0.01, 'momentum': 0.9, 'epochs': 2, 'batch_size': 128, 'seed': 42},
        # Optional delayed optimization example
        {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'Adam',  'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42, 'use_delay_wrapper': True, 'delay_steps': 3},
    ]

    print(f"Total experiments: {len(experiments)}")

    for cfg in tqdm(experiments, desc="NN Experiments"):
        df = train_and_evaluate(cfg)
        fname = result_filename(cfg)
        out_path = os.path.join('results', fname)
        df.to_csv(out_path, index=False)

    print("Done. Results saved to 'results/'.")


if __name__ == '__main__':
    main()
