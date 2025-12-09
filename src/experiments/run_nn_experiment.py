"""
Run neural network experiments on MNIST and CIFAR-10 with detailed logging.

AUDIT FIX (Dec 2025): Added OOM-safe training with taint tracking to ensure
scientific validity of results. Tainted runs (with reduced batch sizes due to OOM)
are marked and should be excluded from statistical analysis.
"""
import os
from typing import Dict, Any, Tuple
import time
import json
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.models import SimpleMLP, SimpleCNN, ConvNet
from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders
from src.core.optimizer_wrappers import DelayedOptimizer
from src.core.training_utils import set_seed
from src.core.pytorch_optimizers import (
    SGDWrapper,
    SGDMomentumWrapper,
    AdamWrapper,
    AdamWWrapper
)

# AUDIT FIX: Import OOM-safe training function for taint tracking
try:
    # Try to import from run_all_kaggle (will work when run from repo root)
    import sys
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from run_all_kaggle import oom_safe_train_step
    HAS_OOM_SAFE = True
except ImportError:
    HAS_OOM_SAFE = False
    # Fallback: define a simple wrapper that doesn't handle OOM
    def oom_safe_train_step(model, optimizer, criterion, inputs, targets, device, opt_name="", max_retries=3, min_batch_size=1):
        """Fallback OOM-safe wrapper without actual OOM handling."""
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
        return loss.item(), inputs.size(0), outputs, False  # tainted=False


# Removed duplicate set_seed - using from src.core.training_utils


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


def build_model_and_data(dataset: str, model_name: str, batch_size: int, device: torch.device, seed: int):
    if dataset.upper() == 'MNIST':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, seed=seed)
        if model_name == 'SimpleMLP':
            model = SimpleMLP()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for MNIST")
    elif dataset.upper() == 'CIFAR-10' or dataset.upper() == 'CIFAR10':
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, seed=seed)
        if model_name == 'SimpleCNN':
            model = SimpleCNN()
        elif model_name == 'ConvNet':
            model = ConvNet()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for CIFAR-10")
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    model.to(device)
    return model, train_loader, test_loader


def build_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
    """Build optimizer using CUSTOM implementations.
    
    Uses custom wrappers from pytorch_optimizers.py to test our implementations.
    """
    name = optimizer_name.upper()
    if name == 'SGD':
        return SGDWrapper(model.parameters(), lr=lr)
    if name in ('SGD_MOMENTUM', 'SGD-MOMENTUM'):
        return SGDMomentumWrapper(model.parameters(), lr=lr, momentum=momentum)
    if name == 'ADAM':
        return AdamWrapper(model.parameters(), lr=lr)
    if name in ('AMSGRAD', 'ADAM_AMSGRAD', 'ADAM_AMS'):
        # Fallback to PyTorch for AMSGrad (not in custom impl)
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    if name == 'ADAMW':
        return AdamWWrapper(model.parameters(), lr=lr)
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
      - optimizer: 'SGD' | 'SGD_Momentum' | 'Adam' | 'AdamW' | 'AMSGrad' (or 'Adam_AMSGrad')
      - lr: float
      - epochs: int
      - batch_size: int
      - seed: int
      - momentum: float (for SGD_Momentum)
      - weight_decay: float (optional)
    - use_delay_wrapper: bool (optional)
    - delay_steps: int (if wrapper is used)
    - capture_layer_grad_epochs: List[int] (optional) -> capture per-layer grad norms on these epochs
    """
    seed = int(config.get('seed', 42))
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = config['model']
    dataset = config['dataset']
    batch_size = int(config.get('batch_size', 128))
    epochs = int(config.get('epochs', 5))

    model, train_loader, test_loader = build_model_and_data(dataset, model_name, batch_size, device, seed)
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
    capture_epochs = set(config.get('capture_layer_grad_epochs', []))
    named_params = list(model.named_parameters())
    start_time = time.time()
    
    # AUDIT FIX: Track tainted status for OOM recovery
    run_tainted = False
    original_batch_size = batch_size
    effective_batch_size = batch_size
    
    # Meta row with environment info
    try:
        history.append({
            'phase': 'meta_begin',
            'seed': seed,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(device),
            'cudnn_deterministic': getattr(torch.backends.cudnn, 'deterministic', None),
            'cudnn_benchmark': getattr(torch.backends.cudnn, 'benchmark', None),
            'time_sec': 0.0,
        })
    except Exception:
        pass
    # Convergence settings (optional)
    # Support both top-level keys (convergence_*), and nested config['convergence'] with keys
    conv_section = config.get('convergence', {}) if isinstance(config.get('convergence', {}), dict) else {}
    conv_grad_thr = float(config.get('convergence_grad_norm_threshold', conv_section.get('grad_norm_threshold', 0.0)))  # e.g., 1e-6
    conv_loss_delta_thr = float(config.get('convergence_loss_delta_threshold', conv_section.get('loss_delta_threshold', 0.0)))  # e.g., 1e-7
    conv_loss_window = int(config.get('convergence_loss_window', conv_section.get('loss_window', 0)))  # e.g., 100 train steps
    train_loss_window = []
    converged_at_step = None
    converged_at_time = None

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{epochs}")
        num_batches = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar, start=1):
            # AUDIT FIX: Use OOM-safe training step with taint tracking
            if HAS_OOM_SAFE:
                # Get optimizer name for SAM handling
                opt_name = config.get('optimizer', 'Unknown')
                
                # grad norm before step (capture before OOM-safe call modifies gradients)
                grad_norm = _flattened_grad_norm(model)
                
                # capture params before update
                params_before = _params_clone(model)
                
                try:
                    loss_value, actual_batch_size, outputs, batch_tainted = oom_safe_train_step(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        inputs=inputs,
                        targets=targets,
                        device=device,
                        opt_name=opt_name,
                        max_retries=3,
                        min_batch_size=1
                    )
                    
                    # Track if any batch was tainted
                    if batch_tainted:
                        run_tainted = True
                        effective_batch_size = actual_batch_size
                    
                    update_norm = _update_norm(model, params_before)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        # OOM that couldn't be recovered
                        run_tainted = True
                        effective_batch_size = 1
                        raise
                    else:
                        raise
            else:
                # Fallback: standard training without OOM handling
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
                loss_value = loss.item()

            global_step += 1

            elapsed = time.time() - start_time
            # AUDIT FIX: Add tainted tracking to history
            history.append({
                'phase': 'train',
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step,
                'train_loss': loss_value,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'lr': float(config.get('lr', 1e-3)),
                'time_sec': elapsed,
                'tainted': run_tainted,
                'effective_batch_size': effective_batch_size,
                'original_batch_size': original_batch_size
            })

            # Maintain loss window for convergence check
            if conv_loss_window > 0:
                train_loss_window.append(loss_value)
                if len(train_loss_window) > conv_loss_window:
                    train_loss_window.pop(0)

            # Convergence detection: grad_norm threshold OR loss improvement below threshold
            if converged_at_step is None:
                grad_ok = (conv_grad_thr > 0.0 and grad_norm < conv_grad_thr)
                loss_ok = False
                if conv_loss_window > 0 and conv_loss_delta_thr > 0.0 and len(train_loss_window) == conv_loss_window:
                    loss_ok = (abs(train_loss_window[0] - train_loss_window[-1]) < conv_loss_delta_thr)
                if grad_ok or loss_ok:
                    converged_at_step = global_step
                    converged_at_time = elapsed

            # Optionally capture per-layer grad norms at chosen epochs on last batch
            if epoch in capture_epochs and batch_idx == num_batches:
                with torch.no_grad():
                    for layer_name, p in named_params:
                        if p.grad is None:
                            ln = 0.0
                        else:
                            ln = torch.linalg.norm(p.grad.view(-1), ord=2).item()
                        history.append({
                            'phase': 'layer_grad',
                            'epoch': epoch,
                            'global_step': global_step,
                            'layer': layer_name,
                            'layer_grad_norm': ln,
                        })

        # evaluation after each epoch
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        # AUDIT FIX: Include tainted status in eval phase
        history.append({
            'phase': 'eval',
            'epoch': epoch,
            'global_step': global_step,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'time_sec': time.time() - start_time,
            'tainted': run_tainted,
            'effective_batch_size': effective_batch_size,
            'original_batch_size': original_batch_size
        })

    df = pd.DataFrame(history)
    # If convergence occurred, annotate once at the end as metadata rows
    if converged_at_step is not None:
        df.loc[len(df)] = {
            'phase': 'meta',
            'epoch': None,
            'global_step': converged_at_step,
            'converged': True,
            'time_sec': converged_at_time,
        }
    return df


def result_filename(config: Dict[str, Any]) -> str:
    """Generate unique result filename with UUID to prevent race conditions."""
    model = config['model']
    dataset = config['dataset']
    optimizer = config['optimizer']
    lr = config.get('lr', 0.0)
    seed = config.get('seed', 0)
    parts = ["NN", model, dataset, optimizer, f"lr{lr}", f"seed{seed}"]
    if config.get('use_delay_wrapper', False):
        parts.append(f"delay{config.get('delay_steps', 1)}")
    if 'momentum' in config and config.get('momentum', 0.0) != 0.0 and (optimizer.upper().startswith('SGD')):
        parts.append(f"mom{config.get('momentum')}")
    if 'weight_decay' in config and float(config.get('weight_decay', 0.0)) != 0.0:
        parts.append(f"wd{config.get('weight_decay')}")
    if 'tag' in config:
        parts.append(str(config['tag']))
    # Add UUID to prevent race conditions
    run_id = str(uuid.uuid4())[:8]
    parts.append(run_id)
    return "_".join(parts) + ".csv"


def main():
    os.makedirs('results', exist_ok=True)

    # Load experiments from JSON config file
    config_path = 'configs/nn_tuning.json'
    if os.path.exists(config_path):
        print(f"Loading experiments from {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Parse config into experiment list
        experiments = []
        for sweep in config_data.get('sweeps', []):
            model = sweep.get('model')
            dataset = sweep.get('dataset')
            for opt_config in sweep.get('optimizers', []):
                optimizer = opt_config.get('name')
                # AUDIT FIX: Support both lr_values and learning_rates for backward compatibility
                lr_list = opt_config.get('lr_values', opt_config.get('learning_rates', []))
                for lr in lr_list:
                    exp = {
                        'model': model,
                        'dataset': dataset,
                        'optimizer': optimizer,
                        'lr': lr,
                        'epochs': sweep.get('epochs', 10),
                        'batch_size': sweep.get('batch_size', 128),
                        'seed': sweep.get('seed', 42)
                    }
                    # Add optional parameters
                    if 'momentum' in opt_config:
                        exp['momentum'] = opt_config['momentum']
                    if 'weight_decay' in opt_config:
                        exp['weight_decay'] = opt_config['weight_decay']
                    experiments.append(exp)
    else:
        # Fallback to hardcoded experiments if config doesn't exist
        print(f"Config file {config_path} not found. Using defaults.")
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
