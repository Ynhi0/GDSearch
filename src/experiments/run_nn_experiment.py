"""
Run neural network experiments on MNIST and CIFAR-10 with detailed logging.
"""
import os
import logging
from typing import Dict, Any, Tuple, Optional
import time
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.models import SimpleMLP, SimpleCNN, ConvNet
from src.core.data_utils import get_mnist_loaders, get_cifar10_loaders
from src.core.optimizer_wrappers import DelayedOptimizer
from src.core.training_utils import (
    set_seed, 
    validate_pytorch_version,
    get_loss_function,
    AMPWrapper,
    ModelEMA,
    create_amp_wrapper,
    create_model_ema
)
from src.utils.file_safety import safe_to_csv  # AUDIT FIX: Safe file I/O
from src.core.pytorch_optimizers import (
    SGDWrapper,
    SGDMomentumWrapper,
    AdamWrapper,
    AdamWWrapper,
    RMSPropWrapper,
    SAMWrapper,
    LookaheadWrapper,
    AdaBoundWrapper,
    RAdamWrapper,
    LAMBWrapper
)

# Import OOM-safe training function from modular src.core.oom_handler
from src.core.oom_handler import oom_safe_train_step
HAS_OOM_SAFE = True


# Removed duplicate set_seed - using from src.core.training_utils


def run_experiment(
    config: Dict[str, Any],
    device: str = 'cuda:0',
    results_dir: Path = Path('results')
) -> Dict[str, Any]:
    """
    Wrapper for running experiments in parallel execution mode.
    
    This function is called by ParallelExperimentRunner to execute
    experiments on specific GPU devices.
    
    Args:
        config: Experiment configuration dictionary
        device: PyTorch device string (e.g., 'cuda:0')
        results_dir: Directory to save results
        
    Returns:
        Dictionary with experiment metadata
    """
    # Ensure results directory exists
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Override device in config
    config = config.copy()
    config['device'] = device
    
    # Run the experiment
    try:
        df = train_and_evaluate(config)
        
        # Generate result filename
        from src.utils.result_filename import generate_result_filename
        result_filename = generate_result_filename(
            model=config['model'],
            dataset=config['dataset'],
            optimizer=config['optimizer'],
            lr=config['lr'],
            seed=config['seed'],
            tag=config.get('tag')
        )
        
        # Save results
        experiment_name = config.get('experiment_name', 'experiment')
        result_dir = results_dir / 'experiments' / experiment_name
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / result_filename
        
        df.to_csv(result_path, index=False)
        logging.info(f"Saved results to {result_path}")
        
        return {
            'status': 'success',
            'result_file': str(result_path),
            'final_test_acc': float(df['test_acc'].iloc[-1]) if 'test_acc' in df.columns else None,
            'final_train_loss': float(df['train_loss'].iloc[-1]) if 'train_loss' in df.columns else None
        }
        
    except Exception as e:
        logging.error(f"Experiment failed: {e}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e)
        }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flattened_grad_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        g = torch.cat(grads)
        
        # Check for non-finite gradients
        if not torch.isfinite(g).all():
            logging.warning("Non-finite gradients detected in norm calculation")
            return float('nan')
        
        return float(torch.norm(g, p=2).item())


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


def build_model_and_data(
    dataset: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
    seed: int,
    val_split: Optional[float] = None
) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    """Build model and data loaders with optional validation split.

    Args:
        dataset: Dataset name (MNIST/CIFAR-10)
        model_name: Model architecture name
        batch_size: Batch size for loaders
        device: Device to place model on
        seed: Random seed for reproducibility
        val_split: Optional validation split fraction (e.g., 0.1 for 10%)

    Returns:
        If val_split is None: (model, train_loader, test_loader)
        If val_split is provided: (model, train_loader, val_loader, test_loader)

    Note:
        Callers MUST handle both return patterns explicitly to avoid tuple unpacking errors.
        Recommended pattern:
            if val_split:
                model, train_loader, val_loader, test_loader = build_model_and_data(...)
            else:
                model, train_loader, test_loader = build_model_and_data(...)
    """
    if dataset.upper() == 'MNIST':
        loaders = get_mnist_loaders(batch_size=batch_size, seed=seed, val_split=val_split)
        if model_name == 'SimpleMLP':
            model = SimpleMLP()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for MNIST")
    elif dataset.upper() == 'CIFAR-10' or dataset.upper() == 'CIFAR10':
        loaders = get_cifar10_loaders(batch_size=batch_size, seed=seed, val_split=val_split)
        if model_name == 'SimpleCNN':
            model = SimpleCNN()
        elif model_name == 'ConvNet':
            model = ConvNet()
        else:
            raise ValueError(f"Unsupported model '{model_name}' for CIFAR-10")
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    model.to(device)

    if val_split is not None:
        train_loader, val_loader, test_loader = loaders
        return model, train_loader, val_loader, test_loader
    else:
        # Ensure we return a consistent 4-tuple signature: (model, train_loader, val_loader, test_loader)
        train_loader, test_loader = loaders
        val_loader = None
        return model, train_loader, val_loader, test_loader


def build_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
    """Build optimizer using CUSTOM implementations.

    Uses custom wrappers from pytorch_optimizers.py to test our implementations.
    """
    # Import constant at function level to avoid circular dependency
    from src.utils.constants import OptimizerNames
    
    name = optimizer_name.upper().replace('-', '_')  # Normalize names

    if name == OptimizerNames.SGD:
        return SGDWrapper(model.parameters(), lr=lr)
    elif name in ('SGD_MOMENTUM', 'SGDMOMENTUM', 'MOMENTUM'):
        return SGDMomentumWrapper(model.parameters(), lr=lr, momentum=momentum)
    elif name == OptimizerNames.ADAM:
        return AdamWrapper(model.parameters(), lr=lr)
    elif name in ('AMSGRAD', 'ADAM_AMSGRAD', 'ADAM_AMS'):
        # AMSGrad: Use AdamW with amsgrad=True for correct decoupled weight decay
        # Original Adam couples weight decay with adaptive learning rate (buggy)
        # AdamW implements correct decoupled weight decay (Loshchilov & Hutter 2019)
        if weight_decay > 0:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            return optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    elif name == OptimizerNames.ADAMW:
        return AdamWWrapper(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == OptimizerNames.RMSPROP:
        return RMSPropWrapper(model.parameters(), lr=lr)
    elif name == 'SAM':
        # SAM requires base optimizer
        if momentum > 0:
            base_opt = SGDMomentumWrapper(model.parameters(), lr=lr, momentum=momentum)
        else:
            base_opt = SGDWrapper(model.parameters(), lr=lr)
        return SAMWrapper(base_opt, rho=0.05)
    elif name in ('SGD_NESTEROV', 'SGDNESTEROV', 'NESTEROV'):
        # Expose SGD with Nesterov momentum to avoid confusion with the SGDNesterov class
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    elif name == 'LOOKAHEAD':
        # Lookahead requires base optimizer
        base_opt = AdamWrapper(model.parameters(), lr=lr)
        return LookaheadWrapper(base_opt, k=5, alpha=0.5)
    elif name == 'ADABOUND':
        return AdaBoundWrapper(model.parameters(), lr=lr)
    elif name == 'RADAM':
        return RAdamWrapper(model.parameters(), lr=lr)
    elif name == 'LAMB':
        return LAMBWrapper(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Available: SGD, SGD_MOMENTUM, SGD_NESTEROV, ADAM, ADAMW, RMSPROP, SAM, LOOKAHEAD, ADABOUND, RADAM, LAMB")


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
    if total_samples == 0:
        logging.warning("evaluate(): No samples processed, returning NaN")
        return float('nan'), float('nan')

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
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
      - val_split: float (optional) -> fraction of training data to use for validation (e.g., 0.1)
      - label_smoothing: float (optional) -> label smoothing factor (0.0 = off, 0.1 typical)
      - use_amp: bool (optional) -> enable automatic mixed precision training
      - use_ema: bool (optional) -> enable exponential moving average of model weights
      - ema_decay: float (optional) -> EMA decay rate (default: 0.9999)
    """
    seed = int(config.get('seed', 42))
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = config['model']
    dataset = config['dataset']
    batch_size = int(config.get('batch_size', 128))
    epochs = int(config.get('epochs', 5))
    val_split = config.get('val_split', None)

    # Build model and data loaders. build_model_and_data always returns a 4-tuple
    model, train_loader, val_loader, test_loader = build_model_and_data(
        dataset, model_name, batch_size, device, seed, val_split=val_split
    )

    # AUDIT FIX: Use configurable loss function to respect label_smoothing config
    label_smoothing = float(config.get('label_smoothing', 0.0))
    criterion = get_loss_function('cross_entropy', label_smoothing=label_smoothing)
    
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

    # AUDIT FIX: Setup AMP if enabled
    use_amp = bool(config.get('use_amp', False))
    amp = create_amp_wrapper(enabled=use_amp) if use_amp else None
    
    # AUDIT FIX: Setup EMA if enabled
    use_ema = bool(config.get('use_ema', False))
    ema_decay = float(config.get('ema_decay', 0.9999))
    ema = create_model_ema(model, decay=ema_decay) if use_ema else None

    history = []
    global_step = 0
    capture_epochs = set(config.get('capture_layer_grad_epochs', []))
    named_params = list(model.named_parameters())
    start_time = time.time()

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
    except Exception as e:
        logging.warning("Could not append meta row to history: %s", e, exc_info=True)
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
            if HAS_OOM_SAFE:
                # grad norm before step (capture before OOM-safe call modifies gradients)
                grad_norm = _flattened_grad_norm(model)

                # capture params before update
                params_before = _params_clone(model)

                try:
                    loss_value, actual_batch_size, _outputs, batch_tainted = oom_safe_train_step(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        inputs=inputs,
                        targets=targets,
                        device=device,
                        max_retries=3,
                        min_batch_size=1
                    )

                    # Track if any batch was tainted
                    if batch_tainted:
                        run_tainted = True
                        effective_batch_size = actual_batch_size

                    # AUDIT FIX: Update EMA if enabled
                    if ema is not None:
                        ema.update(model)

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
                # AUDIT FIX: Use AMP wrapper if enabled
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                # Forward pass with optional AMP
                if amp is not None:
                    with amp.autocast():
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                    amp.backward(loss, optimizer)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # grad norm after clipping (based on current grads)
                grad_norm = _flattened_grad_norm(model)

                # capture params before update to compute update_norm
                params_before = _params_clone(model)

                # step (possibly delayed optimizer)
                if amp is not None:
                    amp.step(optimizer)
                    amp.update()
                else:
                    optimizer.step()
                
                # AUDIT FIX: Update EMA if enabled
                if ema is not None:
                    ema.update(model)

                update_norm = _update_norm(model, params_before)
                loss_value = loss.item()

            global_step += 1

            elapsed = time.time() - start_time
            # AUDIT FIX: Read current LR from optimizer state, not static config (handles schedulers correctly)
            current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else float(config.get('lr', 1e-3))
            history.append({
                'phase': 'train',
                'epoch': epoch,
                'batch': batch_idx,
                'global_step': global_step,
                'train_loss': loss_value,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'lr': current_lr,
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
                            ln = float(torch.norm(p.grad.view(-1), p=2).item())
                        history.append({
                            'phase': 'layer_grad',
                            'epoch': epoch,
                            'global_step': global_step,
                            'layer': layer_name,
                            'layer_grad_norm': ln,
                        })

        if val_loader is not None:
            # AUDIT FIX: Evaluate with EMA model if enabled
            eval_model = ema.shadow if ema is not None else model
            val_loss, val_acc = evaluate(eval_model, val_loader, criterion, device)
            history.append({
                'phase': 'val',
                'epoch': epoch,
                'global_step': global_step,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'time_sec': time.time() - start_time,
                'tainted': run_tainted,
                'effective_batch_size': effective_batch_size,
                'original_batch_size': original_batch_size
            })

        # evaluation after each epoch (test set - only for final reporting)
        # AUDIT FIX: Evaluate with EMA model if enabled
        eval_model = ema.shadow if ema is not None else model
        test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device)
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
        meta_row = pd.DataFrame([{
            'phase': 'meta',
            'epoch': None,
            'global_step': converged_at_step,
            'converged': True,
            'time_sec': converged_at_time,
        }])
        df = pd.concat([df, meta_row], ignore_index=True)
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


def parse_experiments_from_config(cfg: dict):
    """Parse experiments from config data (supports multiple formats).

    Backwards-compatible: supports both an `optimizers` list per sweep and
    the older singular `optimizer` with `lr_values`/`weight_decay_values`.
    """
    exps = []
    for sweep in cfg.get('sweeps', []):
        model = sweep.get('model')
        dataset = sweep.get('dataset')

        if 'optimizers' in sweep and isinstance(sweep.get('optimizers'), list):
            for opt_config in sweep.get('optimizers', []):
                optimizer = opt_config.get('name')
                lr_list = opt_config.get('lr_values', opt_config.get('learning_rates', [])) or []
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
                    if 'momentum' in opt_config:
                        exp['momentum'] = opt_config['momentum']
                    if 'weight_decay' in opt_config:
                        exp['weight_decay'] = opt_config['weight_decay']
                    exps.append(exp)
        elif 'optimizer' in sweep:
            optimizer = sweep.get('optimizer')
            lr_list = sweep.get('lr_values', sweep.get('learning_rates', [])) or []
            weight_decays = sweep.get('weight_decay_values', sweep.get('weight_decay', [])) or []
            momentums = sweep.get('momentum_values', sweep.get('momentum', [])) or []

            if not lr_list and 'lr' in sweep:
                lr_list = [sweep.get('lr')]

            if not weight_decays:
                weight_decays = [None]
            if not momentums:
                momentums = [None]

            for lr in lr_list:
                for wd in weight_decays:
                    for mom in momentums:
                        exp = {
                            'model': model,
                            'dataset': dataset,
                            'optimizer': optimizer,
                            'lr': lr,
                            'epochs': sweep.get('epochs', 10),
                            'batch_size': sweep.get('batch_size', 128),
                            'seed': sweep.get('seed', 42)
                        }
                        if wd is not None:
                            exp['weight_decay'] = wd
                        if mom is not None:
                            exp['momentum'] = mom
                        exps.append(exp)
        else:
            raise ValueError(f"Sweep misconfigured, missing optimizer(s): {sweep}")

    return exps


def main():
    validate_pytorch_version(expected_version="2.6.0", strict=False)

    os.makedirs('results', exist_ok=True)

    # Load experiments from JSON config file
    config_path = 'configs/nn_tuning.json'
    if os.path.exists(config_path):
        logging.info("Loading experiments from %s", config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Parse config into experiment list (backwards-compatible)
        experiments = parse_experiments_from_config(config_data)

        # Fail fast if parsing produced no experiments — avoids silent no-op runs
        if not experiments:
            raise RuntimeError(
                "No experiments parsed from config file. Check 'sweeps' format in configs/nn_tuning.json or use the 'optimizers' list format."
            )
    else:
        # By default we fail-fast when config is missing to avoid silent, non-reproducible defaults.
        # To allow quick local runs using a built-in default sweep, explicitly set the
        # environment variable `GDSEARCH_ALLOW_DEFAULTS=1` (or 'true'). This must be
        # an explicit opt-in rather than the default behavior.
        if os.environ.get('GDSEARCH_ALLOW_DEFAULTS', '').lower() in ('1', 'true', 'yes'):
            logging.warning("Config file %s not found. Using built-in default experiments because GDSEARCH_ALLOW_DEFAULTS is set.", config_path)
            experiments = [
                # MNIST with MLP, Adam vs AdamW
                {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'Adam',  'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42},
                {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'AdamW', 'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42},
                # CIFAR-10 with CNN, SGD Momentum
                {'model': 'SimpleCNN', 'dataset': 'CIFAR-10', 'optimizer': 'SGD_Momentum', 'lr': 0.01, 'momentum': 0.9, 'epochs': 2, 'batch_size': 128, 'seed': 42},
                # Optional delayed optimization example
                {'model': 'SimpleMLP', 'dataset': 'MNIST', 'optimizer': 'Adam',  'lr': 1e-3, 'epochs': 2, 'batch_size': 128, 'seed': 42, 'use_delay_wrapper': True, 'delay_steps': 3},
            ]
        else:
            raise FileNotFoundError(
                f"Config file {config_path} not found. To proceed with default (non-reproducible) experiments, set GDSEARCH_ALLOW_DEFAULTS=1. Otherwise, create a valid config at {config_path}."
            )

    logging.info("Total experiments: %d", len(experiments))

    for cfg in tqdm(experiments, desc="NN Experiments"):
        df = train_and_evaluate(cfg)
        fname = result_filename(cfg)
        out_path = os.path.join('results', fname)
        # AUDIT FIX: Use safe_to_csv for automatic directory creation
        safe_to_csv(df, out_path, index=False)

    print("✅ Done. Results saved to 'results/'.")


if __name__ == '__main__':
    main()
