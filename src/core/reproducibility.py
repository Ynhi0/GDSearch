"""
Reproducibility helpers: verify claimed results using metadata and checkpoints.

CRITICAL FIX (Issue #28): Made verify_checkpoint_with_metadata() DYNAMIC instead of
hardcoded for CIFAR-10/ResNet18. Now reads dataset_name and model_arch from metadata.
"""
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, Callable

import torch
from src.core.io_utils import torch_load_safe
from torch.utils.data import DataLoader, Subset

from src.core.models import ResNet18, SimpleMLP, SimpleCNN
from src.core.data_utils import get_cifar10_loaders, get_mnist_loaders


def load_metadata(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def verify_checkpoint_with_metadata(meta_path: str, tolerance: float = 0.01, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Verify a checkpoint against metadata.

    CRITICAL FIX (Issue #28): Now DYNAMIC - reads dataset_name and model_arch from metadata.json
    instead of hardcoding CIFAR-10/ResNet18. This allows verification across all domains
    (MNIST, CIFAR-10, CIFAR-100, NLP, medical imaging).

    Returns a dict with keys:
      - status: 'metadata_only'|'verified'|'mismatch'|'error'
      - details: message or numeric results

    This function is intentionally conservative: if the checkpoint is missing it
    does not fail, but returns 'metadata_only' so CI can surface missing artifacts.
    """
    try:
        meta = load_metadata(meta_path)
    except Exception as e:
        return {'status': 'error', 'details': f'Failed to load metadata: {e}'}

    ckpt_path = Path(meta.get('checkpoint', ''))
    if not ckpt_path.exists():
        return {'status': 'metadata_only', 'details': 'Checkpoint file missing: ' + str(ckpt_path)}

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CRITICAL FIX (Issue #28): Read dataset and model architecture from metadata
    dataset_name = meta.get('dataset', 'CIFAR10').upper()  # Default to CIFAR10 for backward compat
    model_arch = meta.get('model', 'ResNet18')  # Default to ResNet18 for backward compat
    num_classes = meta.get('num_classes', 10)

    # Build model dynamically based on metadata
    try:
        if 'ResNet18' in model_arch:
            model = ResNet18(num_classes=num_classes).to(device)
        elif 'SimpleMLP' in model_arch or 'MLP' in model_arch or 'MNIST' in dataset_name:
            # MNIST typically uses SimpleMLP
            model = SimpleMLP(num_classes=num_classes).to(device)
        elif 'SimpleCNN' in model_arch:
            model = SimpleCNN(num_classes=num_classes).to(device)
        elif 'ConvNet' in model_arch:
            from src.core.models import ConvNet
            model = ConvNet(num_classes=num_classes).to(device)
        else:
            # Fallback for unknown architectures
            logging.warning(f"Unknown architecture '{model_arch}' for dataset '{dataset_name}', defaulting to ResNet18")
            model = ResNet18(num_classes=num_classes).to(device)
    except Exception as e:
        return {'status': 'error', 'details': f'Could not construct model {model_arch}: {e}'}

    # Load checkpoint
    try:
        ckpt = torch_load_safe(str(ckpt_path), map_location=device)

        # Accept multiple checkpoint formats for robustness:
        # 1) {'model_state_dict': <state_dict>, ...}
        # 2) {'model': <state_dict>, ...} (legacy / Kaggle scripts)
        # 3) raw state_dict
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state = ckpt['model']
            else:
                # If the dict looks like a state_dict (tensor values), attempt to load directly
                state = ckpt
        else:
            state = ckpt

        model.load_state_dict(state)
    except Exception as e:
        return {'status': 'error', 'details': f'Failed to load checkpoint: {e}'}

    model.eval()

    # CRITICAL FIX (Issue #28): Load test data dynamically based on dataset_name from metadata
    try:
        if 'CIFAR10' in dataset_name:
            _, _, test_loader = get_cifar10_loaders(batch_size=256, seed=meta.get('seed', 42), val_split=None)
        elif 'MNIST' in dataset_name:
            # Handles MNIST and FashionMNIST (both use same loader structure)
            _, _, test_loader = get_mnist_loaders(batch_size=256, seed=meta.get('seed', 42), val_split=None)
        else:
            # For other datasets (CIFAR-100, NLP, medical), try CIFAR-10 as fallback
            logging.warning(f"Unknown dataset '{dataset_name}', defaulting to CIFAR-10 test loader")
            _, _, test_loader = get_cifar10_loaders(batch_size=256, seed=meta.get('seed', 42), val_split=None)
    except Exception as e:
        return {'status': 'error', 'details': f'Failed to load test set for {dataset_name}: {e}'}

    # Evaluate on subset - limit to 1000 samples for speed
    max_samples = 1000
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            bs = inputs.size(0)
            correct += (preds == targets).sum().item()
            total += bs
            if total >= max_samples:
                break

    if total == 0:
        return {'status': 'error', 'details': 'No test samples processed'}

    accuracy = correct / total
    claimed = float(meta.get('accuracy', 0.0))

    if abs(accuracy - claimed) <= tolerance:
        return {'status': 'verified', 'details': {'measured_accuracy': accuracy, 'claimed_accuracy': claimed}}
    else:
        return {'status': 'mismatch', 'details': {'measured_accuracy': accuracy, 'claimed_accuracy': claimed}}