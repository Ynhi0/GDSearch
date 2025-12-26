"""
Reproducibility helpers: verify claimed results using metadata and checkpoints.
"""
from pathlib import Path
import json
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Subset

from src.core.models import ResNet18
from src.core.data_utils import get_cifar10_loaders


def load_metadata(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def verify_checkpoint_with_metadata(meta_path: str, tolerance: float = 0.01, device: torch.device = None) -> Dict[str, Any]:
    """Verify a checkpoint against metadata.

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

    # Build model
    try:
        model = ResNet18(num_classes=10).to(device)
    except Exception as e:
        return {'status': 'error', 'details': f'Could not construct model: {e}'}

    # Load checkpoint
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
        model.load_state_dict(state)
    except Exception as e:
        return {'status': 'error', 'details': f'Failed to load checkpoint: {e}'}

    model.eval()

    # Load test data (use a reasonable sample to avoid long runtimes)
    try:
        _, _, test_loader = get_cifar10_loaders(batch_size=256, seed=meta.get('seed', 42), val_split=None)
    except Exception as e:
        return {'status': 'error', 'details': f'Failed to load CIFAR-10 test set: {e}'}

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