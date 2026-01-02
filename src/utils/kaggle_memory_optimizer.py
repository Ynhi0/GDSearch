"""
Memory Optimization for Kaggle T4 x2 GPUs

Kaggle provides 2x NVIDIA Tesla T4 GPUs with 15GB VRAM each.
This module provides utilities to optimize memory usage for this environment.

Key Constraints:
- Total VRAM: 30GB (2x15GB)
- Need to fit: Model + Optimizer State + Gradients + Batch Data + Activations
- ResNet18 on CIFAR-10: ~11M params = ~44MB (fp32)
- Adam state: 2x params = ~88MB
- Batch size tuning is critical

Best Practices:
1. Use mixed precision (FP16) training via torch.amp
2. Enable gradient checkpointing for large models
3. Clear cache between experiments
4. Use gradient accumulation for effective large batches
5. Monitor memory usage and adapt dynamically
"""

import torch
import logging
import gc
from typing import Dict, Tuple, Optional, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KaggleT4Config:
    """Configuration for Kaggle T4 x2 environment."""
    total_vram_gb: float = 30.0  # 2x15GB
    max_batch_size_resnet18_cifar10: int = 512  # Conservative default
    max_batch_size_bert_base: int = 32  # For NLP tasks
    enable_mixed_precision: bool = True  # Use FP16 by default
    gradient_accumulation_steps: int = 1  # For simulating larger batches
    

def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dict with allocated_gb, reserved_gb, free_gb
    """
    if not torch.cuda.is_available():
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'free_gb': 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    
    # Get total memory for device 0
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - allocated
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'free_gb': free,
        'total_gb': total
    }


def estimate_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    max_vram_gb: float = 14.0,  # Leave 1GB margin on 15GB GPU
    safety_factor: float = 0.7  # Conservative multiplier
) -> int:
    """
    Estimate maximum safe batch size for a model.
    
    For Kaggle: Prevents OOM crashes by estimating memory requirements
    before running experiments.
    
    Args:
        model: PyTorch model
        input_shape: Shape of one input sample (C, H, W) or (seq_len,)
        max_vram_gb: Maximum VRAM to use (GB)
        safety_factor: Multiply result by this for safety margin
        
    Returns:
        Estimated max batch size
    """
    device = next(model.parameters()).device
    
    # Count model parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_memory_gb = param_count * 4 / (1024 ** 3)  # 4 bytes per float32
    
    # Estimate optimizer state (Adam needs 2x params)
    optimizer_memory_gb = param_memory_gb * 2
    
    # Estimate gradients memory
    gradient_memory_gb = param_memory_gb
    
    # Estimate activation memory per sample (very rough heuristic)
    # For CNNs: roughly 4x input size through multiple layers
    sample_elements = 1
    for dim in input_shape:
        sample_elements *= dim
    activation_per_sample_gb = sample_elements * 4 * 4 / (1024 ** 3)  # 4x multiplier
    
    # Available memory for batch data and activations
    available_gb = max_vram_gb - param_memory_gb - optimizer_memory_gb - gradient_memory_gb
    
    if available_gb <= 0:
        logger.warning("Model + optimizer exceeds available VRAM!")
        return 1
    
    # Estimate batch size
    estimated_batch = int((available_gb / activation_per_sample_gb) * safety_factor)
    
    # Clamp to reasonable range
    estimated_batch = max(1, min(estimated_batch, 1024))
    
    logger.info(f"Memory estimate: Model={param_memory_gb:.2f}GB, "
                f"Optimizer={optimizer_memory_gb:.2f}GB, "
                f"Available={available_gb:.2f}GB, "
                f"Estimated batch size={estimated_batch}")
    
    return estimated_batch


def clear_memory_cache():
    """
    Aggressively clear GPU memory cache.
    
    Use between experiments to prevent memory fragmentation.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    logger.debug("Cleared GPU memory cache and ran garbage collection")


def setup_mixed_precision_training() -> Tuple[torch.cuda.amp.GradScaler, Type[torch.cuda.amp.autocast]]:
    """
    Setup mixed precision training for Kaggle T4.
    
    T4 has Tensor Cores that accelerate FP16 operations, giving ~2x speedup
    and ~50% memory reduction.
    
    Returns:
        (scaler, autocast_context)
    """
    if not torch.cuda.is_available():
        logger.warning("Mixed precision requested but CUDA not available")
        return None, None  # type: ignore[return-value]
    
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast  # type: ignore[return-value]
    
    logger.info("Mixed precision training enabled (FP16 + Tensor Cores)")
    
    return scaler, autocast


def optimize_dataloader_for_kaggle(
    dataset,
    batch_size: int,
    num_workers: Optional[int] = None,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create optimized DataLoader for Kaggle environment.
    
    Kaggle notebooks have 2-4 CPU cores, so tuning num_workers is important.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of worker processes (auto-detect if None)
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Optimized DataLoader
    """
    import os
    
    if num_workers is None:
        # Kaggle: 2 workers is usually optimal (avoid CPU bottleneck)
        num_workers = min(2, os.cpu_count() or 2)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,  # Reuse workers
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    logger.info(f"Created DataLoader: batch_size={batch_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    return dataloader


def gradient_accumulation_wrapper(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    dataloader,
    accumulation_steps: int = 4,
    device: str = 'cuda',
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Training loop with gradient accumulation.
    
    Simulates larger batch sizes without OOM by accumulating gradients
    across multiple forward/backward passes.
    
    Effective batch size = batch_size * accumulation_steps
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        loss_fn: Loss function
        dataloader: DataLoader
        accumulation_steps: Number of steps to accumulate
        device: Device ('cuda' or 'cpu')
        use_amp: Use mixed precision
        
    Returns:
        Dict with epoch_loss, num_batches
    """
    model.train()
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass (with mixed precision if enabled)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        num_batches += 1
    
    # Handle remaining gradients if not evenly divisible
    if num_batches % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return {
        'epoch_loss': total_loss / num_batches,
        'num_batches': num_batches,
        'effective_batch_size': dataloader.batch_size * accumulation_steps
    }


def monitor_memory_usage(prefix: str = ""):
    """
    Log current GPU memory usage.
    
    Use throughout training to detect memory leaks.
    """
    if not torch.cuda.is_available():
        return
    
    mem_info = get_gpu_memory_info()
    logger.info(f"{prefix}GPU Memory: Allocated={mem_info['allocated_gb']:.2f}GB, "
                f"Reserved={mem_info['reserved_gb']:.2f}GB, "
                f"Free={mem_info['free_gb']:.2f}GB")


def suggest_kaggle_config(experiment_type: str) -> Dict:
    """
    Suggest optimal configuration for Kaggle T4 x2.
    
    Args:
        experiment_type: 'resnet_cifar10', 'bert_nlp', 'medical_segmentation'
        
    Returns:
        Dict with suggested config
    """
    configs = {
        'resnet_cifar10': {
            'batch_size': 256,
            'gradient_accumulation_steps': 2,  # Effective BS = 512
            'num_workers': 2,
            'mixed_precision': True,
            'pin_memory': True,
            'estimated_time_per_epoch_sec': 30
        },
        'bert_nlp': {
            'batch_size': 16,
            'gradient_accumulation_steps': 4,  # Effective BS = 64
            'num_workers': 2,
            'mixed_precision': True,
            'pin_memory': True,
            'estimated_time_per_epoch_sec': 300
        },
        'medical_segmentation': {
            'batch_size': 2,  # 3D volumes are huge
            'gradient_accumulation_steps': 8,  # Effective BS = 16
            'num_workers': 2,
            'mixed_precision': True,
            'pin_memory': True,
            'estimated_time_per_epoch_sec': 600
        }
    }
    
    config = configs.get(experiment_type, configs['resnet_cifar10'])
    logger.info(f"Suggested config for {experiment_type}: {config}")
    
    return config


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check memory
    mem_info = get_gpu_memory_info()
    print(f"GPU Memory: {mem_info}")
    
    # Suggest config
    config = suggest_kaggle_config('resnet_cifar10')
    print(f"Suggested config: {config}")
