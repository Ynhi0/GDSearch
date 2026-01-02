"""
DataLoader performance optimization utilities for fair benchmarking.

Ensures DataLoader settings don't introduce confounding variables
in time-to-convergence comparisons. Provides platform-specific optimizations.
"""

import os
import platform
import logging
import torch
from typing import Dict, Any, Optional


def get_optimal_dataloader_kwargs(
    device: torch.device,
    benchmark_mode: bool = True,
    platform_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get optimal DataLoader kwargs for fair benchmarking.
    
    For time-to-convergence comparisons, DataLoader performance
    must be consistent across experiments. This function ensures:
    - Optimal num_workers for the platform
    - Proper pin_memory settings for GPU
    - Consistent persistent_workers behavior
    
    Args:
        device: torch.device (cuda/cpu)
        benchmark_mode: If True, optimize for throughput; else for reproducibility
        platform_name: Override platform detection ('kaggle', 'colab', 'windows', etc.)
        
    Returns:
        Dict with optimal DataLoader kwargs
    """
    kwargs = {}
    
    # Detect platform
    if platform_name is None:
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            platform_name = 'kaggle'
        elif 'COLAB_GPU' in os.environ:
            platform_name = 'colab'
        elif platform.system() == 'Windows':
            platform_name = 'windows'
        else:
            platform_name = 'linux'
    
    # Set num_workers based on platform and device
    if platform_name == 'windows':
        # Windows has issues with multiprocessing in some contexts
        kwargs['num_workers'] = 0
        kwargs['persistent_workers'] = False
    elif platform_name in ['kaggle', 'colab']:
        # Cloud platforms: use 2-4 workers for balance
        if device.type == 'cuda':
            kwargs['num_workers'] = 4  # GPU benefits from more workers
        else:
            kwargs['num_workers'] = 2
        kwargs['persistent_workers'] = True if kwargs['num_workers'] > 0 else False
    else:
        # Default Linux/Unix
        if device.type == 'cuda':
            # GPU: 4 workers is a good balance
            kwargs['num_workers'] = 4
        else:
            # CPU: fewer workers to avoid overhead
            kwargs['num_workers'] = 2
        kwargs['persistent_workers'] = True if kwargs['num_workers'] > 0 else False
    
    # pin_memory: Always True for CUDA, False for CPU
    if device.type == 'cuda':
        kwargs['pin_memory'] = True
    else:
        kwargs['pin_memory'] = False
    
    # prefetch_factor: for GPU with workers
    if device.type == 'cuda' and kwargs['num_workers'] > 0:
        kwargs['prefetch_factor'] = 2  # Default is 2, good for most cases
    
    # drop_last: Usually False for evaluation, can be True for training
    # (caller should override as needed)
    kwargs['drop_last'] = False
    
    if benchmark_mode:
        logging.info(
            f"DataLoader optimization for benchmarking: platform={platform_name}, "
            f"device={device.type}, num_workers={kwargs['num_workers']}, "
            f"pin_memory={kwargs['pin_memory']}, "
            f"persistent_workers={kwargs.get('persistent_workers', False)}"
        )
    
    return kwargs


def validate_dataloader_consistency(
    loader1: torch.utils.data.DataLoader,
    loader2: torch.utils.data.DataLoader
) -> bool:
    """
    Validate that two DataLoaders have consistent settings for fair comparison.
    
    When comparing optimizers, DataLoader settings must be identical
    to avoid confounding variables.
    
    Args:
        loader1: First DataLoader
        loader2: Second DataLoader
        
    Returns:
        True if settings match, False otherwise
    """
    critical_attrs = ['num_workers', 'pin_memory', 'batch_size', 'drop_last']
    
    mismatches = []
    for attr in critical_attrs:
        val1 = getattr(loader1, attr, None)
        val2 = getattr(loader2, attr, None)
        if val1 != val2:
            mismatches.append(f"{attr}: {val1} != {val2}")
    
    if mismatches:
        logging.warning(
            f"DataLoader settings mismatch (confounding variable risk): {', '.join(mismatches)}"
        )
        return False
    
    return True


def benchmark_dataloader_throughput(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int = 100
) -> float:
    """
    Measure DataLoader throughput (batches/second).
    
    Useful for validating that DataLoader settings are optimal.
    
    Args:
        loader: DataLoader to benchmark
        device: Device to transfer data to
        n_batches: Number of batches to measure
        
    Returns:
        Throughput in batches/second
    """
    import time
    
    start = time.perf_counter()
    count = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        count += 1
        if count >= n_batches:
            break
    
    elapsed = time.perf_counter() - start
    throughput = count / elapsed
    
    return throughput


def recommend_batch_size_for_fair_comparison(
    model: torch.nn.Module,
    device: torch.device,
    sample_input_shape: tuple,
    available_memory_gb: Optional[float] = None
) -> int:
    """
    Recommend batch size for fair optimizer comparison.
    
    Different batch sizes affect convergence speed and can confound
    optimizer comparisons. This function suggests a batch size that:
    - Fits in GPU memory
    - Is large enough to be efficient
    - Is consistent across experiments
    
    Args:
        model: Model to benchmark
        device: Device to use
        sample_input_shape: Shape of one input sample (without batch dim)
        available_memory_gb: GPU memory in GB (auto-detect if None)
        
    Returns:
        Recommended batch size
    """
    if device.type != 'cuda':
        # CPU: use moderate batch size
        return 128
    
    # Detect available memory
    if available_memory_gb is None:
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            available_memory_gb = 4.0  # Conservative default
    # Narrow the Optional type for static analysis
    assert available_memory_gb is not None
    available_memory_gb = float(available_memory_gb)
    
    # Rule of thumb: batch size based on model size and memory
    # Small models (< 10M params): 128-256
    # Medium models (10M-50M params): 64-128
    # Large models (> 50M params): 16-64
    
    n_params = sum(p.numel() for p in model.parameters())
    
    if n_params < 10_000_000:  # < 10M params
        if available_memory_gb >= 8:
            return 256
        else:
            return 128
    elif n_params < 50_000_000:  # 10M-50M params
        if available_memory_gb >= 16:
            return 128
        else:
            return 64
    else:  # > 50M params
        if available_memory_gb >= 16:
            return 64
        else:
            return 32


if __name__ == '__main__':
    # Test utilities
    print("Testing DataLoader optimization utilities...")
    
    # Test 1: Get optimal kwargs
    device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = get_optimal_dataloader_kwargs(device_gpu, benchmark_mode=True)
    print(f"Optimal kwargs for {device_gpu}: {kwargs}")
    
    # Test 2: CPU vs GPU settings should differ
    device_cpu = torch.device('cpu')
    kwargs_cpu = get_optimal_dataloader_kwargs(device_cpu, benchmark_mode=True)
    print(f"Optimal kwargs for CPU: {kwargs_cpu}")
    
    # Validate difference
    if device_gpu.type == 'cuda':
        assert kwargs['pin_memory'] == True, "GPU should have pin_memory=True"
        assert kwargs_cpu['pin_memory'] == False, "CPU should have pin_memory=False"
        print("✓ CPU/GPU settings correctly differ")
    
    # Test 3: Batch size recommendation
    simple_model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    batch_size = recommend_batch_size_for_fair_comparison(
        simple_model, device_gpu, (100,)
    )
    print(f"✓ Recommended batch size for simple model: {batch_size}")
    
    print("\nAll DataLoader optimization tests passed!")
