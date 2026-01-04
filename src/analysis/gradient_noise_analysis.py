"""
Gradient Noise Variance Estimation for SGD Theoretical Bounds.

Addresses the methodological flaw of using arbitrary σ² values
in theoretical convergence bounds. This module provides rigorous estimation of
gradient noise variance from empirical mini-batch gradients.

The gradient noise variance σ² is defined as:
    σ² = E[||∇f_i(x) - ∇f(x)||²]

where ∇f_i is the stochastic gradient from sample i and ∇f is the true gradient.

This is essential for validating theoretical SGD convergence bounds against
empirical results.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Any, Tuple
import logging


def estimate_gradient_noise_variance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_samples: int = 100,
    method: str = 'empirical_variance'
) -> Dict[str, Union[float, str, Dict[str, float]]]:
    """
    Estimate gradient noise variance σ² for stochastic optimization.
    
    This function computes the variance of mini-batch gradients to quantify
    the stochasticity in SGD. This is REQUIRED for making theoretical
    convergence bounds meaningful (not arbitrary).
    
    Methods:
    1. 'empirical_variance': Var[∇f_i] = E[||∇f_i - E[∇f_i]||²]
    2. 'full_batch_comparison': E[||∇f_batch - ∇f_full||²] (if full batch available)
    
    Args:
        model: Neural network model
        data_loader: DataLoader for computing gradients
        criterion: Loss function
        device: Device for computation
        num_samples: Number of gradient samples to collect
        method: Estimation method ('empirical_variance' or 'full_batch_comparison')
        
    Returns:
        Dict containing:
         - sigma_squared: Estimated gradient noise variance
          - sigma: Standard deviation (sqrt of variance)
          - num_samples_used: Actual number of samples used
          - method: Method used for estimation
          - per_param_variance: Variance broken down by parameter group
    """
    model.eval()  # Disable dropout/batchnorm randomness
    
    gradient_samples = []
    param_names = [name for name, _ in model.named_parameters() if _.requires_grad]
    
    # Collect gradient samples from mini-batches
    samples_collected = 0
    for batch_idx, batch in enumerate(data_loader):
        if samples_collected >= num_samples:
            break
        
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
        
        # Compute gradient for this mini-batch
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Extract gradient as flat vector
        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.detach().cpu().flatten())
        
        if len(grad_vector) > 0:
            grad_flat = torch.cat(grad_vector).numpy()
            gradient_samples.append(grad_flat)
            samples_collected += 1
    
    if len(gradient_samples) < 2:
        logging.warning("Insufficient gradient samples for variance estimation")
        return {  # type: ignore[return-value]
            'sigma_squared': 0.0,
            'sigma': 0.0,
            'num_samples_used': float(len(gradient_samples)),
            'method': str(method),
            'per_param_variance': {}  # type: ignore
        }
    
    # Convert to numpy array for computation
    gradient_array = np.array(gradient_samples)
    
    # Compute variance across samples
    if method == 'empirical_variance':
        # CORRECT FORMULA: σ² = E[||g||²] - ||E[g]||²
        # This is the variance of the gradient norm, which measures stochasticity
        # 
        # Mathematical derivation:
        # σ² = E[(g - E[g])ᵀ(g - E[g])]  [definition of variance]
        #    = E[gᵀg - 2gᵀE[g] + E[g]ᵀE[g]]  [expand]
        #    = E[gᵀg] - 2E[g]ᵀE[g] + E[g]ᵀE[g]  [linearity of expectation]
        #    = E[||g||²] - ||E[g]||²  [simplify]
        mean_gradient = np.mean(gradient_array, axis=0)
        
        # E[||g||²]: mean of squared norms
        squared_norms = np.sum(gradient_array ** 2, axis=1)
        mean_squared_norm = np.mean(squared_norms)
        
        # ||E[g]||²: squared norm of mean
        mean_norm_squared = np.sum(mean_gradient ** 2)
        
        # σ² = E[||g||²] - ||E[g]||²
        sigma_squared = mean_squared_norm - mean_norm_squared
        
        # Ensure non-negative (numerical errors can make this slightly negative)
        sigma_squared = max(0.0, sigma_squared)
        
    elif method == 'full_batch_comparison':
        # Compute full-batch gradient (expensive but accurate)
        full_batch_grad = compute_full_batch_gradient(
            model, data_loader, criterion, device
        )
        
        if full_batch_grad is None:
            # Fall back to empirical variance
            logging.warning("Could not compute full-batch gradient, falling back to empirical variance")
            mean_gradient = np.mean(gradient_array, axis=0)
            
            # CONSISTENT: Use same formula as empirical_variance method
            squared_norms = np.sum(gradient_array ** 2, axis=1)
            mean_squared_norm = np.mean(squared_norms)
            mean_norm_squared = np.sum(mean_gradient ** 2)
            sigma_squared = max(0.0, mean_squared_norm - mean_norm_squared)
        else:
            # E[||g_batch - g_full||²]
            deviations = gradient_array - full_batch_grad[np.newaxis, :]
            variances = np.sum(deviations ** 2, axis=1)
            sigma_squared = np.mean(variances)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    sigma = np.sqrt(sigma_squared)
    
    # GAP 17 FIX: Test for heavy-tailed gradients (Shapiro-Wilk normality test)
    # If gradients are NOT Gaussian (heavy-tailed/Levy), standard SGD theory is invalid
    # This is critical for scientific rigor: you can't use Gaussian-based bounds on Levy noise
    normality_pvalue = None
    is_gaussian = None
    
    try:
        from scipy.stats import shapiro
        
        # Test normality on a sample of gradient components (testing all is too slow)
        # Sample 5000 random gradient components across all samples
        n_components_test = min(5000, gradient_array.size)
        flat_grads = gradient_array.flatten()
        sample_indices = np.random.choice(len(flat_grads), n_components_test, replace=False)
        sample_grads = flat_grads[sample_indices]
        
        # Shapiro-Wilk test: H0 = data is Gaussian
        stat, p_value = shapiro(sample_grads)
        normality_pvalue = float(p_value)
        is_gaussian = bool(p_value >= 0.05)  # Fail to reject H0 at α=0.05
        
        if is_gaussian:
            logging.info(f"Gradient noise is approximately Gaussian (p={p_value:.4f} ≥ 0.05)")
        else:
            logging.warning(f"HEAVY-TAILED GRADIENTS DETECTED: p={p_value:.4f} < 0.05. "
                          f"Standard SGD theory assumes Gaussian noise → bounds may be invalid!")
    
    except ImportError:
        logging.warning("scipy not available, skipping normality test")
    except Exception as e:
        logging.warning(f"Normality test failed: {e}")
    
    # Compute per-parameter variance breakdown
    per_param_var = {}
    param_idx = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_size = param.numel()
        param_grads = gradient_array[:, param_idx:param_idx + param_size]
        param_var = np.var(param_grads)
        per_param_var[name] = float(param_var)
        param_idx += param_size
    
    return {  # type: ignore[return-value]
        'sigma_squared': float(sigma_squared),
        'sigma': float(sigma),
        'num_samples_used': float(len(gradient_samples)),
        'method': str(method),
        'per_param_variance': per_param_var,  # type: ignore
        # GAP 17: Heavy-tail detection
        'normality_test_pvalue': float(normality_pvalue) if normality_pvalue is not None else None,
        'is_gaussian': bool(is_gaussian) if is_gaussian is not None else None,
        'heavy_tailed_warning': not is_gaussian if is_gaussian is not None else False
    }


def compute_full_batch_gradient(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Compute the full-batch gradient (true gradient over entire dataset).
    
    WARNING: This is expensive for large datasets. Use sparingly.
    
    Args:
        model: Neural network
        data_loader: DataLoader (should cover entire dataset)
        criterion: Loss function
        device: Computation device
        max_samples: Maximum number of samples to use (None = all)
        
    Returns:
        Flattened gradient vector or None if computation fails
    """
    model.eval()
    model.zero_grad()
    
    total_loss = 0.0
    total_samples = 0
    
    try:
        for batch_idx, batch in enumerate(data_loader):
            if max_samples is not None and total_samples >= max_samples:
                break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Accumulate gradients (weighted by batch size)
            batch_size = inputs.size(0)
            (loss * batch_size).backward()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Normalize gradients by total samples
        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= total_samples
                if param.grad is not None:  # Recheck after division
                    grad_vector.append(param.grad.detach().cpu().flatten())
        
        if len(grad_vector) == 0:
            return None
        
        return torch.cat(grad_vector).numpy()
        
    except Exception as e:
        logging.warning(f"Failed to compute full-batch gradient: {e}")
        return None


def track_gradient_noise_over_training(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    check_every_n_epochs: int = 5,
    num_samples_per_check: int = 50
) -> List[Dict[str, Any]]:
    """
    Track gradient noise variance over the course of training.
    
    This function is meant to be called periodically during training to
    monitor how gradient noise evolves (typically decreases near convergence).
    
    Args:
        model: Neural network
        data_loader: Training data loader
        criterion: Loss function
        device: Computation device
        check_every_n_epochs: Frequency of checks
        num_samples_per_check: Number of gradient samples per check
        
    Returns:
        List of variance estimates with timestamps
    """
    noise_history = []
    
    # This function provides the interface - actual tracking should be
    # integrated into training loops in run_all_kaggle.py or similar
    
    return noise_history


def estimate_effective_batch_size(
    gradient_variance: float,
    gradient_norm: float,
    epsilon: float = 1e-8
) -> float:
    """
    Estimate effective batch size from gradient statistics.
    
    Effective batch size relates to how much gradient averaging reduces noise:
        B_eff ≈ ||E[g]||² / Var[g]
    
    This is useful for understanding the stochasticity of the optimization.
    
    Args:
        gradient_variance: Variance of gradients (σ²)
        gradient_norm: Norm of mean gradient
        epsilon: Numerical stability constant
        
    Returns:
        Estimated effective batch size
    """
    if gradient_variance < epsilon:
        return float('inf')  # No noise = infinite effective batch size
    
    signal = gradient_norm ** 2
    noise = gradient_variance
    
    return signal / (noise + epsilon)
