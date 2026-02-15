#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Gradient Handling Module
================================

Scientific Justification:
-------------------------
Heavy-tailed gradients violate standard SGD convergence theory assumptions
(bounded variance, sub-Gaussian moments). Robust gradient methods are:
1. Theoretically grounded (see Karimireddy et al. 2021, Nazin et al. 2019)
2. Standard practice in production ML (Transformers, GANs, RL)
3. Improve reproducibility by reducing outlier sensitivity
4. Enable fair optimizer comparisons under realistic conditions

All methods are OPTIONAL and transparently logged for audit trails.

References:
-----------
- Karimireddy et al. "Mime: Mimicking centralized stochastic algorithms
  in federated learning." 2021.
- Zhang et al. "Why gradient clipping accelerates training: A theoretical
  justification for adaptivity." NeurIPS 2020.
- Loshchilov & Hutter. "Decoupled weight decay regularization." ICLR 2019.
"""
# broad catch intentional - numerical estimation routines may raise diverse
# numeric/third-party exceptions; broad catches in this module are localized and
# intended to preserve stability during long-running training jobs.

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging


class RobustGradientHandler:
    """
    Unified robust gradient handling for training stability.

    Features:
    ---------
    - Adaptive Gradient Clipping (AGC): Per-layer gradient scaling
    - Gradient Normalization: Stabilizes training dynamics
    - Heavy-tail detection: Statistical monitoring for pathological gradients
    - Trimmed-mean aggregation: Robust gradient estimation
    - Coordinate-wise median: Outlier-resistant gradient processing

    Scientific Rationale:
    ---------------------
    Heavy-tailed gradient distributions (detected via p-value < 0.05) indicate:
    1. SGD theory bounds may not hold (unbounded variance assumption violated)
    2. Learning rate may be too high for the current loss landscape
    3. Batch size may be too small (insufficient variance reduction)
    4. Numerical instability in loss computation

    Robust methods address these issues without artificially suppressing
    informative gradient signals.
    """

    def __init__(
        self,
        enabled: bool = False,
        clip_norm: Optional[float] = None,
        clip_percentile: float = 95.0,
        use_agc: bool = False,
        agc_eps: float = 1e-3,
        use_trimmed_mean: bool = False,
        trim_fraction: float = 0.1,
        use_coordinate_median: bool = False,
        monitor_heavy_tails: bool = True,
        heavy_tail_threshold: float = 0.05,
        log_interval: int = 10
    ):
        """
        Initialize robust gradient handler.

        Args:
            enabled: Master switch for all robust gradient methods
            clip_norm: Global gradient norm threshold (None = disabled)
            clip_percentile: Percentile-based clipping threshold
            use_agc: Enable Adaptive Gradient Clipping (per-layer)
            agc_eps: AGC epsilon for numerical stability
            use_trimmed_mean: Use trimmed mean for gradient aggregation
            trim_fraction: Fraction to trim from each tail (e.g., 0.1 = 10%)
            use_coordinate_median: Use coordinate-wise median (robust but expensive)
            monitor_heavy_tails: Enable heavy-tail detection diagnostics
            heavy_tail_threshold: p-value threshold for heavy-tail warning
            log_interval: Epochs between diagnostic logging
        """
        self.enabled = enabled
        self.clip_norm = clip_norm
        self.clip_percentile = clip_percentile
        self.use_agc = use_agc
        self.agc_eps = agc_eps
        self.use_trimmed_mean = use_trimmed_mean
        self.trim_fraction = trim_fraction
        self.use_coordinate_median = use_coordinate_median
        self.monitor_heavy_tails = monitor_heavy_tails
        self.heavy_tail_threshold = heavy_tail_threshold
        self.log_interval = log_interval

        # Diagnostics tracking
        self.clip_count = 0
        self.total_steps = 0
        self.gradient_norms = []
        self.heavy_tail_events = []

    def __call__(self, model: nn.Module, epoch: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply robust gradient processing to model gradients.

        Args:
            model: PyTorch model with computed gradients
            epoch: Current epoch number (for logging)

        Returns:
            Dictionary with diagnostic information:
            - 'grad_norm': Total gradient norm before clipping
            - 'clipped': Whether gradients were clipped
            - 'clip_ratio': Ratio of clipping applied
            - 'heavy_tail_detected': Heavy-tail diagnostic result
        """
        if not self.enabled:
            return {'grad_norm': 0.0, 'clipped': False, 'clip_ratio': 1.0, 'heavy_tail_detected': False}

        self.total_steps += 1
        diagnostics = {}

        # 1. Collect all gradients
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().flatten())

        if not all_grads:
            return {'grad_norm': 0.0, 'clipped': False, 'clip_ratio': 1.0, 'heavy_tail_detected': False}

        all_grads_tensor = torch.cat(all_grads)

        # 2. Compute gradient norm (before any modification)
        grad_norm = torch.norm(all_grads_tensor).item()
        self.gradient_norms.append(grad_norm)
        diagnostics['grad_norm'] = grad_norm

        # 3. Heavy-tail detection (statistical test)
        if self.monitor_heavy_tails and len(self.gradient_norms) > 30:
            heavy_tail_detected = self._detect_heavy_tails(all_grads_tensor.cpu().numpy())
            diagnostics['heavy_tail_detected'] = heavy_tail_detected

            if heavy_tail_detected:
                self.heavy_tail_events.append(self.total_steps)
                if epoch and self.total_steps % self.log_interval == 0:
                    logging.warning(
                        f"Heavy-tailed gradients detected at step {self.total_steps} "
                        f"(epoch {epoch}). Robust handling active."
                    )
        else:
            diagnostics['heavy_tail_detected'] = False

        # 4. Apply robust gradient processing
        clipped = False
        clip_ratio = 1.0

        # 4a. Trimmed-mean aggregation (for distributed/multi-batch scenarios)
        if self.use_trimmed_mean:
            self._apply_trimmed_mean(model)

        # 4b. Coordinate-wise median (expensive but very robust)
        if self.use_coordinate_median:
            self._apply_coordinate_median(model)

        # 4c. Adaptive Gradient Clipping (per-layer)
        if self.use_agc:
            clip_ratio = self._apply_agc(model)
            if clip_ratio < 1.0:
                clipped = True
                self.clip_count += 1

        # 4d. Global gradient clipping
        elif self.clip_norm is not None:
            clip_ratio = self._apply_global_clip(model)
            if clip_ratio < 1.0:
                clipped = True
                self.clip_count += 1

        diagnostics['clipped'] = clipped
        diagnostics['clip_ratio'] = clip_ratio

        # 5. Periodic logging
        if epoch and epoch % self.log_interval == 0 and self.total_steps % 100 == 0:
            logging.info(
                f"Robust Gradient Stats (Epoch {epoch}): "
                f"Norm={grad_norm:.3e}, Clipped={clipped}, "
                f"Clip Ratio={clip_ratio:.3f}, "
                f"Total Clips={self.clip_count}/{self.total_steps}"
            )

        return diagnostics

    def _detect_heavy_tails(self, grads: np.ndarray) -> bool:
        """
        Detect heavy-tailed gradient distributions using statistical test.

        Uses a conservative multi-criteria approach to avoid false positives:
        1. Kurtosis test: excess kurtosis indicates non-Gaussian heavy tails
        2. IQR-based outlier detection: extreme values beyond 3*IQR
        3. Requires BOTH conditions to trigger (conservative approach)

        Scientific Note:
        ----------------
        Neural network gradients are typically NOT Gaussian. Using p-value alone
        from kurtosistest would trigger constantly. Instead, we require:
        - Statistically significant kurtosis (p < threshold)
        - AND a substantial fraction of extreme outliers (> 5%)

        This catches truly pathological cases (exploding gradients, numerical
        instability) while ignoring the natural non-Gaussianity of DNN gradients.

        Args:
            grads: Gradient values as numpy array

        Returns:
            True if heavy tails detected (pathological case)
        """
        try:
            from scipy import stats

            # Safety check: need enough samples for statistical tests
            if len(grads) < 100:
                return False

            # Subsample for efficiency on large gradient tensors
            if len(grads) > 10000:
                indices = np.random.choice(len(grads), 10000, replace=False)
                grads = grads[indices]

            # Test for excess kurtosis (normal distribution has kurtosis=3)
            # Note: kurtosistest requires n >= 20
            try:
                _, p_value = stats.kurtosistest(grads)
            except Exception:
                # Fall back to simple kurtosis if test fails
                kurtosis = stats.kurtosis(grads)
                # LOGIC REVIEW FIX: Use more conservative threshold
                # Fisher's kurtosis: normal=0, moderately heavy-tail > 2, very heavy > 5
                # DNN gradients naturally have kurtosis ~1-3, so use 5 as pathological threshold
                p_value = 0.01 if kurtosis > 5.0 else 0.5

            # IQR-based extreme value detection (robust method)
            q1, q3 = np.percentile(grads, [25, 75])
            iqr = q3 - q1
            if iqr < 1e-10:
                # Near-zero variance gradients (vanishing) - not heavy-tail
                return False

            # Count values beyond 3*IQR (traditional outlier threshold)
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            extreme_count = np.sum((grads < lower_bound) | (grads > upper_bound))
            extreme_ratio = extreme_count / len(grads)

            # Conservative criteria: BOTH must be true for pathological detection
            # - p_value < threshold (statistically significant kurtosis)
            # - extreme_ratio > 5% (substantial fraction of outliers)
            # This is much more conservative than OR logic
            is_heavy_tail = (p_value < self.heavy_tail_threshold) and (extreme_ratio > 0.05)

            return is_heavy_tail

        except Exception as e:
            logging.debug(f"Heavy-tail detection failed: {e}")
            return False

    def _apply_trimmed_mean(self, model: nn.Module) -> None:
        """
        Apply trimmed-mean gradient aggregation (trim extreme values).

        LOGIC REVIEW FIX: Preserves gradient direction by clipping to percentile
        thresholds instead of replacing entire gradient with scalar mean.
        
        This clips extreme gradient values while maintaining spatial structure.
        Most useful for distributed training or when aggregating gradients from
        multiple sources. For single-batch training, it provides minimal benefit.
        """
        for param in model.parameters():
            if param.grad is None:
                continue
            
            # Find percentile thresholds (faster than full sort)
            grad_flat = param.grad.flatten()
            lower_threshold = torch.quantile(grad_flat, self.trim_fraction)
            upper_threshold = torch.quantile(grad_flat, 1.0 - self.trim_fraction)
            
            # Clip gradients to trimmed range (preserves direction)
            param.grad.clamp_(lower_threshold, upper_threshold)

    def _apply_coordinate_median(self, model: nn.Module) -> None:
        """
        Apply coordinate-wise median (very robust but expensive).

        LOGIC FIX: Coordinate-wise median filtering for outlier suppression.
        Clamps extreme gradient values to a range around the median, preserving
        gradient structure while removing extreme outliers.
        
        Note: For single-batch training with no gradient accumulation, this
        provides minimal benefit. Most useful with gradient accumulation or
        distributed training where multiple gradient estimates are available.
        """
        for param in model.parameters():
            if param.grad is not None:
                # Compute median of gradient tensor
                grad_median = param.grad.median()
                # Compute robust scale estimate (MAD - Median Absolute Deviation)
                mad = torch.median(torch.abs(param.grad - grad_median))
                # Clamp outliers to 3*MAD range (standard robust threshold)
                # If MAD is too small, use absolute threshold
                scale = max(mad.item(), 1e-3)
                param.grad.clamp_(grad_median - 3 * scale, grad_median + 3 * scale)

    def _apply_agc(self, model: nn.Module) -> float:
        """
        Apply Adaptive Gradient Clipping (per-layer scaling).

        AGC clips gradients relative to parameter norms, preventing
        destabilization from layers with small parameters.
        
        NOTE: clip_percentile is used as a percentage (e.g., 95.0 means 0.95 * param_norm).
        This differs slightly from the paper's lambda parameter but achieves the same goal.
        To match paper exactly: set clip_percentile to desired lambda * 100 (e.g., 1.0 for lambda=0.01).

        Returns:
            Minimum clip ratio applied across layers
        """
        min_clip_ratio: float = 1.0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.detach())
                grad_norm = torch.norm(param.grad.detach())

                if param_norm > self.agc_eps and grad_norm > self.agc_eps:
                    # Clip gradient norm to be proportional to parameter norm
                    max_norm = self.clip_percentile * param_norm / 100.0
                    clip_coef = max_norm / (grad_norm + 1e-6)

                    if clip_coef < 1.0:
                        param.grad.mul_(clip_coef)
                        # Convert tensor to float for comparison
                        clip_coef_float: float = float(clip_coef.item() if isinstance(clip_coef, torch.Tensor) else clip_coef)
                        min_clip_ratio = min(min_clip_ratio, clip_coef_float)

        return min_clip_ratio

    def _apply_global_clip(self, model: nn.Module) -> float:
        """
        Apply global gradient norm clipping.

        Returns:
            Clip ratio applied (1.0 = no clipping)
        """
        if self.clip_norm is None:
            return 1.0

        clip_norm_value: float = float(self.clip_norm)

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_norm_value
        )

        # Convert tensor to float for comparison
        total_norm_float: float = float(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)

        if total_norm_float > clip_norm_value:
            return clip_norm_value / total_norm_float
        return 1.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get diagnostic statistics for logging/analysis.

        Returns:
            Dictionary with:
            - 'mean_grad_norm': Average gradient norm
            - 'max_grad_norm': Maximum gradient norm
            - 'clip_fraction': Fraction of steps that required clipping
            - 'heavy_tail_fraction': Fraction of steps with heavy tails
        """
        if not self.gradient_norms:
            return {
                'mean_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'clip_fraction': 0.0,
                'heavy_tail_fraction': 0.0
            }

        return {
            'mean_grad_norm': np.mean(self.gradient_norms),
            'max_grad_norm': np.max(self.gradient_norms),
            'clip_fraction': self.clip_count / max(1, self.total_steps),
            'heavy_tail_fraction': len(self.heavy_tail_events) / max(1, self.total_steps)
        }

    def reset_statistics(self) -> None:
        """Reset diagnostic counters for new experiment."""
        self.gradient_norms = []
        self.heavy_tail_events = []
        self.clip_count = 0
        self.total_steps = 0


def create_robust_gradient_handler(
    enabled: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> RobustGradientHandler:
    """
    Factory function to create robust gradient handler from config.

    Args:
        enabled: Master switch for robust gradient handling
        config: Configuration dictionary with optional keys:
            - 'clip_norm': Global gradient clipping threshold
            - 'clip_percentile': Percentile-based clipping
            - 'use_agc': Enable adaptive gradient clipping
            - 'use_trimmed_mean': Enable trimmed-mean aggregation
            - 'use_coordinate_median': Enable coordinate-wise median
            - 'monitor_heavy_tails': Enable heavy-tail detection

    Returns:
        Configured RobustGradientHandler instance
    """
    if config is None:
        config = {}

    return RobustGradientHandler(
        enabled=enabled,
        clip_norm=config.get('clip_norm', None),
        clip_percentile=config.get('clip_percentile', 95.0),
        use_agc=config.get('use_agc', False),
        agc_eps=config.get('agc_eps', 1e-3),
        use_trimmed_mean=config.get('use_trimmed_mean', False),
        trim_fraction=config.get('trim_fraction', 0.1),
        use_coordinate_median=config.get('use_coordinate_median', False),
        monitor_heavy_tails=config.get('monitor_heavy_tails', True),
        heavy_tail_threshold=config.get('heavy_tail_threshold', 0.05),
        log_interval=config.get('log_interval', 10)
    )


class HuberLoss(nn.Module):
    """
    Huber Loss (robust to outliers).

    Combines L2 loss (for small errors) with L1 loss (for large errors).
    More robust than MSE to extreme gradient magnitudes.

    Scientific Justification:
    -------------------------
    Huber loss reduces influence of outlier labels/predictions without
    completely ignoring them (unlike trimmed losses). Widely used in
    robust regression and reinforcement learning.

    Args:
        delta: Threshold between L2 and L1 behavior (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        error = pred - target
        abs_error = torch.abs(error)

        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def get_robust_loss_function(
    loss_type: str = 'cross_entropy',
    robust: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory for robust loss functions.

    Args:
        loss_type: 'cross_entropy', 'mse', 'huber', 'focal', 'label_smoothing'
        robust: If True, use robust variant when available
        **kwargs: Additional arguments for loss function

    Returns:
        PyTorch loss module
    """
    if loss_type == 'cross_entropy':
        if robust and 'label_smoothing' in kwargs:
            return nn.CrossEntropyLoss(label_smoothing=float(kwargs['label_smoothing']))
        return nn.CrossEntropyLoss()

    elif loss_type == 'mse':
        if robust:
            return HuberLoss(delta=kwargs.get('huber_delta', 1.0))
        return nn.MSELoss()

    elif loss_type == 'huber':
        return HuberLoss(delta=kwargs.get('huber_delta', 1.0))

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
