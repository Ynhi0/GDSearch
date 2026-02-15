"""
Advanced training utilities for deep learning.

This module provides:
- Reproducibility utilities (set_seed)
- Mixed Precision Training (AMP) wrapper
- Label Smoothing Loss
- Model EMA (Exponential Moving Average)
- Additional training enhancements
"""
# broad catch intentional - compatibility layers with PyTorch may use guarded broad
# catches to detect available APIs across versions; where broad catches are used they
# should be localized and documented.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, cast
import contextlib

# Compatibility imports for mixed precision APIs across torch versions
# Try explicit submodule imports first (pyright suggests these), then fallback to cuda submodules
try:
    from torch.amp.grad_scaler import GradScaler as _GradScaler  # type: ignore[reportPrivateImportUsage]
    from torch.amp.autocast_mode import autocast as _autocast  # type: ignore[reportPrivateImportUsage]
except Exception:
    try:
        from torch.cuda.amp.grad_scaler import GradScaler as _GradScaler
        from torch.cuda.amp.autocast_mode import autocast as _autocast
    except Exception:
        # Last-resort: do not attempt broad top-level imports (stubs are noisy); fall back to None and handle at runtime
        _GradScaler = None
        _autocast = None
import copy
import numpy as np
import random
import os
import warnings
import logging


def validate_pytorch_version(expected_version: str = "2.6.0", strict: bool = False):
    """
    Validate PyTorch version to prevent version-sensitive API failures.

    Args:
        expected_version: Expected PyTorch version (from requirements.txt)
        strict: If True, raise error on mismatch; if False, only warn

    Raises:
        RuntimeError: If strict=True and version mismatch detected
    """
    try:
        current_version = torch.__version__.split('+')[0]  # Remove cuda/cpu suffix
        major_minor_current = '.'.join(current_version.split('.')[:2])
        major_minor_expected = '.'.join(expected_version.split('.')[:2])

        if major_minor_current != major_minor_expected:
            msg = (
                f"PyTorch version mismatch detected!\n"
                f"  Expected: {expected_version} (from requirements.txt)\n"
                f"  Current:  {current_version}\n"
                f"  This may cause checkpoint save/load failures and optimizer behavior changes.\n"
                f"  Recommendation: pip install torch=={expected_version}"
            )
            if strict:
                raise RuntimeError(msg)
            else:
                # Prefer logging over warnings here to avoid noisy test output while still
                # providing visibility in logs.
                logging.warning(msg)
        else:
            logging.debug(f"PyTorch version OK: {current_version}")
    except Exception as e:
        logging.warning(f"Could not validate PyTorch version: {e}")


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.

    This function ensures deterministic behavior by default. If `deterministic`
    is False, the function will set RNG seeds but will NOT force cuDNN to disable
    `benchmark`, allowing performance optimizations for fixed-size workloads
    (e.g., ResNet on CIFAR/ImageNet) to remain active.

    Args:
        seed: Random seed value
        deterministic: If True (default) enforce deterministic cuDNN and algorithms.
                     If False, preserve performance-related settings (e.g., cudnn.benchmark).

    Note:
        Deterministic operations may reduce performance. Use deterministic=True in
        research for reproducibility, but consider disabling it for performance
        comparisons or production benchmarking.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Enforce deterministic behavior where possible
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            logging.debug("Could not set cudnn deterministic flags: %s", e, exc_info=True)

        try:
            torch.use_deterministic_algorithms(True)
            # Set CUBLAS environment variable for deterministic CUDA operations
            # Required for CUDA >= 10.2 when using deterministic algorithms
            if torch.cuda.is_available() and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        except Exception as e:
            # Older PyTorch versions may not support this
            logging.debug("Could not enable torch.use_deterministic_algorithms: %s", e, exc_info=True)
    else:
        # Preserve performance-related flags (do not override cudnn.benchmark)
        try:
            if hasattr(torch.backends, 'cudnn'):
                logging.debug("set_seed(..., deterministic=False) preserving cudnn.benchmark=%s", getattr(torch.backends.cudnn, 'benchmark', None))
        except Exception as e:
            logging.debug("Could not inspect cudnn backend flags: %s", e, exc_info=True)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.

    Label smoothing is a regularization technique that prevents the model
    from becoming overconfident by softening the hard targets.

    Args:
        smoothing: Label smoothing factor (0.0 to 1.0)
        reduction: Specifies the reduction to apply to the output

    Reference:
        "Rethinking the Inception Architecture for Computer Vision"
        Szegedy et al., CVPR 2016

    GAP 36 FIX - Entropy Floor Warning:
        Label smoothing enforces a mathematical minimum loss (Entropy Floor) > 0.
        For num_classes=10 and smoothing=0.1:
            min_loss ≈ -[0.9*log(0.9) + 9*(0.1/9)*log(0.1/9)] ≈ 0.54

        This means:
        - Loss will NEVER converge to 0, even with perfect predictions
        - Convergence analysis must account for this floor
        - Use get_entropy_floor() to compute the theoretical minimum
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    @staticmethod
    def compute_entropy_floor(num_classes: int, smoothing: float) -> float:
        """
        GAP 36 FIX: Compute the theoretical minimum loss for label smoothing.

        When using label smoothing, the loss cannot converge to 0.
        This function computes the entropy floor so convergence analysis
        can subtract it from the loss curve.

        Args:
            num_classes: Number of output classes
            smoothing: Label smoothing factor

        Returns:
            Entropy floor (minimum achievable loss)

        Example:
            For CIFAR-10 (10 classes) with smoothing=0.1:
            >>> LabelSmoothingCrossEntropy.compute_entropy_floor(10, 0.1)
            0.5404...  # Loss will never go below this
        """
        import math
        
        # LOGIC REVIEW FIX: Validate inputs to prevent mathematical errors
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        
        if smoothing == 0.0 or num_classes == 1:
            return 0.0

        # Smoothed target distribution: [1-s, s/(n-1), s/(n-1), ...]
        p_true = 1.0 - smoothing
        p_other = smoothing / (num_classes - 1)

        # Cross-entropy with perfect predictions (q = p):
        # H(p, q) = -sum(p * log(q)) = -p_true*log(p_true) - (n-1)*p_other*log(p_other)
        entropy = -p_true * math.log(p_true + 1e-12)
        if p_other > 0:
            entropy -= (num_classes - 1) * p_other * math.log(p_other + 1e-12)

        return entropy

    def get_entropy_floor(self, num_classes: int) -> float:
        """
        Get the entropy floor for this loss function instance.

        Args:
            num_classes: Number of output classes

        Returns:
            Minimum achievable loss value
        """
        return self.compute_entropy_floor(num_classes, self.smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.

        Args:
            pred: Predictions (logits) of shape [batch_size, num_classes]
            target: Target labels of shape [batch_size]

        Returns:
            Loss value
        """
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = torch.sum(-true_dist * log_preds, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters that is updated using
    exponential moving average. This can improve generalization and
    provide more stable predictions.

    Args:
        model: PyTorch model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA model

    Reference:
        "Mean teachers are better role models"
        Tarvainen & Valpola, NeurIPS 2017
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.decay = decay
        # Default to model's device if not specified
        if device is None:
            device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        self.device = device

        # Create shadow model
        self.shadow = copy.deepcopy(model).to(self.device)
        self.shadow.eval()

        # Store original model for reference
        self.model = model

        # Backup storage for restore functionality
        self.backup = {}

        # Disable gradients for shadow model
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None):
        """
        Update EMA parameters.

        Args:
            model: Model to update from (uses self.model if None)
        """
        if model is None:
            model = self.model

        # Move to same device as shadow
        model_params = {name: param.data.to(self.device)
                       for name, param in model.named_parameters()}

        # Update shadow parameters
        for name, shadow_param in self.shadow.named_parameters():
            if name in model_params:
                shadow_param.mul_(self.decay).add_(
                    model_params[name], alpha=1 - self.decay
                )

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            'shadow': self.shadow.state_dict(),
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.decay = state_dict.get('decay', self.decay)

    def apply_shadow(self, model: Optional[nn.Module] = None):
        """
        Apply EMA weights to model (for evaluation).
        Backs up current weights so they can be restored later.

        Args:
            model: Model to apply shadow weights to (uses self.model if None)
        """
        if model is None:
            model = self.model

        with torch.no_grad():
            # Backup current weights before applying shadow
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone()
            
            # Apply shadow weights
            for param, shadow_param in zip(model.parameters(), self.shadow.parameters()):
                param.data.copy_(shadow_param.data.to(param.device))

    def restore(self, model: Optional[nn.Module] = None):
        """
        Restore backed-up model weights (after evaluation with shadow weights).

        Args:
            model: Model to restore (uses self.model if None)

        Raises:
            RuntimeError: If no backup is available (apply_shadow() not called)
        """
        if model is None:
            model = self.model

        if not self.backup:
            raise RuntimeError(
                "No backup available. Call apply_shadow() before restore(). "
                "The typical workflow is: apply_shadow() -> evaluate -> restore()."
            )

        # Restore backed-up weights
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        
        # Clear backup after restore
        self.backup.clear()


class AMPWrapper:
    """
    Automatic Mixed Precision Training Wrapper.

    Wraps training loop with mixed precision support using torch.cuda.amp.
    Automatically handles gradient scaling and prevents underflow/overflow.

    Args:
        enabled: Whether to enable AMP (default: True if CUDA available)
        dtype: Data type for autocast (default: torch.float16)

    Usage:
        amp = AMPWrapper()

        for inputs, targets in loader:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            amp.backward(loss, optimizer)
            amp.step(optimizer)
            amp.update()

    Reference:
        PyTorch Automatic Mixed Precision documentation
        https://pytorch.org/docs/stable/amp.html
    """

    def __init__(self, enabled: Optional[bool] = None, dtype: torch.dtype = torch.float16):
        if enabled is None:
            enabled = torch.cuda.is_available()
        
        # LOGIC FIX: Validate enabled flag against CUDA availability to prevent device mismatch
        if enabled and not torch.cuda.is_available():
            logging.warning(
                "AMPWrapper: AMP enabled=True but CUDA not available. "
                "Disabling AMP (CPU does not support mixed precision training)."
            )
            enabled = False

        self.enabled = enabled
        self.dtype = dtype
        self.device_type = 'cuda' if enabled else 'cpu'

        if self.enabled:
            # Now safe: enabled=True implies CUDA is available
            assert torch.cuda.is_available(), "Internal error: AMP enabled but CUDA unavailable"
            # Prefer the new public API `torch.amp.GradScaler` when available
            # to avoid deprecation warnings for `torch.cuda.amp.GradScaler`.
            if callable(_GradScaler):
                import warnings as _warnings
                import inspect
                with _warnings.catch_warnings():
                    _warnings.filterwarnings('ignore', category=FutureWarning, message='.*GradScaler.*')
                    sig = None
                    try:
                        sig = inspect.signature(_GradScaler.__init__)
                    except Exception:
                        sig = None
                    kwargs = {'device_type': 'cuda'} if sig is not None and 'device_type' in sig.parameters else {}
                    ScalerCls = cast(Any, _GradScaler)
                    try:
                        self.scaler = ScalerCls(**kwargs)
                    except Exception:
                        try:
                            self.scaler = ScalerCls()
                        except Exception:
                            self.scaler = None
            else:
                # No scaler implementation available in this torch build
                self.scaler = None
        else:
            self.scaler = None

    def autocast(self):
        """
        Context manager for automatic mixed precision.

        Returns:
            Autocast context manager
        """
        if self.enabled and torch.cuda.is_available() and _autocast is not None:
            # Use a guarded call sequence at runtime while silencing static checks via cast to Any.
            # Some torch.autocast implementations require device_type, others accept only dtype.
            Sc = cast(Any, _autocast)
            try:
                return Sc(device_type=self.device_type, dtype=self.dtype)
            except TypeError:
                try:
                    return Sc(dtype=self.dtype)
                except TypeError:
                    return Sc()
        else:
            # Return no-op context manager for CPU or when autocast not available
            return contextlib.nullcontext()

    def backward(self, loss: torch.Tensor, optimizer: Any):
        """
        Backward pass with gradient scaling.

        Args:
            loss: Loss tensor
            optimizer: Optimizer-like object (supports zero_grad())
        """
        # Call optimizer.zero_grad on the provided optimizer-like object
        optimizer.zero_grad()

        if self.enabled and self.scaler is not None:
            # Use the scaler to scale the loss, which handles .backward()
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: Any):
        """
        Optimizer step with gradient unscaling.

        Args:
            optimizer: Optimizer-like object (supports step())
        """
        if self.enabled and self.scaler is not None:
            # Scaler.step expects an object with a .step() method (optimizer-like);
            # allow wrapper objects that implement step() even if not a subclass
            # of torch.optim.Optimizer.
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        """Update gradient scaler."""
        if self.enabled and self.scaler is not None:
            self.scaler.update()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        if self.enabled and self.scaler is not None:
            return {
                'scaler': self.scaler.state_dict(),
                'enabled': self.enabled,
                'dtype': str(self.dtype)
            }
        return {'enabled': False}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.enabled = state_dict.get('enabled', False)

        if self.enabled and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])


def get_loss_function(
    loss_type: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function with optional label smoothing.

    Args:
        loss_type: Type of loss ('cross_entropy', 'bce', 'mse')
        label_smoothing: Label smoothing factor for classification
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing, **kwargs)
        else:
            return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'mse':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_amp_wrapper(enabled: Optional[bool] = None) -> AMPWrapper:
    """
    Create AMP wrapper with automatic device detection.

    Args:
        enabled: Whether to enable AMP (auto-detect if None)

    Returns:
        AMPWrapper instance
    """
    return AMPWrapper(enabled=enabled)


def create_model_ema(model: nn.Module, decay: float = 0.9999) -> ModelEMA:
    """
    Create Model EMA tracker.

    Args:
        model: Model to track
        decay: EMA decay rate

    Returns:
        ModelEMA instance
    """
    return ModelEMA(model, decay=decay)
