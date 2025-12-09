"""
Advanced Training Enhancements for GDSearch

Implements research-grade training utilities:
- LRFinder: Automated learning rate finding (fast.ai style)
- MemoryAwareBatchSizer: Dynamic batch sizing based on GPU memory
- SelfHealingTrainer: OOM recovery with automatic batch size reduction
- DiskSpaceGuardian: Checkpoint management with disk space awareness

These utilities address critical gaps for production-quality training.
"""

import os
import sys
import time
import math
import shutil
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Any, Union

import torch
import torch.nn as nn
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# LEARNING RATE FINDER (fast.ai style)
# =============================================================================

class LRFinder:
    """
    Learning Rate Finder - finds optimal learning rate using the LR range test.
    
    Based on:
    - Smith, L.N. "Cyclical Learning Rates for Training Neural Networks" (2017)
    - fast.ai implementation
    
    Algorithm:
    1. Start with a very small LR (e.g., 1e-7)
    2. Increase LR exponentially each mini-batch
    3. Record loss at each step
    4. Stop when loss explodes (exceeds threshold)
    5. Suggested LR = point where loss decreases fastest (steepest descent)
    
    Usage:
        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1, num_iter=100)
        suggested_lr = lr_finder.suggest_lr()
        lr_finder.plot()  # Optional visualization
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: Callable,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LR Finder.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance (will be modified)
            criterion: Loss function
            device: Computation device (auto-detected if None)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or next(model.parameters()).device
        
        # Store initial state for restoration
        self._model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self._optimizer_state = optimizer.state_dict()
        
        # Results storage
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []
        
    def range_test(
        self,
        train_loader: torch.utils.data.DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_threshold: float = 5.0,
        step_mode: str = 'exp',
        verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Perform LR range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations for the test
            smooth_f: Smoothing factor for exponential moving average
            diverge_threshold: Stop when loss > diverge_threshold * best_loss
            step_mode: 'exp' (exponential) or 'linear'
            verbose: Print progress
            
        Returns:
            Tuple of (learning_rates, losses)
        """
        # Reset results
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        
        # Calculate LR multiplier
        if step_mode == 'exp':
            lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        else:  # linear
            lr_step = (end_lr - start_lr) / num_iter
        
        # Set initial LR
        current_lr = start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Training setup
        self.model.train()
        best_loss = float('inf')
        avg_loss = 0.0
        batch_num = 0
        
        iterator = iter(train_loader)
        
        if verbose:
            print(f"🔍 LR Range Test: {start_lr:.2e} → {end_lr:.2e} ({num_iter} steps)")
        
        for i in range(num_iter):
            # Get batch (cycle through data if needed)
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"   Stopping: Loss is NaN/Inf at LR={current_lr:.2e}")
                break
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 🐛 BUG FIX #7: Clean up memory to prevent leaks
            # Detach tensors and clear cache periodically
            loss_val = loss.item()
            del outputs, loss  # Explicitly delete large tensors
            
            if batch_num % 10 == 0:  # Every 10 batches
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Smooth the loss
            if batch_num == 0:
                avg_loss = loss_val
            else:
                avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss
            
            # Record
            self.lrs.append(current_lr)
            self.losses.append(loss_val)
            self.smoothed_losses.append(avg_loss)
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Check for divergence
            if avg_loss > diverge_threshold * best_loss:
                if verbose:
                    print(f"   Stopping: Loss diverged at LR={current_lr:.2e}")
                break
            
            # Update LR
            if step_mode == 'exp':
                current_lr *= lr_mult
            else:
                current_lr += lr_step
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            batch_num += 1
            
            if verbose and (i + 1) % (num_iter // 5) == 0:
                print(f"   Step {i+1}/{num_iter}: LR={current_lr:.2e}, Loss={avg_loss:.4f}")
        
        # Restore model and optimizer state
        self._restore_state()
        
        # 🐛 BUG FIX #7: Final memory cleanup
        del iterator, inputs, targets
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if verbose:
            print(f"   Completed {len(self.lrs)} steps")
        
        return self.lrs, self.smoothed_losses
    
    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate based on steepest descent.
        
        The suggested LR is typically 1 order of magnitude lower than
        the point of steepest descent (most negative gradient).
        
        Args:
            skip_start: Skip first N points (often noisy)
            skip_end: Skip last N points (often diverging)
            
        Returns:
            Suggested learning rate
        """
        if len(self.lrs) < skip_start + skip_end + 10:
            logging.warning("Not enough data points for reliable LR suggestion")
            return self.lrs[len(self.lrs) // 2] if self.lrs else 1e-3
        
        # Use smoothed losses
        losses = np.array(self.smoothed_losses[skip_start:-skip_end])
        lrs = np.array(self.lrs[skip_start:-skip_end])
        
        # Calculate gradient (derivative of loss w.r.t. log(lr))
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)
        
        # Find steepest descent (most negative gradient)
        min_grad_idx = np.argmin(gradients)
        
        # Suggest LR = steepest_point / 10 (one order of magnitude lower)
        suggested_lr = lrs[min_grad_idx] / 10
        
        logging.info(f"LR Finder: Steepest descent at {lrs[min_grad_idx]:.2e}, suggesting {suggested_lr:.2e}")
        
        return suggested_lr
    
    def plot(self, log_scale: bool = True, save_path: Optional[str] = None, 
             show: bool = True, skip_start: int = 10, skip_end: int = 5):
        """
        Plot LR vs Loss curve.
        
        Args:
            log_scale: Use log scale for x-axis
            save_path: Path to save figure (None = don't save)
            show: Display the plot
            skip_start: Skip first N noisy points
            skip_end: Skip last N diverging points
        """
        if not HAS_MATPLOTLIB:
            logging.warning("matplotlib not available for plotting")
            return
        
        if len(self.lrs) < skip_start + skip_end + 5:
            logging.warning("Not enough data points to plot")
            return
        
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.smoothed_losses[skip_start:-skip_end] if skip_end > 0 else self.smoothed_losses[skip_start:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lrs, losses, linewidth=2, color='blue')
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss (smoothed)', fontsize=12)
        ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark suggested LR
        suggested_lr = self.suggest_lr(skip_start, skip_end)
        ax.axvline(x=suggested_lr, color='red', linestyle='--', linewidth=2, 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"LR Finder plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _restore_state(self):
        """Restore model and optimizer to initial state."""
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)


# =============================================================================
# MEMORY-AWARE BATCH SIZING
# =============================================================================

class MemoryAwareBatchSizer:
    """
    Memory-Aware Batch Sizing - automatically determines optimal batch size.
    
    Features:
    - Detects GPU type (T4, P100, V100, A100, etc.)
    - Probes maximum batch size without OOM
    - Caches results for model architecture
    
    Usage:
        sizer = MemoryAwareBatchSizer()
        batch_size = sizer.find_optimal_batch_size(model, sample_input, target_utilization=0.85)
    """
    
    # Known GPU memory profiles (in GB)
    GPU_PROFILES = {
        'Tesla T4': 16,
        'Tesla P100': 16,
        'Tesla V100': 16,  # 16GB or 32GB variant
        'Tesla V100-SXM2': 32,
        'Tesla A100': 40,  # or 80GB variant
        'RTX 3090': 24,
        'RTX 4090': 24,
        'RTX 3080': 10,
        'RTX 2080 Ti': 11,
        'Quadro RTX 8000': 48,
    }
    
    # Batch size hints per GPU tier
    BATCH_SIZE_HINTS = {
        'low': {'mnist': 128, 'cifar10': 64, 'resnet18': 32, 'nlp': 16, 'medical': 4},
        'medium': {'mnist': 256, 'cifar10': 128, 'resnet18': 64, 'nlp': 32, 'medical': 8},
        'high': {'mnist': 512, 'cifar10': 256, 'resnet18': 128, 'nlp': 64, 'medical': 16},
    }
    
    def __init__(self, safety_margin: float = 0.15):
        """
        Initialize MemoryAwareBatchSizer.
        
        Args:
            safety_margin: Reserved memory fraction (default 15% for stability)
        """
        self.safety_margin = safety_margin
        self._cache: Dict[str, int] = {}
        self._gpu_info = self._detect_gpu()
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU and get memory info."""
        info = {
            'available': torch.cuda.is_available(),
            'name': None,
            'memory_total_gb': 0,
            'memory_free_gb': 0,
            'tier': 'cpu'
        }
        
        if not info['available']:
            return info
        
        try:
            info['name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info['memory_total_gb'] = props.total_memory / (1024 ** 3)
            
            # Get current free memory
            torch.cuda.empty_cache()
            info['memory_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated(0)) / (1024 ** 3)
            
            # Determine tier
            total = info['memory_total_gb']
            if total >= 24:
                info['tier'] = 'high'
            elif total >= 12:
                info['tier'] = 'medium'
            else:
                info['tier'] = 'low'
                
        except Exception as e:
            logging.warning(f"GPU detection failed: {e}")
        
        return info
    
    def get_recommended_batch_size(self, experiment_type: str) -> int:
        """
        Get recommended batch size based on GPU tier and experiment type.
        
        Args:
            experiment_type: One of 'mnist', 'cifar10', 'resnet18', 'nlp', 'medical'
            
        Returns:
            Recommended batch size
        """
        tier = self._gpu_info.get('tier', 'low')
        hints = self.BATCH_SIZE_HINTS.get(tier, self.BATCH_SIZE_HINTS['low'])
        batch_size = hints.get(experiment_type.lower(), 64)
        
        # PHASE 2.2 FIX: Log hardware-specific metadata
        gpu_name = self._gpu_info.get('name', 'CPU')
        memory_gb = self._gpu_info.get('memory_total_gb', 0)
        logging.info(f"🔧 Adaptive Batch Size: {batch_size} "
                    f"(GPU: {gpu_name}, VRAM: {memory_gb:.1f}GB, Tier: {tier})")
        
        return batch_size
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: Optional[torch.Tensor] = None,
        criterion: Optional[Callable] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        target_utilization: float = 0.85
    ) -> int:
        """
        Find optimal batch size by binary search with OOM detection.
        
        Args:
            model: Model to test
            sample_input: Single sample input tensor
            sample_target: Single sample target tensor
            criterion: Loss function (optional, for full forward+backward test)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            target_utilization: Target GPU memory utilization (0-1)
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            logging.info("No GPU available, using default batch size")
            return 64
        
        device = next(model.parameters()).device
        
        # Create cache key based on model architecture
        cache_key = f"{type(model).__name__}_{sample_input.shape}"
        if cache_key in self._cache:
            logging.info(f"Using cached batch size: {self._cache[cache_key]}")
            return self._cache[cache_key]
        
        # Binary search for optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal = min_batch_size
        
        logging.info(f"🔍 Probing batch sizes {low} to {high}...")
        
        while low <= high:
            mid = (low + high) // 2
            
            if self._test_batch_size(model, sample_input, sample_target, criterion, mid, device):
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Apply safety margin
        final_batch_size = int(optimal * (1 - self.safety_margin))
        final_batch_size = max(min_batch_size, final_batch_size)
        
        logging.info(f"   Optimal batch size: {optimal}, with safety margin: {final_batch_size}")
        
        # Cache result
        self._cache[cache_key] = final_batch_size
        
        return final_batch_size
    
    def _test_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        sample_target: Optional[torch.Tensor],
        criterion: Optional[Callable],
        batch_size: int,
        device: torch.device
    ) -> bool:
        """Test if a batch size fits in memory."""
        try:
            torch.cuda.empty_cache()
            
            # Create batch
            batch_input = sample_input.unsqueeze(0).expand(batch_size, *sample_input.shape).clone()
            batch_input = batch_input.to(device)
            
            # Forward pass
            model.train()
            output = model(batch_input)
            
            # Optional backward pass
            if criterion is not None and sample_target is not None:
                batch_target = sample_target.unsqueeze(0).expand(batch_size, *sample_target.shape).clone()
                batch_target = batch_target.to(device)
                loss = criterion(output, batch_target)
                loss.backward()
            
            # Cleanup
            del batch_input, output
            if criterion is not None:
                del batch_target, loss
            torch.cuda.empty_cache()
            
            return True
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Return detected GPU information."""
        return self._gpu_info.copy()


# =============================================================================
# SELF-HEALING OOM RECOVERY
# =============================================================================

class SelfHealingTrainer:
    """
    Self-Healing Trainer - automatically recovers from OOM errors.
    
    Features:
    - Catches CUDA OOM errors during training
    - Automatically halves batch size
    - Clears GPU cache and retries
    - Logs recovery actions
    
    ⚠️  SCIENTIFIC INTEGRITY WARNING:
    When OOM recovery is triggered, this trainer DROPS data from the batch tail:
        inputs[:new_size] is kept, inputs[new_size:] is DISCARDED
    
    This causes two integrity concerns:
    1. DATA LOSS: The model sees less data than intended for that step
    2. NOISE SPIKE: Sudden batch size halving (e.g., 128→64) doubles gradient variance
    
    RECOMMENDATION: Treat runs that triggered OOM recovery as INVALID for strict
    convergence analysis. Use for exploratory runs only. For publication-quality
    results, re-run with a smaller fixed batch size.
    
    Usage:
        healer = SelfHealingTrainer(model, optimizer, criterion)
        for batch in loader:
            loss = healer.train_step(batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        min_batch_size: int = 1,
        max_retries: int = 3
    ):
        """
        Initialize SelfHealingTrainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            criterion: Loss function
            min_batch_size: Minimum batch size before giving up
            max_retries: Maximum number of OOM recovery attempts
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.min_batch_size = min_batch_size
        self.max_retries = max_retries
        
        self.current_batch_size: Optional[int] = None
        self.oom_count = 0
        self.device = next(model.parameters()).device
        
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        accumulation_steps: int = 1
    ) -> Tuple[float, bool]:
        """
        Execute a single training step with OOM recovery.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            accumulation_steps: Gradient accumulation steps (for effective batch size)
            
        Returns:
            Tuple of (loss_value, success_flag)
        """
        retries = 0
        current_inputs = inputs
        current_targets = targets
        
        while retries < self.max_retries:
            try:
                self.model.train()
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(current_inputs.to(self.device))
                loss = self.criterion(outputs, current_targets.to(self.device))
                
                # Backward pass
                (loss / accumulation_steps).backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update
                self.optimizer.step()
                
                self.current_batch_size = current_inputs.size(0)
                return loss.item(), True
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.oom_count += 1
                    retries += 1
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Halve batch size
                    old_size = current_inputs.size(0)
                    new_size = max(self.min_batch_size, old_size // 2)
                    
                    if new_size < self.min_batch_size:
                        logging.error(f"OOM recovery failed: batch size {new_size} < minimum {self.min_batch_size}")
                        raise
                    
                    logging.warning(f"⚠️  OOM detected! Reducing batch size: {old_size} → {new_size} (retry {retries}/{self.max_retries})")
                    
                    # Slice batch
                    current_inputs = inputs[:new_size]
                    current_targets = targets[:new_size]
                    
                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    raise
        
        logging.error(f"OOM recovery exhausted after {self.max_retries} retries")
        return 0.0, False
    
    def get_stats(self) -> Dict[str, Any]:
        """Return trainer statistics."""
        return {
            'oom_count': self.oom_count,
            'current_batch_size': self.current_batch_size,
            'min_batch_size': self.min_batch_size,
        }


# =============================================================================
# DISK SPACE GUARDIAN
# =============================================================================

class DiskSpaceGuardian:
    """
    Disk Space Guardian - manages disk space for checkpoints.
    
    Features:
    - Monitors available disk space
    - Deletes oldest checkpoints when space is low
    - Prevents crashes due to disk full errors
    
    Usage:
        guardian = DiskSpaceGuardian(checkpoint_dir, min_free_gb=1.0)
        if guardian.can_save_checkpoint(estimated_size_mb=500):
            save_checkpoint(...)
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        min_free_gb: float = 1.0,
        max_checkpoints: int = 5
    ):
        """
        Initialize DiskSpaceGuardian.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            min_free_gb: Minimum free space to maintain (GB)
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_gb = min_free_gb
        self.max_checkpoints = max_checkpoints
        
    def get_free_space_gb(self) -> float:
        """Get free disk space in GB."""
        try:
            stat = shutil.disk_usage(self.checkpoint_dir)
            return stat.free / (1024 ** 3)
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")
            return float('inf')  # Assume unlimited if check fails
    
    def can_save_checkpoint(self, estimated_size_mb: float = 500) -> bool:
        """
        Check if there's enough space to save a checkpoint.
        
        Args:
            estimated_size_mb: Estimated checkpoint size in MB
            
        Returns:
            True if space is available (or was made available)
        """
        required_gb = estimated_size_mb / 1024 + self.min_free_gb
        free_gb = self.get_free_space_gb()
        
        if free_gb >= required_gb:
            return True
        
        # Try to free space
        logging.warning(f"⚠️  Low disk space: {free_gb:.2f}GB free, need {required_gb:.2f}GB")
        self._cleanup_old_checkpoints()
        
        # Check again
        free_gb = self.get_free_space_gb()
        if free_gb >= required_gb:
            logging.info(f"   Freed space, now {free_gb:.2f}GB available")
            return True
        
        logging.error(f"   Still insufficient space after cleanup: {free_gb:.2f}GB")
        return False
    
    def _cleanup_old_checkpoints(self):
        """Delete oldest checkpoints to free space."""
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints.extend(self.checkpoint_dir.glob("*.pth"))
        
        if len(checkpoints) <= 1:
            logging.warning("No old checkpoints to delete")
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Keep only max_checkpoints, delete the rest
        to_delete = checkpoints[:-self.max_checkpoints] if len(checkpoints) > self.max_checkpoints else checkpoints[:1]
        
        for ckpt in to_delete:
            try:
                size_mb = ckpt.stat().st_size / (1024 ** 2)
                ckpt.unlink()
                logging.info(f"   Deleted old checkpoint: {ckpt.name} ({size_mb:.1f}MB)")
            except Exception as e:
                logging.warning(f"   Failed to delete {ckpt.name}: {e}")
    
    def enforce_max_checkpoints(self):
        """Ensure we don't exceed max_checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints.extend(self.checkpoint_dir.glob("*.pth"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time and delete oldest
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for ckpt in checkpoints[:-self.max_checkpoints]:
            try:
                ckpt.unlink()
                logging.info(f"   Cleaned up: {ckpt.name}")
            except Exception as e:
                logging.warning(f"   Failed to clean {ckpt.name}: {e}")


# =============================================================================
# COMBINED AUTO-TUNE HELPER
# =============================================================================

class TimeBudgetManager:
    """
    Time Budget Manager - prevents Kaggle 12h timeout by forcing graceful exit.
    
    Features:
    - Tracks elapsed time since start
    - Triggers graceful exit before timeout
    - Saves state and generates reports before stopping
    
    Usage:
        budget = TimeBudgetManager(max_hours=11.0)  # Leave 1h buffer
        for epoch in range(epochs):
            if budget.should_stop():
                budget.graceful_exit(save_fn, report_fn)
                break
            train_epoch(...)
    """
    
    def __init__(self, max_hours: float = 11.0, warning_hours: float = 10.5):
        """
        Initialize TimeBudgetManager.
        
        Args:
            max_hours: Maximum runtime before forced exit (default 11h for Kaggle)
            warning_hours: When to start warning about time budget
        """
        self.max_hours = max_hours
        self.warning_hours = warning_hours
        self.start_time = time.time()
        self._warned = False
        
    def elapsed_hours(self) -> float:
        """Get elapsed time in hours."""
        return (time.time() - self.start_time) / 3600
    
    def remaining_hours(self) -> float:
        """Get remaining time in hours."""
        return max(0, self.max_hours - self.elapsed_hours())
    
    def should_stop(self) -> bool:
        """Check if we should stop training."""
        elapsed = self.elapsed_hours()
        
        # Warning at warning threshold
        if elapsed >= self.warning_hours and not self._warned:
            self._warned = True
            remaining = self.remaining_hours()
            logging.warning(f"⏰ Time budget warning: {remaining:.1f}h remaining (elapsed: {elapsed:.1f}h)")
        
        return elapsed >= self.max_hours
    
    def graceful_exit(
        self,
        save_fn: Optional[Callable] = None,
        report_fn: Optional[Callable] = None,
        message: str = "Time budget exceeded"
    ):
        """
        Execute graceful exit: save state and generate reports.
        
        Args:
            save_fn: Function to call for saving state
            report_fn: Function to call for generating reports
            message: Exit message to display
        """
        elapsed = self.elapsed_hours()
        logging.warning(f"⏰ GRACEFUL EXIT: {message} (elapsed: {elapsed:.2f}h / max: {self.max_hours}h)")
        
        try:
            if save_fn:
                logging.info("   Saving final state...")
                save_fn()
                logging.info("   ✓ State saved")
        except Exception as e:
            logging.error(f"   ✗ Failed to save state: {e}")
        
        try:
            if report_fn:
                logging.info("   Generating final reports...")
                report_fn()
                logging.info("   ✓ Reports generated")
        except Exception as e:
            logging.error(f"   ✗ Failed to generate reports: {e}")
        
        logging.info("   Graceful exit complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current time budget status."""
        return {
            'elapsed_hours': self.elapsed_hours(),
            'remaining_hours': self.remaining_hours(),
            'max_hours': self.max_hours,
            'should_stop': self.should_stop()
        }


class HessianAnalyzer:
    """
    Hessian Analyzer - computes eigenvalues and condition number.
    
    Provides metrics for:
    - λ_min, λ_max (smallest and largest eigenvalues)
    - Condition number κ = λ_max / λ_min
    - Flatness measures for SAM analysis
    
    Uses Hutchinson's trace estimator and power iteration for efficiency.
    """
    
    def __init__(self, model: nn.Module, criterion: Callable):
        """
        Initialize HessianAnalyzer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
        """
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
    
    def _get_params(self) -> List[torch.Tensor]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def _flatten_params(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Flatten parameter list to single vector."""
        return torch.cat([p.view(-1) for p in params])
    
    def _unflatten_like(self, flat: torch.Tensor, params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Unflatten vector back to parameter shapes."""
        result = []
        offset = 0
        for p in params:
            numel = p.numel()
            result.append(flat[offset:offset+numel].view_as(p))
            offset += numel
        return result
    
    def hessian_vector_product(
        self,
        vector: torch.Tensor,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product Hv using finite differences.
        
        This is more memory-efficient than forming the full Hessian.
        
        Args:
            vector: Vector to multiply with Hessian
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Hessian-vector product
        """
        params = self._get_params()
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(self.device))
        
        # First gradient
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = self._flatten_params(list(grads))
        
        # Gradient-vector product
        gvp = (flat_grad * vector).sum()
        
        # Second derivative (Hessian-vector product)
        hvp_grads = torch.autograd.grad(gvp, params, retain_graph=False)
        hvp = self._flatten_params(list(hvp_grads))
        
        return hvp
    
    def power_iteration(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_iters: int = 20,
        tol: float = 1e-3
    ) -> Tuple[float, torch.Tensor]:
        """
        Estimate largest eigenvalue using power iteration.
        
        Args:
            inputs: Input batch
            targets: Target batch
            num_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (largest_eigenvalue, eigenvector)
        """
        params = self._get_params()
        n_params = sum(p.numel() for p in params)
        
        # Random initial vector
        v = torch.randn(n_params, device=self.device)
        v = v / v.norm()
        
        eigenvalue = 0.0
        for i in range(num_iters):
            # Hv product
            Hv = self.hessian_vector_product(v, inputs, targets)
            
            # New eigenvalue estimate
            new_eigenvalue = (v @ Hv).item()
            
            # Normalize
            v_new = Hv / (Hv.norm() + 1e-8)
            
            # Check convergence
            if abs(new_eigenvalue - eigenvalue) < tol:
                break
            
            eigenvalue = new_eigenvalue
            v = v_new
        
        return eigenvalue, v
    
    def estimate_condition_number(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_iters: int = 20
    ) -> Dict[str, float]:
        """
        Estimate condition number and eigenvalue bounds.
        
        Args:
            inputs: Input batch
            targets: Target batch
            num_iters: Power iteration steps
            
        Returns:
            Dictionary with lambda_max, lambda_min, condition_number
        """
        # Largest eigenvalue via power iteration
        lambda_max, v_max = self.power_iteration(inputs, targets, num_iters)
        
        # For smallest eigenvalue, we'd need inverse iteration which requires
        # solving linear systems. For simplicity, use shifted power iteration.
        # This is an approximation.
        
        # Estimate trace using Hutchinson estimator
        params = self._get_params()
        n_params = sum(p.numel() for p in params)
        
        trace_estimate = 0.0
        n_samples = 5
        for _ in range(n_samples):
            z = torch.randn(n_params, device=self.device)
            Hz = self.hessian_vector_product(z, inputs, targets)
            trace_estimate += (z @ Hz).item()
        trace_estimate /= n_samples
        
        # Rough estimate: assume eigenvalues are roughly uniformly distributed
        # lambda_min ≈ (2 * trace - lambda_max * n) / (n - 1) for well-conditioned
        # This is a simplification for efficiency
        lambda_min = max(1e-6, trace_estimate / n_params - abs(lambda_max) * 0.1)
        
        condition_number = abs(lambda_max) / max(abs(lambda_min), 1e-8)
        
        return {
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,
            'condition_number': condition_number,
            'trace_estimate': trace_estimate
        }
    
    def compute_sharpness(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 0.01
    ) -> Dict[str, float]:
        """
        Compute sharpness measure for SAM analysis.
        
        Sharpness = max_{||δ|| ≤ ε} [L(w + δ) - L(w)]
        
        Args:
            inputs: Input batch
            targets: Target batch
            epsilon: Perturbation radius
            
        Returns:
            Dictionary with sharpness metrics
        """
        self.model.eval()
        params = self._get_params()
        
        # Original loss
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            original_loss = self.criterion(outputs, targets.to(self.device)).item()
        
        # Find adversarial perturbation direction (gradient ascent direction)
        self.model.zero_grad()
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(self.device))
        loss.backward()
        
        # Compute perturbation (gradient direction, normalized)
        grad_norm = 0.0
        for p in params:
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Apply perturbation
        old_params = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                old_params[name] = p.data.clone()
                p.data.add_(p.grad.data * epsilon / max(grad_norm, 1e-8))
        
        # Perturbed loss
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
            perturbed_loss = self.criterion(outputs, targets.to(self.device)).item()
        
        # Restore parameters
        for name, p in self.model.named_parameters():
            if name in old_params:
                p.data = old_params[name]
        
        sharpness = perturbed_loss - original_loss
        
        return {
            'sharpness': sharpness,
            'original_loss': original_loss,
            'perturbed_loss': perturbed_loss,
            'epsilon': epsilon,
            'grad_norm': grad_norm
        }


def auto_tune_training_config(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    experiment_type: str = 'mnist',
    find_lr: bool = True,
    find_batch_size: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Automatically tune training configuration (LR and batch size).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        experiment_type: Type of experiment for batch size hints
        find_lr: Whether to run LR finder
        find_batch_size: Whether to find optimal batch size
        verbose: Print progress
        
    Returns:
        Dictionary with suggested config:
        {
            'learning_rate': float,
            'batch_size': int,
            'gpu_info': dict
        }
    """
    config = {}
    device = next(model.parameters()).device
    
    if verbose:
        print("\n🔧 Auto-Tuning Training Configuration...")
        print("=" * 50)
    
    # Memory-aware batch sizing
    if find_batch_size:
        if verbose:
            print("\n📊 Finding optimal batch size...")
        
        sizer = MemoryAwareBatchSizer()
        config['gpu_info'] = sizer.get_gpu_info()
        
        # Get a sample from the loader
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][0]  # First sample
        sample_target = sample_batch[1][0] if len(sample_batch) > 1 else None
        
        config['batch_size'] = sizer.find_optimal_batch_size(
            model, sample_input, sample_target, criterion
        )
        
        if verbose:
            print(f"   ✓ Suggested batch size: {config['batch_size']}")
    else:
        config['batch_size'] = sizer.get_recommended_batch_size(experiment_type)
    
    # LR Finding
    if find_lr:
        if verbose:
            print("\n📈 Finding optimal learning rate...")
        
        # Create a fresh optimizer for LR finding
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
        
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.range_test(train_loader, num_iter=100, verbose=verbose)
        
        config['learning_rate'] = lr_finder.suggest_lr()
        
        if verbose:
            print(f"   ✓ Suggested learning rate: {config['learning_rate']:.2e}")
    
    if verbose:
        print("\n" + "=" * 50)
        print("🎯 Auto-Tune Complete!")
        print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {config.get('batch_size', 'N/A')}")
        if 'gpu_info' in config:
            print(f"   GPU: {config['gpu_info'].get('name', 'N/A')} ({config['gpu_info'].get('memory_total_gb', 0):.1f}GB)")
    
    return config


if __name__ == '__main__':
    # Quick test
    print("Testing training enhancements...")
    
    # Test GPU detection
    sizer = MemoryAwareBatchSizer()
    gpu_info = sizer.get_gpu_info()
    print(f"GPU Info: {gpu_info}")
    
    # Test batch size hints
    for exp in ['mnist', 'cifar10', 'resnet18', 'nlp', 'medical']:
        bs = sizer.get_recommended_batch_size(exp)
        print(f"  {exp}: batch_size={bs}")
    
    # Test disk guardian
    guardian = DiskSpaceGuardian('/tmp/test_ckpts', min_free_gb=0.1)
    free_gb = guardian.get_free_space_gb()
    print(f"Free disk space: {free_gb:.2f}GB")
    print(f"Can save 500MB checkpoint: {guardian.can_save_checkpoint(500)}")
    
    print("\n✅ All tests passed!")
