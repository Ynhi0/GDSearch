"""
Standardized training loops to eliminate code duplication.

This module provides unified training loop implementations that are reused
across all experiments (MNIST, CIFAR-10, NLP, Medical Segmentation).

Key Features:
- DRY principle: Single implementation, multiple callers
- Consistent metrics computation
- Built-in checkpointing support
- Robust error handling
- Validation-based early stopping
- Gradient norm tracking for convergence analysis

Example:
    >>> from src.experiments.training_loops import standard_classification_loop
    >>> results = standard_classification_loop(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     optimizer=optimizer,
    ...     criterion=criterion,
    ...     device=device,
    ...     epochs=50,
    ...     checkpoint_manager=checkpoint_manager
    ... )
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, Callable, List
import logging
import time
import random  # BUG FIX #2: For RNG state saving
import numpy as np  # BUG FIX #2: For RNG state saving
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training loops."""
    epochs: int
    device: torch.device
    patience: int = 10
    grad_clip_norm: Optional[float] = None
    use_amp: bool = False
    log_interval: int = 10
    compute_grad_noise_every: int = 0
    grad_noise_samples: int = 10
    checkpoint_every: int = 1
    min_train_acc_sanity: float = 10.0  # Minimum expected train accuracy (sanity check)
    
    
@dataclass
class TrainingResults:
    """Results from a training run."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_val_loss: float = float('inf')
    best_model_state: Optional[Dict[str, Any]] = None
    final_test_acc: float = 0.0
    final_test_loss: float = 0.0
    total_training_time: float = 0.0
    early_stopped_at_epoch: Optional[int] = None
    run_tainted: bool = False
    effective_batch_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Exclude large state dict from dict representation
        if 'best_model_state' in result:
            result['best_model_state'] = result['best_model_state'] is not None
        return result


def standard_classification_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    scheduler: Optional[_LRScheduler] = None,
    test_loader: Optional[DataLoader] = None,
    checkpoint_manager: Optional[Any] = None,
    experiment_tracker: Optional[Any] = None,
    robust_grad_handler: Optional[Any] = None,
    metrics_callback: Optional[Callable[[int, Dict], None]] = None,
    optimizer_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> TrainingResults:
    """
    Standard classification training loop with comprehensive monitoring.
    
    This function eliminates ~1000 lines of duplicated training logic across
    run_all_kaggle.py and provides a unified, tested implementation.
    
    Args:
        model: Neural network to train
        train_loader: Training data loader
        val_loader: Validation data loader (for early stopping)
        optimizer: Optimizer instance
        criterion: Loss function
        config: Training configuration
        scheduler: Optional learning rate scheduler
        test_loader: Optional test loader for final evaluation
        checkpoint_manager: Optional checkpoint manager for saving
        experiment_tracker: Optional MLflow/experiment tracker
        robust_grad_handler: Optional robust gradient processing handler
        metrics_callback: Optional callback(epoch, metrics_dict)
        optimizer_name: Optimizer name for logging
        seed: Random seed for this run
        
    Returns:
        TrainingResults containing history, best model, and metrics
        
    Note:
        - Uses validation set for early stopping (never touches test set during training)
        - Computes gradient norms for convergence analysis if requested
        - Handles OOM recovery if robust_grad_handler provided
        - Saves checkpoints if checkpoint_manager provided
    """
    results = TrainingResults()
    training_start_time = time.time()
    patience_counter = 0
    start_epoch = 1
    
    # Import OOM-safe training step if available
    try:
        from src.core.oom_handler import oom_safe_train_step
    except ImportError:
        logging.warning("OOM handler not available - using standard training step")
        oom_safe_train_step = None
    
    # Get dataset sizes for metric computation
    train_dataset_size = len(getattr(train_loader, 'dataset', []))
    val_dataset_size = len(getattr(val_loader, 'dataset', []))
    
    logging.info(
        f"Starting training: {config.epochs} epochs, "
        f"train_size={train_dataset_size}, val_size={val_dataset_size}"
    )
    
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()
        
        # ============ Training Phase ============
        model.train()
        train_loss, train_correct = 0.0, 0
        train_total_samples = 0
        batch_count = 0
        
        for inputs, targets in train_loader:
            batch_count += 1
            current_batch_size = inputs.size(0)
            
            if oom_safe_train_step is not None:
                # Use OOM-safe training step with recovery
                try:
                    loss_value, actual_batch_size, outputs, batch_tainted = oom_safe_train_step(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        inputs=inputs,
                        targets=targets,
                        device=config.device,
                        max_retries=3,
                        min_batch_size=1,
                        robust_grad_handler=robust_grad_handler,
                        epoch=epoch
                    )
                    
                    if batch_tainted:
                        results.run_tainted = True
                        results.effective_batch_size = actual_batch_size
                    
                    # BUG FIX: Weight loss by actual batch size for correct averaging
                    train_loss += loss_value * actual_batch_size
                    _, predicted = outputs.max(1)
                    train_correct += predicted.eq(targets.to(config.device)).sum().item()
                    train_total_samples += actual_batch_size
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logging.error(f"OOM Error (unrecoverable) at epoch {epoch}: {e}")
                        results.run_tainted = True
                        results.effective_batch_size = 1
                        break
                    else:
                        raise
            else:
                # Standard training step
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Optional gradient clipping
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                
                optimizer.step()
                
                # BUG FIX: Weight loss by batch size for correct averaging
                train_loss += loss.item() * current_batch_size
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(targets).sum().item()
                train_total_samples += current_batch_size
        
        # Compute epoch metrics
        # BUG FIX: Divide by total samples, not number of batches
        train_loss /= max(1, train_total_samples)
        train_acc = 100.0 * train_correct / max(1, train_total_samples)
        
        # ============ Compute Gradient Norm (BEFORE validation clears gradients) ============
        # BUG FIX: Must compute gradient norm here while training gradients still exist
        grad_norm = 0.0
        if epoch % config.log_interval == 0:
            try:
                from src.core.training_utils import compute_gradient_norm
                grad_norm = float(compute_gradient_norm(model))
            except (ImportError, AttributeError):
                # Fallback computation with explicit no-gradient handling
                has_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        has_grad = True
                        grad_norm += param.grad.data.norm(2).item() ** 2
                
                if has_grad:
                    grad_norm = grad_norm ** 0.5
                else:
                    # No gradients available - explicit 0.0
                    grad_norm = 0.0
        
        # Sanity check: training should be progressing
        if epoch > 2 and train_acc < config.min_train_acc_sanity:
            logging.error(
                f"SANITY CHECK FAILED: Train accuracy {train_acc:.1f}% is suspiciously low at epoch {epoch}. "
                f"Expected at least {config.min_train_acc_sanity}%. "
                f"This may indicate a bug in the training loop."
            )
        
        # ============ Validation Phase ============
        model.eval()
        val_loss, val_correct = 0.0, 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                batch_size = inputs.size(0)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # BUG FIX: Weight loss by batch size for correct averaging
                val_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total_samples += batch_size
        
        # BUG FIX: Divide by total samples, not number of batches
        val_loss /= max(1, val_total_samples)
        val_acc = 100.0 * val_correct / max(1, val_total_samples)
        
        # Optional gradient noise variance estimation
        grad_noise_var = None
        if config.compute_grad_noise_every > 0 and epoch % config.compute_grad_noise_every == 0:
            try:
                from src.analysis.gradient_noise_analysis import estimate_gradient_noise_variance
                noise_stats = estimate_gradient_noise_variance(
                    model=model,
                    data_loader=train_loader,
                    criterion=criterion,
                    device=config.device,
                    num_samples=config.grad_noise_samples,
                    method='empirical_variance'
                )
                grad_noise_var = noise_stats.get('sigma_squared')
                if grad_noise_var is not None:
                    logging.info(f"Epoch {epoch}: Gradient noise σ² = {grad_noise_var:.4e}")
            except Exception as e:
                logging.debug(f"Could not compute gradient noise: {e}")
        
        # ============ Learning Rate Scheduling ============
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # ============ Best Model Tracking ============
        if val_acc > results.best_val_acc:
            results.best_val_acc = val_acc
            results.best_val_loss = val_loss
            # Deep copy state dict to CPU to avoid GPU memory accumulation
            results.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logging.info(f"✓ New best model at epoch {epoch}: val_acc={val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # ============ Record Epoch History ============
        elapsed_seconds = time.time() - training_start_time
        epoch_duration = time.time() - epoch_start_time
        
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'grad_norm': grad_norm,
            'elapsed_seconds': elapsed_seconds,
            'epoch_duration': epoch_duration,
        }
        
        # Add optional metrics
        if grad_noise_var is not None:
            epoch_metrics['grad_noise_var'] = grad_noise_var
        if results.run_tainted:
            epoch_metrics['tainted'] = True
            epoch_metrics['effective_batch_size'] = results.effective_batch_size
        
        results.history.append(epoch_metrics)
        
        # ============ Logging & Callbacks ============
        if epoch % config.log_interval == 0 or epoch == config.epochs:
            log_msg = (
                f"Epoch {epoch}/{config.epochs} - "
                f"Train: Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                f"Val: Loss={val_loss:.4f} Acc={val_acc:.2f}% | "
                f"LR={current_lr:.6f}"
            )
            logging.info(log_msg)
        
        # Optional experiment tracking (MLflow, etc.)
        if experiment_tracker is not None:
            try:
                tracker_metrics = {
                    f'{optimizer_name}_train_loss': train_loss,
                    f'{optimizer_name}_train_acc': train_acc,
                    f'{optimizer_name}_val_loss': val_loss,
                    f'{optimizer_name}_val_acc': val_acc,
                }
                experiment_tracker.log_metrics(tracker_metrics, step=epoch)
            except Exception as e:
                logging.debug(f"Could not log to experiment tracker: {e}")
        
        # Optional custom metrics callback
        if metrics_callback is not None:
            try:
                metrics_callback(epoch, epoch_metrics)
            except Exception as e:
                logging.warning(f"Metrics callback failed: {e}")
        
        # ============ Checkpointing ============
        if checkpoint_manager is not None and epoch % config.checkpoint_every == 0:
            try:
                # BUG FIX #2: Save RNG states for deterministic resume
                # Without this, Train(10) ≠ Train(5)→Resume→Train(5) due to different batch orderings
                rng_state = {
                    'torch_rng_state': torch.random.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate(),
                }
                if torch.cuda.is_available():
                    rng_state['cuda_rng_state'] = torch.cuda.get_rng_state()
                    rng_state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
                
                checkpoint_data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': epoch,
                    'history': results.history,
                    'best_val_acc': results.best_val_acc,
                    'rng_state': rng_state,  # ← FIX: Save all RNG states
                    'metadata': {
                        'optimizer_name': optimizer_name,
                        'seed': seed,
                        'current_lr': current_lr,
                        'patience_counter': patience_counter,
                        'training_time_sec': elapsed_seconds,
                    }
                }
                checkpoint_manager.save(
                    checkpoint_data,
                    epoch=epoch,
                    optimizer_name=optimizer_name or 'unknown',
                    seed=seed
                )
            except Exception as e:
                logging.warning(f"Failed to save checkpoint: {e}")
        
        # ============ Early Stopping ============
        if patience_counter >= config.patience:
            logging.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {config.patience} epochs)"
            )
            results.early_stopped_at_epoch = epoch
            # Restore best model
            if results.best_model_state is not None:
                model.load_state_dict(results.best_model_state)
            break
    
    # ============ Final Test Evaluation (if test set provided) ============
    if test_loader is not None:
        logging.info("Running final evaluation on test set...")
        model.eval()
        test_loss, test_correct = 0.0, 0
        test_total_samples = 0
        test_dataset_size = len(getattr(test_loader, 'dataset', []))
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                batch_size = inputs.size(0)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # BUG FIX: Weight loss by batch size for correct averaging
                test_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total_samples += batch_size
        
        # BUG FIX: Divide by total samples, not number of batches
        results.final_test_loss = test_loss / max(1, test_total_samples)
        results.final_test_acc = 100.0 * test_correct / max(1, test_dataset_size)
        
        logging.info(
            f"Final Test Results: Loss={results.final_test_loss:.4f}, "
            f"Acc={results.final_test_acc:.2f}%"
        )
    
    results.total_training_time = time.time() - training_start_time
    logging.info(f"Training complete in {results.total_training_time:.1f} seconds")
    
    return results


def standard_segmentation_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: Callable,
    config: TrainingConfig,
    *,
    scheduler: Optional[_LRScheduler] = None,
    test_loader: Optional[DataLoader] = None,
    checkpoint_manager: Optional[Any] = None,
    experiment_tracker: Optional[Any] = None,
    metric_fn: Optional[Callable] = None,
    optimizer_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> TrainingResults:
    """
    Standard segmentation training loop (e.g., U-Net for medical imaging).
    
    Similar to classification loop but uses segmentation-specific metrics
    like Dice coefficient instead of accuracy.
    
    Args:
        model: Segmentation model (e.g., U-Net)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        criterion: Segmentation loss function (e.g., Dice loss)
        config: Training configuration
        scheduler: Optional learning rate scheduler
        test_loader: Optional test loader for final evaluation
        checkpoint_manager: Optional checkpoint manager
        experiment_tracker: Optional MLflow tracker
        metric_fn: Segmentation metric function (default: Dice coefficient)
        optimizer_name: Optimizer name for logging
        seed: Random seed
        
    Returns:
        TrainingResults with segmentation metrics
    """
    # Import segmentation-specific metric
    if metric_fn is None:
        try:
            from run_all_kaggle import dice_coefficient
            metric_fn = dice_coefficient
        except ImportError:
            logging.warning("Could not import dice_coefficient - metrics will be limited")
            metric_fn = lambda pred, target: 0.0
    
    results = TrainingResults()
    training_start_time = time.time()
    patience_counter = 0
    best_dice = 0.0
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            optimizer.step()
            
            # BUG FIX: Weight loss by batch size for correct averaging
            train_loss += loss.item() * batch_size
            
            # Compute Dice coefficient
            with torch.no_grad():
                dice = metric_fn(torch.sigmoid(outputs), targets)
                train_dice += dice.item()
            
            train_total_samples += batch_size
        
        # BUG FIX: Divide by total samples, not number of batches
        train_loss /= max(1, train_total_samples)
        train_dice /= max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                batch_size = inputs.size(0)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # BUG FIX: Weight loss by batch size for correct averaging
                val_loss += loss.item() * batch_size
                
                dice = metric_fn(torch.sigmoid(outputs), targets)
                val_dice += dice.item()
                
                val_total_samples += batch_size
        
        # BUG FIX: Divide by total samples, not number of batches
        val_loss /= max(1, val_total_samples)
        val_dice /= max(1, len(val_loader))
        
        # LR scheduling
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Best model tracking (use Dice for segmentation)
        if val_dice > best_dice:
            best_dice = val_dice
            results.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logging.info(f"✓ New best model: Dice={val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Record history
        elapsed_seconds = time.time() - training_start_time
        results.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'learning_rate': current_lr,
            'elapsed_seconds': elapsed_seconds,
        })
        
        if epoch % config.log_interval == 0:
            logging.info(
                f"Epoch {epoch}/{config.epochs} - "
                f"Train: Loss={train_loss:.4f} Dice={train_dice:.4f} | "
                f"Val: Loss={val_loss:.4f} Dice={val_dice:.4f}"
            )
        
        # Early stopping
        if patience_counter >= config.patience:
            results.early_stopped_at_epoch = epoch
            if results.best_model_state is not None:
                model.load_state_dict(results.best_model_state)
            break
    
    results.total_training_time = time.time() - training_start_time
    results.best_val_acc = best_dice  # Store best Dice as "accuracy" equivalent
    
    return results
