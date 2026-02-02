"""
Training Module for GDSearch.

Handles model training loops, evaluation, and checkpoint management.
"""

import logging
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
               criterion: nn.Module, device: torch.device, 
               gradient_clipping: Optional[float] = None) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        gradient_clipping: Max gradient norm for clipping (None to disable)
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # LOGIC REVIEW FIX: Detect SAM optimizer to use closure-based training
    # SAM requires two forward passes: one at current point, one at adversarial point
    from src.core.pytorch_optimizers import SAMWrapper
    is_sam = isinstance(optimizer, SAMWrapper)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        if is_sam:
            # SAM requires closure for adversarial gradient computation
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply gradient clipping inside closure for SAM
                if gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                
                return loss
            
            # SAM step with closure
            loss = optimizer.step(closure)
            
            # Track metrics from final forward pass
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            # Standard optimizer (non-SAM)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for non-SAM optimizers
            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            optimizer.step()
            
            # Track metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Handle loss return type (SAM returns Optional[float], others return Tensor)
        if isinstance(loss, torch.Tensor):
            total_loss += loss.item()
        elif loss is not None:
            total_loss += float(loss)
        else:
            # Fallback for optimizers that return None
            total_loss += 0.0
        
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'samples_processed': total
    }


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on given data.
    
    Args:
        model: Neural network model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'samples_evaluated': total
    }


def train_model(model: nn.Module, train_loader: DataLoader, 
               val_loader: Optional[DataLoader], test_loader: DataLoader,
               optimizer: torch.optim.Optimizer, criterion: nn.Module,
               device: torch.device, epochs: int,
               early_stopping_patience: Optional[int] = None,
               checkpoint_callback: Optional[Callable] = None,
               log_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Complete training loop with validation and testing.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        test_loader: Test data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        epochs: Number of training epochs
        early_stopping_patience: Epochs to wait before early stopping (None to disable)
        checkpoint_callback: Function to call for checkpointing
        log_callback: Function to call for logging each epoch
    
    Returns:
        Dictionary with final results and training history
    """
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_metrics['train_loss'])
        history['train_accuracy'].append(train_metrics['train_accuracy'])
        
        # Validation (if available)
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            current_val_loss = val_metrics['loss']
            
            # Early stopping check
            if early_stopping_patience is not None:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    
                    # Save best checkpoint
                    if checkpoint_callback:
                        checkpoint_callback(model, optimizer, epoch, val_metrics)
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Logging
        if log_callback:
            log_data = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'train_accuracy': train_metrics['train_accuracy'],
            }
            if val_loader:
                log_data['val_loss'] = val_metrics['loss']
                log_data['val_accuracy'] = val_metrics['accuracy']
            
            log_callback(log_data)
    
    # Final test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    results = {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'epochs_trained': epoch,
        'best_epoch': best_epoch if val_loader else epoch,
        'history': history
    }
    
    return results


def check_divergence(history: Dict[str, list], window: int = 5,
                     threshold: float = 10.0) -> bool:
    """
    Check if training has diverged (loss exploding).
    
    Args:
        history: Training history dictionary
        window: Number of recent epochs to check
        threshold: Loss threshold to consider as diverged
    
    Returns:
        True if training appears to have diverged
    """
    if 'train_loss' not in history or len(history['train_loss']) < window:
        return False
    
    recent_losses = history['train_loss'][-window:]
    
    # Check for NaN or Inf
    if any(np.isnan(loss) or np.isinf(loss) for loss in recent_losses):
        return True
    
    # Check for exploding loss
    if any(loss > threshold for loss in recent_losses):
        return True
    
    # Check for rapid increase
    if len(recent_losses) >= 2:
        growth_rate = recent_losses[-1] / recent_losses[0] if recent_losses[0] > 0 else float('inf')
        if growth_rate > 100:  # 100x increase
            return True
    
    return False


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm across all model parameters.
    
    Args:
        model: Neural network model
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
