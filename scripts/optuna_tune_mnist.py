"""
Optuna-based Hyperparameter Tuning for MNIST

Demonstrates automated hyperparameter optimization using Optuna.
Tunes optimizer hyperparameters (lr, momentum, betas) for best VALIDATION accuracy.
Test set is held out and only used for final evaluation after selecting best hyperparameters.
"""

import sys
import logging
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import optuna
from src.core.data_utils import get_mnist_loaders
from src.core.models import SimpleMLP
from src.core.optuna_tuner import OptunaHyperparameterTuner, suggest_optimizer_params
from src.core.loader_validation import enforce_no_test_in_tuning
from src.core.optimizer_adapter import build_optimizer_for_tuning
import argparse
import random
import numpy as np
from typing import Any, Union, cast


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_objective_function(optimizer_name: str = 'Adam', epochs: int = 10, device: Union[str, torch.device] = 'cpu', seed: int = 42):
    """Create objective function for Optuna optimization.

    Accept both string device names and torch.device instances for flexibility.
    """
    """
    Create objective function for Optuna optimization.
    
    Args:
        optimizer_name: Name of optimizer to tune
        epochs: Number of training epochs per trial
        device: Device to train on
        seed: Random seed for reproducibility
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial, seed=seed):
        """Objective function: train model and return validation accuracy."""
        
        # Suggest hyperparameters
        params = suggest_optimizer_params(trial, optimizer_name)
        
        # Get data loaders with validation split (NO TEST SET ACCESS)
        # Using 10% of training data for validation to prevent data leakage
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=128, 
            num_workers=2,
            seed=seed,  # Use seed parameter passed to objective
            val_split=0.1  # 10% validation split
        )
        
        # CRITICAL: Enforce that we're using validation (not test) for tuning
        # This prevents test set leakage which would invalidate generalization claims
        val_loader.name = 'validation'
        enforce_no_test_in_tuning(val_loader)
        
        # Create model with correct parameter name: num_classes (not output_size)
        model = SimpleMLP(input_size=784, hidden_size=256, num_classes=10).to(device)
        
        # Use optimizer adapter to ensure consistency between tuning and experiments
        # NOTE: use_custom_wrappers=False for faster tuning with native PyTorch optimizers
        # The adapter ensures hyperparameters will transfer correctly to custom wrappers
        optimizer = build_optimizer_for_tuning(
            optimizer_name=optimizer_name,
            model=model,
            params=params,
            use_custom_wrappers=False  # Use native PyTorch for speed during tuning
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Update weights using PyTorch optimizer
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Report intermediate value for pruning
            avg_loss = epoch_loss / len(train_loader)
            trial.report(avg_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Evaluate on VALIDATION set (NOT TEST SET)
        # The test set must remain untouched during hyperparameter optimization
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100.0 * correct / max(1, total)
        
        return val_accuracy
    
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for MNIST')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                       choices=['Adam', 'SGDMomentum'],
                       help='Optimizer to tune')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Epochs per trial')
    parser.add_argument('--study-name', type=str, default='mnist_optimization',
                       help='Study name')
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random'],
                       help='Sampling algorithm')
    parser.add_argument('--pruner', type=str, default='median',
                       choices=['median', 'percentile', 'none'],
                       help='Pruning algorithm')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-results', type=str, default='results/optuna_results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")    
    # Set random seeds for full reproducibility
    set_seed(args.seed)
    
    # Prepare data loaders and create objective function
    # Create validation loader upfront and pass to tuner for strict checking
    from src.core.data_utils import get_mnist_loaders
    train_loader_template, val_loader_template, test_loader_template = get_mnist_loaders(
        batch_size=128,
        num_workers=2,
        seed=args.seed,
        val_split=0.1  # 10% validation split used during tuning
    )

    # Create objective function (it can still create its own per-trial loaders, but we pass val_loader for verification)
    objective_fn = create_objective_function(
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        device=device,
        seed=args.seed
    )

    # Create tuner
    pruner = None if args.pruner == 'none' else args.pruner
    tuner = OptunaHyperparameterTuner(
        objective_fn=objective_fn,
        direction="maximize",  # Maximize accuracy
        study_name=args.study_name,
        sampler=args.sampler,
        pruner=pruner,
        seed=args.seed
    )

    # Run optimization and provide val_loader + test_dataset for stricter checks
    results = tuner.optimize(
        n_trials=args.n_trials,
        show_progress_bar=True,
        val_loader=val_loader_template,
        test_dataset=getattr(test_loader_template, 'dataset', None),
        enforce_validation=True
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    tuner.save_results(args.save_results)
    
    # Print parameter importance
    logging.info("\n" + "="*80)
    logging.info("Parameter Importance:")
    logging.info("="*80)
    importance = tuner.get_importance()
    for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{param:20s}: {score:.4f}")
    
    logging.info("\n" + "="*80)
    logging.info("Best Configuration:")
    logging.info("="*80)
    logging.info(f"Optimizer: {args.optimizer}")
    for param, value in results['best_params'].items():
        logging.info(f"  {param}: {value}")
    logging.info(f"\nValidation Accuracy: {results['best_value']:.2f}%")
    logging.info("="*80)
    logging.info("\nNOTE: This is VALIDATION accuracy (used for tuning).")
    logging.info("Proceeding to retrain with best params on TRAIN+VAL for final test...")
    
    # Automated retrain on train+val, then evaluate on test
    logging.info("\n" + "="*80)
    logging.info("RETRAINING WITH BEST HYPERPARAMETERS ON TRAIN+VAL")
    logging.info("="*80)

    # Get train and val loaders (same split used during tuning) and combine them for final training
    from src.core.data_utils import get_mnist_loaders
    train_loader_final, val_loader_final, test_loader_final = get_mnist_loaders(
        batch_size=128,
        num_workers=2,
        seed=args.seed,
        val_split=0.1  # Use same validation fraction as during tuning
    )

    # SCIENTIFIC FIX: Combine train + val datasets with CONSISTENT transforms
    # PROBLEM: ConcatDataset([train.dataset, val.dataset]) mixes augmented (train) 
    # with non-augmented (val) data, creating a heterogeneous distribution.
    # SOLUTION: Get the underlying base dataset and create a combined subset with
    # training transforms applied to ALL samples.
    from torch.utils.data import Subset, DataLoader
    import torchvision.transforms as transforms
    
    # Get the base MNIST dataset (before splitting)
    from torchvision.datasets import MNIST
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full training set with consistent transforms
    full_train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transform)
    
    # Get the indices from train and val loaders
    train_indices = train_loader_final.dataset.indices if hasattr(train_loader_final.dataset, 'indices') else range(len(train_loader_final.dataset))
    val_indices = val_loader_final.dataset.indices if hasattr(val_loader_final.dataset, 'indices') else range(len(val_loader_final.dataset))
    
    # Combine indices
    combined_indices = list(train_indices) + list(val_indices)
    
    # Create combined dataset with training transforms applied to ALL samples
    combined_dataset = Subset(full_train_dataset, combined_indices)
    train_val_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = test_loader_final
    
    logging.info(f"Combined dataset size: {len(combined_dataset)} (train: {len(train_indices)}, val: {len(val_indices)})")
    logging.info("All samples use training transforms for consistency.")

    # Create model with best params (num_classes not output_size)
    from src.core.models import SimpleMLP
    final_model = SimpleMLP(input_size=784, hidden_size=256, num_classes=10).to(device)

    # Build optimizer with best params
    best_params = results['best_params'].copy()
    lr = best_params.pop('lr')

    # Use optimizer adapter for final retrain to ensure consistency
    best_params['lr'] = lr  # Restore lr to params dict
    final_optimizer = build_optimizer_for_tuning(
        optimizer_name=args.optimizer,
        model=final_model,
        params=best_params,
        use_custom_wrappers=False  # Keep using native PyTorch for consistency
    )

    # Train for same number of epochs
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()

    logging.info(f"Training for {args.epochs} epochs on combined train+val set...")
    for epoch in range(args.epochs):
        final_model.train()
        for batch_idx, (data, target) in enumerate(train_val_loader):
            data, target = data.to(device), target.to(device)
            final_optimizer.zero_grad()
            output = final_model(data)
            loss = criterion(output, target)
            loss.backward()
            final_optimizer.step()
    
    # Evaluate on test set (NEVER SEEN DURING TUNING)
    final_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # Defensive dataset length retrieval: some dataset objects don't have a static stub for __len__
    dataset_obj = getattr(test_loader, 'dataset', None)
    try:
        dataset_len = len(cast(Any, dataset_obj)) if dataset_obj is not None else 0
    except Exception:
        dataset_len = 0

    test_accuracy = 100. * correct / max(1, dataset_len)
    logging.info(f"\n{'='*80}")
    logging.info(f"FINAL TEST SET ACCURACY: {test_accuracy:.2f}%")
    logging.info(f"{'='*80}")
    logging.info("This is the accuracy to report in reports.")

if __name__ == '__main__':
    main()
