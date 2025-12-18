"""
Optuna-based Hyperparameter Tuning for MNIST

Demonstrates automated hyperparameter optimization using Optuna.
Tunes optimizer hyperparameters (lr, momentum, betas) for best test accuracy.
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
import argparse
import random
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_objective_function(optimizer_name='Adam', epochs=10, device='cpu'):
    """
    Create objective function for Optuna optimization.
    
    Args:
        optimizer_name: Name of optimizer to tune
        epochs: Number of training epochs per trial
        device: Device to train on
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial):
        """Objective function: train model and return validation accuracy."""
        
        # Suggest hyperparameters
        params = suggest_optimizer_params(trial, optimizer_name)
        
        # Get data loaders with validation split (NO TEST SET ACCESS)
        # Using 10% of training data for validation to prevent data leakage
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=128, 
            num_workers=2,
            seed=42,  # Fixed seed for reproducible splits
            val_split=0.1  # 10% validation split
        )
        
        # Create model
        model = SimpleMLP(input_size=784, hidden_size=256, output_size=10).to(device)
        
        # Create optimizer using PyTorch optimizers (not custom numpy-based ones)
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['lr'],
                betas=(params['beta1'], params['beta2']),
                eps=params['epsilon']
            )
        elif optimizer_name.lower() == 'sgdmomentum':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['lr'],
                momentum=params['momentum']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
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
    
    # Create objective function
    objective_fn = create_objective_function(
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        device=device
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
    
    # Run optimization
    results = tuner.optimize(
        n_trials=args.n_trials,
        show_progress_bar=True
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
    logging.info("Final TEST accuracy should be reported separately after retraining.")

if __name__ == '__main__':
    main()
