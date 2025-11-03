"""
Optuna-based Hyperparameter Tuning for MNIST

Demonstrates automated hyperparameter optimization using Optuna.
Tunes optimizer hyperparameters (lr, momentum, betas) for best test accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import optuna
from src.core.data_utils import get_mnist_loaders
from src.core.models import SimpleMLP
from src.core.optimizers import SGDMomentum, Adam
from src.core.optuna_tuner import OptunaHyperparameterTuner, suggest_optimizer_params
import argparse


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
        
        # Get data loaders
        train_loader, test_loader = get_mnist_loaders(batch_size=128, train_size=50000)
        
        # Create model
        model = SimpleMLP(input_size=784, hidden_size=256, output_size=10).to(device)
        
        # Create optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = Adam(
                lr=params['lr'],
                beta1=params['beta1'],
                beta2=params['beta2'],
                epsilon=params['epsilon']
            )
        elif optimizer_name.lower() == 'sgdmomentum':
            optimizer = SGDMomentum(
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
                model.zero_grad()
                loss.backward()
                
                # Update weights
                for param in model.parameters():
                    if param.grad is not None:
                        update = optimizer.step(param.grad.data.cpu().numpy())
                        param.data.add_(torch.from_numpy(update).to(device))
                
                epoch_loss += loss.item()
            
            # Report intermediate value for pruning
            avg_loss = epoch_loss / len(train_loader)
            trial.report(avg_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        
        return accuracy
    
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
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
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
    print("\n" + "="*80)
    print("Parameter Importance:")
    print("="*80)
    importance = tuner.get_importance()
    for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{param:20s}: {score:.4f}")
    
    print("\n" + "="*80)
    print("Best Configuration:")
    print("="*80)
    print(f"Optimizer: {args.optimizer}")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nTest Accuracy: {results['best_value']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
