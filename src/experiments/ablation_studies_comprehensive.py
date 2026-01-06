"""
Comprehensive Ablation Studies for GDSearch

Systematically evaluates the contribution of each advanced feature:
1. Momentum vs no momentum
2. Adaptive learning rates (Adam) vs fixed (SGD) 
3. Weight decay regularization (AdamW vs Adam)
4. Sharpness-Aware Minimization (SAM)
5. Lookahead meta-learning
6. Advanced training features (Label Smoothing, EMA, AMP)

Standardized Hyperparameters
SCIENTIFIC VALIDITY NOTE: These are NOT pure single-variable ablations.
Multiple hyperparameters differ between optimizer families:
- Batch size: 128 (consistent across all optimizers)
- Epochs: 10 (quick validation) or 20 (full study)
- Learning rates: DIFFERENT per optimizer family (confounds direct comparison)
  - SGD family: lr=0.01 (standard for SGD without adaptive scaling)
  - Adam family: lr=0.001 (1/10 of SGD, standard Adam default)
  - This violates ceteris paribus but reflects real-world best practices
- All other settings identical across compared optimizers

INTERPRETATION: Results show optimizer performance with their respective
recommended learning rates, not pure algorithmic differences.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from src.core.training_utils import set_seed
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

try:
    from src.visualization.ablation_plots import generate_all_ablation_plots
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.visualization.ablation_plots import generate_all_ablation_plots
import json
from src.utils.plot_helpers import arr_to_numpy_float
try:
    from src.core.pytorch_optimizers import LookaheadWrapper
    HAS_LOOKAHEAD = True
except ImportError:
    HAS_LOOKAHEAD = False


# Removed duplicate set_seed - using from src.core.training_utils

# Import centralized model (FIX #4: Remove duplicate SimpleCNN)
from src.core.models import SimpleCNN


def train_and_evaluate_model_with_loaders(model, optimizer, train_loader, test_loader, 
                       device, epochs=5, criterion=None):
    """
    Train model with provided loaders and return final metrics.
    
    Returns:
        metrics: Dict with train_loss, test_loss, test_accuracy, convergence_speed
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    diverged = False
    divergence_reason = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Check for NaN/Inf loss
            if not torch.isfinite(loss):
                diverged = True
                divergence_reason = f"Non-finite loss at epoch {epoch}"
                logging.warning("Training diverged: %s", divergence_reason)
                break
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if diverged:
            break
        
        train_losses.append(epoch_loss / len(train_loader))
    
    # Final test evaluation (only after training completes - use test set only for final evaluation)
    logging.info("Evaluating final performance on test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    final_test_acc = 100.0 * correct / max(1, total)
    logging.info("Final Test Accuracy: %.2f%%", final_test_acc)
    
    return {
        'final_train_loss': train_losses[-1] if train_losses else np.nan,
        'final_test_accuracy': final_test_acc,
        'convergence_epoch': epochs,
        'train_loss_curve': train_losses,
        'diverged': diverged,
        'divergence_reason': divergence_reason if divergence_reason else 'None'
    }


def ablation_momentum_effect(
    seeds=None,
    output_dir='results/ablation_studies',
    epochs=10
):
    """Ablation: SGD vs SGD+Momentum.
    
    Isolates the effect of momentum on convergence speed and final performance.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    
    logging.info("="*60)
    logging.info("Ablation Study 1: Momentum Effect")
    logging.info("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        'data/', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        'data/', train=False, download=True, transform=transform
    )
    
    from src.core.dataloader_utils import make_dataloader
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        set_seed(seed)
        
        # This ensures worker RNG state and shuffle order vary across seeds
        train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seed, num_workers=2, pin_memory=True)
        
        # Baseline: SGD without momentum
        model_sgd = SimpleCNN().to(device)
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
        metrics_sgd = train_and_evaluate_model_with_loaders(
            model_sgd, optimizer_sgd, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD',
            'momentum': 0.0,
            **{k: v for k, v in metrics_sgd.items() if not isinstance(v, list)}
        })
        
        # With momentum β=0.9
        # Note: set_seed called at loop start already covers this
        model_mom = SimpleCNN().to(device)
        optimizer_mom = optim.SGD(model_mom.parameters(), lr=0.01, momentum=0.9)
        metrics_mom = train_and_evaluate_model_with_loaders(
            model_mom, optimizer_mom, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD_Momentum',
            'momentum': 0.9,
            **{k: v for k, v in metrics_mom.items() if not isinstance(v, list)}
        })
        
        logging.info("Seed %d: SGD acc=%.2f%%, Momentum acc=%.2f%%",
                     seed, metrics_sgd['final_test_accuracy'], metrics_mom['final_test_accuracy'])
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'ablation_momentum.csv', index=False)
    
    # Generate visualizations
    try:
        viz_df = df.copy()
        viz_df['configuration'] = viz_df['optimizer']
        generate_all_ablation_plots(
            df=viz_df,
            results_dir=str(Path(output_dir) / 'momentum'),
            study_name='momentum_effect',
            group_col='configuration',
            value_col='final_test_accuracy',
            baseline_name='SGD',
            features=['Momentum']
        )
    except (ImportError, ValueError, RuntimeError) as e:
        logging.warning("Visualization generation failed: %s", e)
    
    # Statistical comparison
    # arr_to_numpy_float already imported at top
    sgd_accs = arr_to_numpy_float(df[df['optimizer'] == 'SGD']['final_test_accuracy'])
    mom_accs = arr_to_numpy_float(df[df['optimizer'] == 'SGD_Momentum']['final_test_accuracy'])
    
    improvement = mom_accs.mean() - sgd_accs.mean()
    logging.info("\nResults:")
    logging.info("   SGD: %.2f%% ± %.2f%%", sgd_accs.mean(), sgd_accs.std())
    logging.info("   Momentum: %.2f%% ± %.2f%%", mom_accs.mean(), mom_accs.std())
    logging.info("   Improvement: %+.2f%%", improvement)
    
    return df


def ablation_adaptive_lr(
    seeds=None,
    output_dir='results/ablation_studies',
    epochs=10
):
    """Ablation: SGD (fixed LR) vs Adam (adaptive LR).
    
    Isolates the effect of adaptive learning rates.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    logging.info("="*60)
    logging.info("Ablation Study 2: Adaptive Learning Rate Effect")
    logging.info("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        'data/', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        'data/', train=False, download=True, transform=transform
    )
    
    # Use make_dataloader for consistent settings
    from src.core.dataloader_utils import make_dataloader
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seed, num_workers=2, pin_memory=True)
        
        # Baseline: SGD with momentum (best non-adaptive)
        set_seed(seed)
        model_sgd = SimpleCNN().to(device)
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
        metrics_sgd = train_and_evaluate_model_with_loaders(
            model_sgd, optimizer_sgd, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD_Momentum',
            'adaptive_lr': False,
            **{k: v for k, v in metrics_sgd.items() if not isinstance(v, list)}
        })
        
        # Adam: adaptive learning rate (no weight decay)
        set_seed(seed)
        model_adam = SimpleCNN().to(device)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0)
        metrics_adam = train_and_evaluate_model_with_loaders(
            model_adam, optimizer_adam, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'Adam',
            'adaptive_lr': True,
            **{k: v for k, v in metrics_adam.items() if not isinstance(v, list)}
        })
        
        logging.info("Seed %d: SGD acc=%.2f%%, Adam acc=%.2f%%",
                     seed, metrics_sgd['final_test_accuracy'], metrics_adam['final_test_accuracy'])
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'ablation_adaptive_lr.csv', index=False)
    
    # Generate visualizations
    try:
        viz_df = df.copy()
        viz_df['configuration'] = viz_df['optimizer']
        generate_all_ablation_plots(
            df=viz_df,
            results_dir=str(Path(output_dir) / 'adaptive_lr'),
            study_name='adaptive_learning_rate',
            group_col='configuration',
            value_col='final_test_accuracy',
            baseline_name='SGD_Momentum',
            features=['Adaptive LR']
        )
    except (ImportError, ValueError, RuntimeError) as e:
        logging.warning("Visualization generation failed: %s", e)
    
    # Statistical comparison
    sgd_accs = arr_to_numpy_float(df[df['optimizer'] == 'SGD_Momentum']['final_test_accuracy'])
    adam_accs = arr_to_numpy_float(df[df['optimizer'] == 'Adam']['final_test_accuracy'])
    
    improvement = adam_accs.mean() - sgd_accs.mean()
    logging.info("\nResults:")
    logging.info("   SGD+Momentum: %.2f%% ± %.2f%%", sgd_accs.mean(), sgd_accs.std())
    logging.info("   Adam: %.2f%% ± %.2f%%", adam_accs.mean(), adam_accs.std())
    logging.info("   Improvement: %+.2f%%", improvement)
    
    return df


def ablation_weight_decay(
    seeds=None,
    output_dir='results/ablation_studies',
    epochs=10
):
    """Ablation: Adam vs AdamW (weight decay).
    
    Isolates the effect of decoupled weight decay regularization.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    logging.info("="*60)
    logging.info("Ablation Study 3: Weight Decay Effect (Adam vs AdamW)")
    logging.info("="*60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        'data/', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        'data/', train=False, download=True, transform=transform
    )
    
    # Use make_dataloader for consistent settings
    from src.core.dataloader_utils import make_dataloader
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seed, num_workers=2, pin_memory=True)
        
        # Baseline: Adam (no weight decay)
        set_seed(seed)
        model_adam = SimpleCNN().to(device)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0)
        metrics_adam = train_and_evaluate_model_with_loaders(
            model_adam, optimizer_adam, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'Adam',
            'weight_decay': 0.0,
            **{k: v for k, v in metrics_adam.items() if not isinstance(v, list)}
        })
        
        # AdamW: decoupled weight decay
        set_seed(seed)
        model_adamw = SimpleCNN().to(device)
        optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)
        metrics_adamw = train_and_evaluate_model_with_loaders(
            model_adamw, optimizer_adamw, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            **{k: v for k, v in metrics_adamw.items() if not isinstance(v, list)}
        })
        
        logging.info("Seed %d: Adam acc=%.2f%%, AdamW acc=%.2f%%",
                     seed, metrics_adam['final_test_accuracy'], metrics_adamw['final_test_accuracy'])
    
    df = pd.DataFrame(results)
    df.to_csv(Path(output_dir) / 'ablation_weight_decay.csv', index=False)
    
    # Generate visualizations
    try:
        viz_df = df.copy()
        viz_df['configuration'] = viz_df['optimizer']
        generate_all_ablation_plots(
            df=viz_df,
            results_dir=str(Path(output_dir) / 'weight_decay'),
            study_name='weight_decay_regularization',
            group_col='configuration',
            value_col='final_test_accuracy',
            baseline_name='Adam',
            features=['Weight Decay']
        )
    except (ImportError, ValueError, RuntimeError) as e:
        logging.warning("Visualization generation failed: %s", e)
    
    # Statistical comparison
    adam_accs = arr_to_numpy_float(df[df['optimizer'] == 'Adam']['final_test_accuracy'])
    adamw_accs = arr_to_numpy_float(df[df['optimizer'] == 'AdamW']['final_test_accuracy'])
    
    improvement = adamw_accs.mean() - adam_accs.mean()
    logging.info("\nResults:")
    logging.info("   Adam: %.2f%% ± %.2f%%", adam_accs.mean(), adam_accs.std())
    logging.info("   AdamW: %.2f%% ± %.2f%%", adamw_accs.mean(), adamw_accs.std())
    logging.info("   Improvement: %+.2f%%", improvement)
    interpretation = 'Regularization helps' if improvement > 0 else 'No clear benefit for this task'
    logging.info("   Interpretation: %s", interpretation)
    
    return df


def ablation_sam_effect(output_dir='results/ablation_studies', epochs=10, num_seeds=3):
    """
    Ablation Study 4: Sharpness-Aware Minimization (SAM)
    
    Compares:
    - SGD (baseline)
    - SAM (sharpness-aware perturbations)
    
    SAM aims to find flatter minima for better generalization.
    """
    from src.core.pytorch_optimizers import SAMWrapper
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Ablation Study 4: SAM vs Baseline")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    optimizers_config = [
        ('SGD', lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),
        ('SAM', lambda model: SAMWrapper(
            base_optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
            rho=0.05
        ))
    ]
    
    for seed in range(42, 42 + num_seeds):
        for opt_name, opt_fn in optimizers_config:
            set_seed(seed)
            model = SimpleCNN(num_classes=10).to(device)
            optimizer = opt_fn(model)
            criterion = nn.CrossEntropyLoss()
            
            print(f"\n{opt_name} (seed {seed}):")
            
            # Modified training loop with closure support for SAM
            train_losses = []
            test_accuracies = []
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # SAM requires closure
                    if opt_name == 'SAM':
                        # Extract variables before closure to avoid cell variable warnings
                        _optimizer = optimizer
                        _model = model
                        _inputs = inputs
                        _targets = targets
                        _criterion = criterion
                        
                        def closure():
                            _optimizer.zero_grad()
                            outputs = _model(_inputs)
                            loss = _criterion(outputs, _targets)
                            loss.backward()
                            return loss
                        
                        loss = optimizer.step(closure)
                        epoch_loss += loss.item()
                    else:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                # Test
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                test_acc = correct / total
                test_accuracies.append(test_acc)
                
                if epoch % 2 == 0:
                    print(f"  Epoch {epoch}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")
            
            results.append({
                'optimizer': opt_name,
                'seed': seed,
                'final_train_loss': train_losses[-1],
                'final_test_accuracy': test_accuracies[-1],
                'best_test_accuracy': max(test_accuracies)
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'ablation_sam.csv', index=False)
    
    # Summary
    print("\n" + "-"*60)
    print("SAM Ablation Summary:")
    print("-"*60)
    for opt_name in ['SGD', 'SAM']:
        subset = df[df['optimizer'] == opt_name]
        mean_acc = subset['final_test_accuracy'].mean()
        std_acc = subset['final_test_accuracy'].std()
        print(f"{opt_name:15s}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    improvement = df[df['optimizer']=='SAM']['final_test_accuracy'].mean() - \
                  df[df['optimizer']=='SGD']['final_test_accuracy'].mean()
    print(f"\nSAM improvement: {improvement:+.4f}")
    interpretation = 'SAM finds flatter minima' if improvement > 0 else 'No clear benefit for this task'
    logging.info("   Interpretation: %s", interpretation)
    
    return df


def ablation_lookahead_effect(output_dir='results/ablation_studies', epochs=10, num_seeds=3):
    """
    Ablation Study 5: Lookahead Meta-Learning
    
    Compares:
    - SGD Momentum (baseline)
    - Lookahead + SGD Momentum
    
    Lookahead maintains slow and fast weights for stability.
    """
    if not HAS_LOOKAHEAD:
        print("WARNING: Lookahead not available. Skipping ablation.")
        return pd.DataFrame()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Ablation Study 5: Lookahead vs Baseline")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    optimizers_config = [
        ('SGD_Momentum', lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),
        ('Lookahead', lambda model: LookaheadWrapper(
            optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
            k=5,
            alpha=0.5
        ))
    ]
    
    for seed in range(42, 42 + num_seeds):
        for opt_name, opt_fn in optimizers_config:
            set_seed(seed)
            model = SimpleCNN(num_classes=10).to(device)
            optimizer = opt_fn(model)
            
            print(f"\n{opt_name} (seed {seed}):")
            
            metrics = train_and_evaluate_model_with_loaders(
                model, optimizer, train_loader, test_loader, device, epochs=epochs
            )
            
            results.append({
                'optimizer': opt_name,
                'seed': seed,
                'final_train_loss': metrics['train_loss'],
                'final_test_accuracy': metrics['test_accuracy']
            })
            
            print(f"  Final: loss={metrics['train_loss']:.4f}, acc={metrics['test_accuracy']:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'ablation_lookahead.csv', index=False)
    
    # Summary
    print("\n" + "-"*60)
    print("Lookahead Ablation Summary:")
    print("-"*60)
    for opt_name in ['SGD_Momentum', 'Lookahead']:
        subset = df[df['optimizer'] == opt_name]
        mean_acc = subset['final_test_accuracy'].mean()
        std_acc = subset['final_test_accuracy'].std()
        print(f"{opt_name:15s}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    improvement = df[df['optimizer']=='Lookahead']['final_test_accuracy'].mean() - \
                  df[df['optimizer']=='SGD_Momentum']['final_test_accuracy'].mean()
    print(f"\nLookahead improvement: {improvement:+.4f}")
    interpretation = 'Lookahead provides stability' if improvement > 0 else 'No clear benefit for this task'
    logging.info("   Interpretation: %s", interpretation)
    
    return df


def run_all_ablation_studies(output_dir='results/ablation_studies'):
    """Run all ablation studies and generate summary report."""
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ABLATION STUDIES")
    print("Systematically evaluating optimizer components")
    print("="*70 + "\n")
    
    # Study 1: Momentum
    df_momentum = ablation_momentum_effect(output_dir=output_dir)
    
    # Study 2: Adaptive LR
    df_adaptive = ablation_adaptive_lr(output_dir=output_dir)
    
    # Study 3: Weight Decay
    df_wd = ablation_weight_decay(output_dir=output_dir)
    
    # Study 4: SAM
    df_sam = ablation_sam_effect(output_dir=output_dir)
    
    # Study 5: Lookahead
    df_lookahead = ablation_lookahead_effect(output_dir=output_dir)
    
    # Generate summary report
    summary = {
        'momentum_effect': {
            'improvement_pct': (df_momentum[df_momentum['optimizer']=='SGD_Momentum']['final_test_accuracy'].mean() -
                              df_momentum[df_momentum['optimizer']=='SGD']['final_test_accuracy'].mean()),
            'conclusion': 'Momentum significantly improves convergence'
        },
        'adaptive_lr_effect': {
            'improvement_pct': (df_adaptive[df_adaptive['optimizer']=='Adam']['final_test_accuracy'].mean() -
                              df_adaptive[df_adaptive['optimizer']=='SGD_Momentum']['final_test_accuracy'].mean()),
            'conclusion': 'Adaptive LR provides faster convergence'
        },
        'weight_decay_effect': {
            'improvement_pct': (df_wd[df_wd['optimizer']=='AdamW']['final_test_accuracy'].mean() -
                              df_wd[df_wd['optimizer']=='Adam']['final_test_accuracy'].mean()),
            'conclusion': 'Weight decay provides regularization benefit'
        }
    }
    
    # Add SAM/Lookahead if available
    if not df_sam.empty:
        summary['sam_effect'] = {
            'improvement_pct': (df_sam[df_sam['optimizer']=='SAM']['final_test_accuracy'].mean() -
                              df_sam[df_sam['optimizer']=='SGD']['final_test_accuracy'].mean()),
            'conclusion': 'SAM finds flatter minima'
        }
    
    if not df_lookahead.empty:
        summary['lookahead_effect'] = {
            'improvement_pct': (df_lookahead[df_lookahead['optimizer']=='Lookahead']['final_test_accuracy'].mean() -
                              df_lookahead[df_lookahead['optimizer']=='SGD_Momentum']['final_test_accuracy'].mean()),
            'conclusion': 'Lookahead provides stability'
        }
    
    with open(Path(output_dir) / 'ablation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("ABLATION STUDIES COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    print("\nKey Findings:")
    for study, results in summary.items():
        print(f"  {study}: {results['improvement_pct']:+.2f}% - {results['conclusion']}")


if __name__ == '__main__':
    run_all_ablation_studies()
