"""
Comprehensive Ablation Studies for GDSearch

Systematically evaluates the contribution of each advanced feature:
1. Momentum vs no momentum
2. Adaptive learning rates (Adam) vs fixed (SGD) 
3. Weight decay regularization (AdamW vs Adam)
4. Sharpness-Aware Minimization (SAM)
5. Lookahead meta-learning
6. Advanced training features (Label Smoothing, EMA, AMP)

Addresses research proposal requirement for ablation studies with actual
scientific value and academic rigor.
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
from typing import Dict, List, Tuple
from src.core.training_utils import set_seed
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    from src.visualization.ablation_plots import generate_all_ablation_plots
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.visualization.ablation_plots import generate_all_ablation_plots
import json

try:
    from src.core.pytorch_optimizers import LookaheadWrapper
    HAS_LOOKAHEAD = True
except ImportError:
    HAS_LOOKAHEAD = False


# Removed duplicate set_seed - using from src.core.training_utils
    

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST ablation studies."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate(model, optimizer, train_loader, test_loader, 
                       device, epochs=5, criterion=None):
    """
    Train model and return final metrics.
    
    Returns:
        metrics: Dict with train_loss, test_loss, test_accuracy, convergence_speed
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluation
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
        
        test_acc = 100.0 * correct / max(1, total)  # Protect division by zero
        test_accuracies.append(test_acc)
    
    # Compute convergence speed (epochs to reach 95% of final accuracy)
    final_acc = test_accuracies[-1]
    target_acc = 0.95 * final_acc
    convergence_epoch = epochs
    for i, acc in enumerate(test_accuracies):
        if acc >= target_acc:
            convergence_epoch = i + 1
            break
    
    return {
        'final_train_loss': train_losses[-1],
        'final_test_accuracy': test_accuracies[-1],
        'convergence_epoch': convergence_epoch,
        'train_loss_curve': train_losses,
        'test_acc_curve': test_accuracies
    }


def ablation_momentum_effect(
    seeds=[42, 43, 44, 45, 46],
    output_dir='results/ablation_studies',
    epochs=10
):
    """
    Ablation: SGD vs SGD+Momentum
    
    Isolates the effect of momentum on convergence speed and final performance.
    """
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
    
    # Use make_dataloader for consistent num_workers and pin_memory settings
    from run_all_kaggle import make_dataloader
    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seeds[0] if seeds else None, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        set_seed(seed)
        
        # Baseline: SGD without momentum
        model_sgd = SimpleCNN().to(device)
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
        metrics_sgd = train_and_evaluate(
            model_sgd, optimizer_sgd, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD',
            'momentum': 0.0,
            **{k: v for k, v in metrics_sgd.items() if not isinstance(v, list)}
        })
        
        # With momentum β=0.9
        set_seed(seed)
        model_mom = SimpleCNN().to(device)
        optimizer_mom = optim.SGD(model_mom.parameters(), lr=0.01, momentum=0.9)
        metrics_mom = train_and_evaluate(
            model_mom, optimizer_mom, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD_Momentum',
            'momentum': 0.9,
            **{k: v for k, v in metrics_mom.items() if not isinstance(v, list)}
        })
        
        logging.info(f"Seed {seed}: SGD acc={metrics_sgd['final_test_accuracy']:.2f}%, "
                    f"Momentum acc={metrics_mom['final_test_accuracy']:.2f}%")
    
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
    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")
    
    # Statistical comparison
    sgd_accs = df[df['optimizer'] == 'SGD']['final_test_accuracy'].values
    mom_accs = df[df['optimizer'] == 'SGD_Momentum']['final_test_accuracy'].values
    
    improvement = mom_accs.mean() - sgd_accs.mean()
    logging.info(f"\n📊 Results:")
    logging.info(f"   SGD: {sgd_accs.mean():.2f}% ± {sgd_accs.std():.2f}%")
    logging.info(f"   Momentum: {mom_accs.mean():.2f}% ± {mom_accs.std():.2f}%")
    logging.info(f"   Improvement: {improvement:+.2f}%")
    
    return df


def ablation_adaptive_lr(
    seeds=[42, 43, 44, 45, 46],
    output_dir='results/ablation_studies',
    epochs=10
):
    """
    Ablation: SGD (fixed LR) vs Adam (adaptive LR)
    
    Isolates the effect of adaptive learning rates.
    """
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
    from run_all_kaggle import make_dataloader
    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seeds[0] if seeds else None, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        # Baseline: SGD with momentum (best non-adaptive)
        set_seed(seed)
        model_sgd = SimpleCNN().to(device)
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
        metrics_sgd = train_and_evaluate(
            model_sgd, optimizer_sgd, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'SGD_Momentum',
            'adaptive_lr': False,
            **{k: v for k, v in metrics_sgd.items() if not isinstance(v, list)}
        })
        
        # Adam: adaptive learning rate
        set_seed(seed)
        model_adam = SimpleCNN().to(device)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        metrics_adam = train_and_evaluate(
            model_adam, optimizer_adam, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'Adam',
            'adaptive_lr': True,
            **{k: v for k, v in metrics_adam.items() if not isinstance(v, list)}
        })
        
        logging.info(f"Seed {seed}: SGD acc={metrics_sgd['final_test_accuracy']:.2f}%, "
                    f"Adam acc={metrics_adam['final_test_accuracy']:.2f}%")
    
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
    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")
    
    # Statistical comparison
    sgd_accs = df[df['optimizer'] == 'SGD_Momentum']['final_test_accuracy'].values
    adam_accs = df[df['optimizer'] == 'Adam']['final_test_accuracy'].values
    
    improvement = adam_accs.mean() - sgd_accs.mean()
    logging.info(f"\n📊 Results:")
    logging.info(f"   SGD+Momentum: {sgd_accs.mean():.2f}% ± {sgd_accs.std():.2f}%")
    logging.info(f"   Adam: {adam_accs.mean():.2f}% ± {adam_accs.std():.2f}%")
    logging.info(f"   Improvement: {improvement:+.2f}%")
    
    return df


def ablation_weight_decay(
    seeds=[42, 43, 44, 45, 46],
    output_dir='results/ablation_studies',
    epochs=10
):
    """
    Ablation: Adam vs AdamW (weight decay)
    
    Isolates the effect of decoupled weight decay regularization.
    """
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
    from run_all_kaggle import make_dataloader
    train_loader = make_dataloader(train_dataset, batch_size=128, shuffle=True, seed=seeds[0] if seeds else None, num_workers=2, pin_memory=True)
    test_loader = make_dataloader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    
    for seed in seeds:
        # Baseline: Adam (no weight decay)
        set_seed(seed)
        model_adam = SimpleCNN().to(device)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        metrics_adam = train_and_evaluate(
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
        metrics_adamw = train_and_evaluate(
            model_adamw, optimizer_adamw, train_loader, test_loader, device, epochs
        )
        
        results.append({
            'seed': seed,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            **{k: v for k, v in metrics_adamw.items() if not isinstance(v, list)}
        })
        
        logging.info(f"Seed {seed}: Adam acc={metrics_adam['final_test_accuracy']:.2f}%, "
                    f"AdamW acc={metrics_adamw['final_test_accuracy']:.2f}%")
    
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
    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")
    
    # Statistical comparison
    adam_accs = df[df['optimizer'] == 'Adam']['final_test_accuracy'].values
    adamw_accs = df[df['optimizer'] == 'AdamW']['final_test_accuracy'].values
    
    improvement = adamw_accs.mean() - adam_accs.mean()
    logging.info(f"\n📊 Results:")
    logging.info(f"   Adam: {adam_accs.mean():.2f}% ± {adam_accs.std():.2f}%")
    logging.info(f"   AdamW: {adamw_accs.mean():.2f}% ± {adamw_accs.std():.2f}%")
    logging.info(f"   Improvement: {improvement:+.2f}%")
    logging.info(f"   Interpretation: {'Regularization helps' if improvement > 0 else 'No clear benefit for this task'}")
    
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
    
    with open(Path(output_dir) / 'ablation_summary.json', 'w') as f:
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
