import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# AUDIT FIX: Robust imports that work from any location
try:
    from run_nn_experiment import build_model_and_data, build_optimizer, set_seed
    from loss_landscape import _random_direction_like, probe_loss_1d, probe_loss_2d
except ImportError:
    # Try absolute imports
    try:
        from src.experiments.run_nn_experiment import build_model_and_data, build_optimizer
        from src.core.training_utils import set_seed
        from src.visualization.loss_landscape import _random_direction_like, probe_loss_1d, probe_loss_2d
    except ImportError:
        # Add parent directories to path
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        sys.path.insert(0, str(repo_root))
        
        from src.experiments.run_nn_experiment import build_model_and_data, build_optimizer
        from src.core.training_utils import set_seed
        from src.visualization.loss_landscape import _random_direction_like, probe_loss_1d, probe_loss_2d


def train_quick(config):
    """
    Train a model quickly for visualization (snapshot mode).
    
    NOTE: This is a QUICK TRAINING SNAPSHOT for visualization purposes.
    For high-quality loss landscapes, use --load-checkpoint with a
    fully trained model checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.get('seed', 42))
    model, train_loader, test_loader = build_model_and_data(
        dataset=config['dataset'],
        model_name='SimpleMLP' if config['model'].lower() in ('mlp', 'simplemlp') else config['model'],
        batch_size=config.get('batch_size', 128),
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        optimizer_name=config.get('optimizer', 'AdamW'),
        model=model,
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.0),
        momentum=config.get('momentum', 0.0),
    )

    model.train()
    epochs = config.get('epochs', 2)
    print(f"Training quick snapshot model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    return model, test_loader, criterion, device


def load_checkpoint_model(checkpoint_path, config):
    """
    Load a fully trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
    
    Returns:
        model, loader, criterion, device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model architecture
    model, train_loader, test_loader = build_model_and_data(
        dataset=config['dataset'],
        model_name='SimpleMLP' if config['model'].lower() in ('mlp', 'simplemlp') else config['model'],
        batch_size=config.get('batch_size', 128),
        device=device,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✓ Loaded FINAL CHECKPOINT (epoch {checkpoint.get('epoch', 'unknown')})")
    
    criterion = nn.CrossEntropyLoss()
    return model, test_loader, criterion, device


def main():
    parser = argparse.ArgumentParser(description='Visualize loss landscape around a model')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint file (for final model visualization)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                       help='Dataset name (default: MNIST)')
    parser.add_argument('--model', type=str, default='mlp',
                       help='Model type (default: mlp)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (default: 123)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Minimal MNIST MLP training to get a reasonable point
    config = {
        'dataset': args.dataset,
        'model': args.model,
        'batch_size': 128,
        'epochs': 2,
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'seed': args.seed,
    }

    # Load model: either from checkpoint or train quick snapshot
    if args.load_checkpoint:
        model, loader, criterion, device = load_checkpoint_model(args.load_checkpoint, config)
        mode_label = "Final Checkpoint"
        filename_suffix = "_final"
    else:
        print("WARNING: Using quick training snapshot (2 epochs).")
        print("For high-quality plots, use --load-checkpoint with a fully trained model.")
        model, loader, criterion, device = train_quick(config)
        mode_label = "Quick Snapshot (2 epochs)"
        filename_suffix = "_snapshot"

    # 1D probe
    dir1 = _random_direction_like(model, seed=0).to(device)
    alphas = np.linspace(-1.0, 1.0, 41)
    a, losses_1d = probe_loss_1d(model, loader, criterion, device, dir1, alphas, max_batches=50)

    plt.figure(figsize=(6,4))
    plt.plot(a, losses_1d, 'k-')
    plt.xlabel('Alpha (direction 1)')
    plt.ylabel('Loss')
    plt.title(f'Loss Landscape 1D - {mode_label}')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/loss_landscape_1d_{args.dataset.lower()}{filename_suffix}.png', dpi=300)
    plt.close()

    # 2D probe
    dir2 = _random_direction_like(model, seed=1).to(device)
    alphas2 = np.linspace(-0.5, 0.5, 41)
    betas2 = np.linspace(-0.5, 0.5, 41)
    A, B, Z = probe_loss_2d(model, loader, criterion, device, dir1, dir2, alphas2, betas2, max_batches=30)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, Z, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel('Alpha (dir1)')
    ax.set_ylabel('Beta (dir2)')
    ax.set_zlabel('Loss')
    ax.set_title(f'Loss Landscape 2D - {mode_label}')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/loss_landscape_2d_surface_{args.dataset.lower()}{filename_suffix}.png', dpi=300)
    plt.close(fig)

    # Contour
    plt.figure(figsize=(6,5))
    cs = plt.contourf(A, B, Z, levels=30, cmap='viridis')
    plt.colorbar(cs, label='Loss')
    plt.xlabel('Alpha (dir1)')
    plt.ylabel('Beta (dir2)')
    plt.title(f'Loss Landscape 2D Contour - {mode_label}')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/loss_landscape_2d_contour_{args.dataset.lower()}{filename_suffix}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
