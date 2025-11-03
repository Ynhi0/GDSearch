import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from run_nn_experiment import build_model_and_data, build_optimizer, set_seed
from loss_landscape import _random_direction_like, probe_loss_1d, probe_loss_2d


def train_quick(config):
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


def main():
    os.makedirs('plots', exist_ok=True)
    # Minimal MNIST MLP training to get a reasonable point
    config = {
        'dataset': 'MNIST',
        'model': 'mlp',
        'batch_size': 128,
        'epochs': 2,
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'seed': 123,
    }

    model, loader, criterion, device = train_quick(config)

    # 1D probe
    dir1 = _random_direction_like(model, seed=0).to(device)
    alphas = np.linspace(-1.0, 1.0, 41)
    a, losses_1d = probe_loss_1d(model, loader, criterion, device, dir1, alphas, max_batches=50)

    plt.figure(figsize=(6,4))
    plt.plot(a, losses_1d, 'k-')
    plt.xlabel('Alpha (direction 1)')
    plt.ylabel('Loss')
    plt.title('Loss Landscape 1D (around trained point)')
    plt.tight_layout()
    plt.savefig('plots/loss_landscape_1d_mnist.png', dpi=150)
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
    ax.set_title('Loss Landscape 2D (around trained point)')
    plt.tight_layout()
    plt.savefig('plots/loss_landscape_2d_surface_mnist.png', dpi=150)
    plt.close(fig)

    # Contour
    plt.figure(figsize=(6,5))
    cs = plt.contourf(A, B, Z, levels=30, cmap='viridis')
    plt.colorbar(cs, label='Loss')
    plt.xlabel('Alpha (dir1)')
    plt.ylabel('Beta (dir2)')
    plt.title('Loss Landscape 2D Contour (MNIST)')
    plt.tight_layout()
    plt.savefig('plots/loss_landscape_2d_contour_mnist.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
