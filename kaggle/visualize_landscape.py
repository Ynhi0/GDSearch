#!/usr/bin/env python3
"""
Loss Landscape Visualization Tool
=================================
Creates 1D and 2D visualization of the loss landscape around a trained model.
Required: A trained checkpoint file (.pt)
"""

import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --- Định nghĩa lại Model SimpleMLP để load checkpoint (Copy từ run_mnist.py) ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# -----------------------------------------------------------------------------

def get_random_direction(model):
    """Generate a random direction vector with same dimension as model params."""
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Filter normalization (important for visualization scale)
        d = d * (p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction

def get_loss_value(model, loader, direction1, direction2, alpha, beta, device):
    """Calculate loss at model + alpha*dir1 + beta*dir2."""
    # Backup original weights
    orig_weights = [p.data.clone() for p in model.parameters()]

    # Perturb weights
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.data.add_(direction1[i] * alpha + direction2[i] * beta)

    # Compute loss
    model.eval()
    loss_sum = 0
    count = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * data.size(0)
            count += data.size(0)
            # Optimization: only use partial dataset for speed
            if count > 1000: break

    # Restore weights
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.data.copy_(orig_weights[i])

    return loss_sum / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--output-dir', type=str, default='plots_landscape')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print(f"Loading model from {args.ckpt}...")
    model = SimpleMLP().to(device)
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt_name = checkpoint.get('opt', 'Unknown')
        print(f"Model loaded. Optimizer used: {opt_name}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Prepare Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)

    # Generate Directions
    print("Generating random directions...")
    dir1 = get_random_direction(model)
    dir2 = get_random_direction(model)

    # Grid Search
    print("Computing loss surface (this takes time)...")
    steps = 21
    rang = 1.0 # Range of exploration
    alphas = np.linspace(-rang, rang, steps)
    betas = np.linspace(-rang, rang, steps)

    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            losses[i, j] = get_loss_value(model, loader, dir1, dir2, alpha, beta, device)
            print(f".", end="", flush=True)
        print()

    # Plotting
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    CS = plt.contour(alphas, betas, losses, levels=20, cmap='viridis')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(0, 0, 'rx', markersize=10, label='Converged Minimum') # Center is the model
    plt.title(f"Loss Landscape around Minimum ({opt_name})")
    plt.xlabel("Direction 1")
    plt.ylabel("Direction 2")
    plt.legend()

    save_path = f"{args.output_dir}/landscape_{opt_name}.png"
    plt.savefig(save_path)
    print(f"\n✅ Landscape plot saved to {save_path}")

if __name__ == "__main__":
    main()