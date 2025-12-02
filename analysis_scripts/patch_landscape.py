
#!/usr/bin/env python3
import argparse
import copy
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --- Fallback Models ---
class FallbackSimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FallbackResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(FallbackResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_random_direction(model):
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if p.dim() > 1:
             d_norm = d.view(d.size(0), -1).norm(dim=1).view(-1, *([1]*(p.dim()-1)))
             p_norm = p.view(p.size(0), -1).norm(dim=1).view(-1, *([1]*(p.dim()-1)))
             d = d * p_norm / (d_norm + 1e-10)
        else:
             d = d * p.norm() / (d.norm() + 1e-10)
        direction.append(d)
    return direction

def compute_hessian_eigenvalue(model, loader, criterion, device, max_iter=50, tol=1e-2):
    model.eval()
    try:
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(device), targets.to(device)
    except StopIteration:
        return 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
    v = [vi / v_norm for vi in v]
    eigenvalue = 0.0
    for i in range(max_iter):
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv_product = sum(torch.sum(gi * vi) for gi, vi in zip(grads, v))
        Hv = torch.autograd.grad(gv_product, params, retain_graph=True)
        new_eigenvalue = sum(torch.sum(vi * hvi) for vi, hvi in zip(v, Hv)).item()
        Hv_norm = torch.sqrt(sum(torch.sum(hvi**2) for hvi in Hv))
        if Hv_norm == 0: break
        v = [hvi / Hv_norm for hvi in Hv]
        if abs(new_eigenvalue - eigenvalue) < tol: break
        eigenvalue = new_eigenvalue
    return eigenvalue

def compute_loss_surface(model, loader, dir1, dir2, range_val=1.0, steps=21, device='cpu'):
    alphas = np.linspace(-range_val, range_val, steps)
    betas = np.linspace(-range_val, range_val, steps)
    losses = np.zeros((steps, steps))
    criterion = nn.CrossEntropyLoss()
    orig_weights = [p.data.clone() for p in model.parameters()]
    try:
        data_iter = iter(loader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)
    except: return alphas, betas, losses
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                for k, p in enumerate(model.parameters()):
                    p.data.copy_(orig_weights[k] + alpha * dir1[k] + beta * dir2[k])
                output = model(inputs)
                losses[i, j] = criterion(output, targets).item()
    for k, p in enumerate(model.parameters()):
        p.data.copy_(orig_weights[k])
    return alphas, betas, losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--compute_hessian', action='store_true')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--model_arch', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        opt_name = checkpoint.get('opt', 'Unknown')
        if args.model_arch: arch = args.model_arch
        elif 'SimpleMLP' in args.ckpt or args.dataset == 'MNIST': arch = 'SimpleMLP'
        else: arch = 'ResNet18'
            
        if arch == 'SimpleMLP': model = FallbackSimpleMLP().to(device)
        else: model = FallbackResNet18(num_classes=10).to(device)
        
        if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
        else: model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading: {e}")
        return

    if args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else: 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    max_eig_str = "N/A"
    if args.compute_hessian:
        try:
            max_eig = compute_hessian_eigenvalue(model, loader, criterion, device)
            max_eig_str = f"{max_eig:.2f}"
            with open(os.path.join(args.output_dir, "hessian_metrics.txt"), "a") as f:
                f.write(f"{os.path.basename(args.ckpt)},{opt_name},{max_eig:.4f}\n")
        except Exception as e:
            logging.debug(f"Failed to compute Hessian eigenvalue: {e}")

    try:
        dir1 = get_random_direction(model)
        dir2 = get_random_direction(model)
        alphas, betas, losses = compute_loss_surface(model, loader, dir1, dir2, 1.0, 25, device)
        plt.figure(figsize=(8, 7))
        contours = plt.contourf(alphas, betas, losses, levels=25, cmap='viridis')
        plt.colorbar(contours, label='Loss')
        plt.plot(0, 0, 'r*', markersize=15, label='Minima')
        plt.title(f"Landscape: {opt_name} | Hessian: {max_eig_str}")
        plt.savefig(os.path.join(args.output_dir, f"landscape_{args.dataset}_{opt_name}.png"))
        plt.close()
    except Exception as e: print(f"Viz Error: {e}")

if __name__ == "__main__":
    main()
