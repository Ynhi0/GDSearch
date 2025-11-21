import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# --- 1. MODEL DEFINITION (Phải khớp với model đã train) ---
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
        return self.fc3(x)

# --- 2. HESSIAN CALCULATION (Tính Max Eigenvalue - Độ phẳng) ---
def compute_max_eigenvalue(model, loader, device, max_iter=100, tol=1e-3):
    """Sử dụng Power Iteration để ước lượng Max Eigenvalue của Hessian"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Lấy 1 batch mẫu
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    
    # Khởi tạo vector ngẫu nhiên v
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    # Chuẩn hóa v
    v_norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
    v = [vi / v_norm for vi in v]
    
    eigenvalue = 0.0
    
    for _ in range(max_iter):
        # Tính Hv (Hessian vector product)
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Dot product g*v
        gv = sum(torch.sum(gi * vi) for gi, vi in zip(grads, v))
        
        # Gradient của gv chính là Hv
        Hv = torch.autograd.grad(gv, params, retain_graph=True)
        
        # Tính Rayleigh quotient: v^T * H * v = v^T * Hv
        new_eigenvalue = sum(torch.sum(vi * hvi) for vi, hvi in zip(v, Hv)).item()
        
        # Chuẩn hóa lại Hv để update v
        Hv_norm = torch.sqrt(sum(torch.sum(hvi**2) for hvi in Hv))
        v = [hvi / Hv_norm for hvi in Hv]
        
        if abs(new_eigenvalue - eigenvalue) < tol:
            break
        eigenvalue = new_eigenvalue
        
    return eigenvalue

# --- 3. VISUALIZATION UTILS ---
def get_random_direction(model):
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        d = d * (p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction

def compute_loss_surface(model, loader, dir1, dir2, range_val, steps, device):
    alphas = np.linspace(-range_val, range_val, steps)
    betas = np.linspace(-range_val, range_val, steps)
    losses = np.zeros((steps, steps))
    criterion = nn.CrossEntropyLoss()
    orig_weights = [p.data.clone() for p in model.parameters()]
    
    data, target = next(iter(loader))
    data, target = data.to(device), target.to(device)
    
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                for k, p in enumerate(model.parameters()):
                    p.data.copy_(orig_weights[k] + alpha * dir1[k] + beta * dir2[k])
                output = model(data)
                losses[i, j] = criterion(output, target).item()
                for k, p in enumerate(model.parameters()):
                    p.data.copy_(orig_weights[k])
    return alphas, betas, losses

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--compute_hessian', action='store_true')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading: {args.ckpt}")
    model = SimpleMLP().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    opt_name = checkpoint.get('opt', 'Unknown')
    
    # Load Data
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
    
    # 1. Compute Hessian (Định lượng)
    if args.compute_hessian:
        print("Calculating Max Eigenvalue (Sharpness)...")
        max_eig = compute_max_eigenvalue(model, loader, device)
        print(f"🎓 Max Eigenvalue ({opt_name}): {max_eig:.4f}")
        # Lưu vào file text để tổng hợp sau
        with open(f"{args.output_dir}/hessian_metrics.txt", "a") as f:
            f.write(f"{opt_name},{max_eig:.4f}\n")
    
    # 2. Visualize (Định tính)
    print("Generating Contour Plot...")
    dir1 = get_random_direction(model)
    dir2 = get_random_direction(model)
    alphas, betas, losses = compute_loss_surface(model, loader, dir1, dir2, 1.0, 20, device)
    
    plt.figure(figsize=(7, 6))
    plt.contourf(alphas, betas, losses, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.plot(0, 0, 'rx', markersize=10, label='Minima')
    plt.title(f"Loss Landscape: {opt_name}")
    plt.savefig(f"{args.output_dir}/landscape_{opt_name}.png")
    plt.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()