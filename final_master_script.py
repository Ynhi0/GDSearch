#!/usr/bin/env python3
"""
Final Master Script for GDSearch Thesis Pipeline (Local Version)

This script runs the complete thesis experiments locally, with smart resume,
patch creation, and result packaging.
"""

import os
import sys
import glob
import shutil
import subprocess
import time
import argparse
from pathlib import Path
import pandas as pd

# ==================================================================================
# 1. CẤU HÌNH HỆ THỐNG & TÌM MÃ NGUỒN (AUTO-DETECT)
# ==================================================================================
print(f"{'='*80}")
print("🚀 [INIT] KHỞI TẠO HỆ THỐNG GDSearch (THESIS FINAL RUN - LOCAL VERSION)...")
print(f"{'='*80}")

POSSIBLE_PATHS = [
    "/workspaces/GDSearch",
    ".",
    ".."
]

REPO_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(os.path.join(path, "src")):
        REPO_PATH = os.path.abspath(path)
        break

if not REPO_PATH:
    print("⚠️ Không tìm thấy đường dẫn mặc định. Đang quét...")
    for root, dirs, files in os.walk("."):
        if "src" in dirs and "scripts" in dirs:
            REPO_PATH = os.path.abspath(root)
            break

if REPO_PATH:
    print(f"✅ Đã tìm thấy Repo tại: {REPO_PATH}")
    os.environ['PYTHONPATH'] = f"{REPO_PATH}:{os.environ.get('PYTHONPATH', '')}"
    sys.path.append(REPO_PATH)
else:
    raise RuntimeError("❌ LỖI: Không tìm thấy thư mục code.")

# ==================================================================================
# 2. CÀI ĐẶT THƯ VIỆN PHỤ TRỢ
# ==================================================================================
print("\n📦 [SETUP] Đang cài đặt thư viện (torchtext, medpy, portalocker)...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchtext", "portalocker", "medpy", "pandas", "matplotlib", "seaborn", "scipy", "--quiet"])
    print("✅ Cài đặt hoàn tất.")
except Exception as e:
    print(f"⚠️ Cảnh báo cài đặt: {e}")

# ==================================================================================
# 3. THIẾT LẬP THƯ MỤC ĐẦU RA
# ==================================================================================
BASE_OUT = f"{REPO_PATH}/thesis_final_output"
DIRS = {
    "results": f"{BASE_OUT}/results",
    "plots": f"{BASE_OUT}/plots",
    "checkpoints": f"{BASE_OUT}/checkpoints",
    "tables": f"{BASE_OUT}/tables",
    "reports": f"{BASE_OUT}/reports",
    "hessian": f"{BASE_OUT}/hessian"
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
os.makedirs(f"{DIRS['results']}/mnist", exist_ok=True)
os.makedirs(f"{DIRS['results']}/cifar10", exist_ok=True)
os.makedirs(f"{DIRS['results']}/sam_sensitivity", exist_ok=True)
os.makedirs(f"{DIRS['results']}/components", exist_ok=True)
os.makedirs(f"{DIRS['results']}/robustness", exist_ok=True)
os.makedirs(f"{DIRS['results']}/nlp", exist_ok=True)
os.makedirs(f"{DIRS['results']}/medical", exist_ok=True)

# ==================================================================================
# 4. KHÔI PHỤC DỮ LIỆU & SỬA LỖI (SMART RESUME)
# ==================================================================================
print(f"\n{'='*80}")
print("♻️ [RESUME] KIỂM TRA VÀ KHÔI PHỤC KẾT QUẢ CŨ...")

PARTIAL_PATH = None
# Tìm nguồn dữ liệu cũ trong workspace
for root, dirs, files in os.walk(REPO_PATH):
    if "results" in dirs and "checkpoints" in dirs and "thesis_final_output" in root:
        PARTIAL_PATH = root
        break
if not PARTIAL_PATH:
    cand = glob.glob(f"{REPO_PATH}/**/results", recursive=True)
    if cand: PARTIAL_PATH = os.path.dirname(cand[0])

if PARTIAL_PATH and PARTIAL_PATH != BASE_OUT:
    print(f"📦 Tìm thấy dữ liệu cũ tại: {PARTIAL_PATH}")
    print("⏳ Đang copy dữ liệu (Chế độ an toàn - Không ghi đè)...")
    
    # Copy toàn bộ
    subprocess.run(f"cp -rn {PARTIAL_PATH}/* {BASE_OUT}/", shell=True)
    
    # --- MIGRATION: GOM FILE MNIST VỀ ĐÚNG CHỖ ---
    print("🔧 Đang đồng bộ cấu trúc thư mục MNIST...")
    old_mnist_files = glob.glob(f"{BASE_OUT}/results/mnist_bs*/*.csv")
    moved_count = 0
    for f in old_mnist_files:
        fname = os.path.basename(f)
        dest = f"{DIRS['results']}/mnist/{fname}"
        if not os.path.exists(dest):
            shutil.copy2(f, dest)
            moved_count += 1
            
    print(f"✅ Đã đồng bộ {moved_count} file MNIST.")
    
    # --- MIGRATION: GOM FILE CIFAR-10 VỀ ĐÚNG CHỖ ---
    print("🔧 Đang đồng bộ cấu trúc thư mục CIFAR-10...")
    old_cifar_files = glob.glob(f"{BASE_OUT}/results/cifar10*/*.csv")
    cifar_moved_count = 0
    for f in old_cifar_files:
        fname = os.path.basename(f)
        dest = f"{DIRS['results']}/cifar10/{fname}"
        if os.path.abspath(f) != os.path.abspath(dest) and not os.path.exists(dest):
             shutil.copy2(f, dest)
             cifar_moved_count += 1
    print(f"✅ Đã đồng bộ {cifar_moved_count} file CIFAR-10.")

    total_files = len(glob.glob(f"{BASE_OUT}/**/*.csv", recursive=True))
    print(f"📊 Tổng số file kết quả hiện có: {total_files}")
else:
    print("ℹ️ Chạy mới hoàn toàn (Fresh Run).")

# --- FIX LỖI DATASET BỊ HỎNG (CRITICAL) ---
if os.path.exists("./data"):
    print("\n🧹 [CLEANUP] Phát hiện folder './data' cũ. Đang xóa để tránh lỗi nổ gradient/crash...")
    shutil.rmtree("./data")
    print("✅ Đã xóa sạch dữ liệu cũ.")

# ==================================================================================
# 5. TẠO FILE PATCH: VISUALIZATION & HESSIAN (OPTIMIZED)
# ==================================================================================
print("\n🛠️ [PATCH] Đang tạo file 'patch_landscape.py' (Tối ưu hóa đa luồng)...")

patch_content = r'''
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
        except: pass

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
'''
with open(f"{REPO_PATH}/patch_landscape.py", "w") as f:
    f.write(patch_content)

# ==================================================================================
# 6. TẠO SCRIPT CHẠY MNIST AN TOÀN (WRAPPER METHOD)
# ==================================================================================
print("\n🛠️ [WRAPPER] Đang tạo script chạy MNIST riêng biệt (Không vá file gốc)...")

wrapper_content = f"""
import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# 1. Thêm đường dẫn code vào hệ thống để import được
sys.path.append("{REPO_PATH}/kaggle/mnist_benchmark")

# 2. Import module gốc (không cần sửa file gốc)
try:
    import run_mnist
    print("✅ Đã import thành công module run_mnist từ repo.")
except ImportError as e:
    print(f"❌ Lỗi import: {{e}}")
    print("Đường dẫn sys.path hiện tại:", sys.path)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    # Cấu hình chạy
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    optimizers = [
        ('SGD', 0.01), 
        ('SGD_Momentum', 0.05), 
        ('Adam', 0.001), 
        ('AdamW', 0.001), 
        ('AMSGrad', 0.001),
        ('SAM_SGD', 0.01),
        ('SAM_Adam', 0.001)
    ]
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Bắt đầu chạy MNIST Wrapper | Seeds: {{seeds}} | Opts: {{len(optimizers)}}")
    
    count_run = 0
    count_skip = 0
    
    for opt_name, lr in optimizers:
        for seed in seeds:
            # Tên file chuẩn khớp với run_mnist.py
            out_name = f"NN_SimpleMLP_MNIST_{{opt_name}}_lr{{lr}}_seed{{seed}}_benchmark.csv"
            out_path = results_dir / out_name
            
            if out_path.exists():
                # LOGIC BỎ QUA Ở ĐÂY
                print(f"⏩ SKIP: {{out_name}} (Đã có)")
                count_skip += 1
                continue
            
            print(f"\\n▶️ RUN: {{opt_name}} | seed={{seed}}")
            try:
                # Gọi hàm chạy từ module gốc
                run_mnist.run_single_experiment(
                    optimizer_name=opt_name,
                    seed=seed,
                    lr=lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    results_dir=results_dir,
                    resume=True,  # Enable resume
                    ckpt_dir=Path("checkpoints_mnist")
                )
                count_run += 1
            except Exception as e:
                print(f"❌ LỖI khi chạy {{opt_name}} seed {{seed}}: {{e}}")

    print(f"\\n✅ HOÀN TẤT MNIST! Chạy mới: {{count_run}}, Bỏ qua: {{count_skip}}")

if __name__ == "__main__":
    main()
"""

with open(f"{REPO_PATH}/run_mnist_safe.py", "w") as f:
    f.write(wrapper_content)

# ==================================================================================
# 7. CHẠY THÍ NGHIỆM (CẬP NHẬT LỆNH GỌI MNIST)
# ==================================================================================
def run_cmd(desc, cmd):
    print(f"\n{'='*60}\n🚀 [EXEC] {desc}\n{'='*60}")
    try: 
        subprocess.run(cmd, shell=True, check=True)
        print("✅ DONE")
    except subprocess.CalledProcessError: 
        print("⚠️ FAIL / SKIPPED")

print(f"\n{'='*80}\n▶️ BẮT ĐẦU CHẠY TOÀN BỘ (CHẾ ĐỘ THÔNG MINH)...")

# [1] MNIST (Dùng file wrapper mới tạo)
run_cmd("1. MNIST Benchmark (Safe Wrapper)", 
    f"python {REPO_PATH}/run_mnist_safe.py --epochs 30 --results-dir {DIRS['results']}/mnist")


# [2] CIFAR-10 (Dữ liệu đã xóa sạch ở trên -> Sẽ chạy ngon)
run_cmd("2. CIFAR-10 Benchmark", 
    f"python {REPO_PATH}/kaggle/cifar10_benchmark/run_cifar10.py --epochs 50 --results-dir {DIRS['results']}/cifar10 --ckpt-dir {DIRS['checkpoints']} --resume")

# [3] SAM Sensitivity
run_cmd("3. SAM Sensitivity", 
    f"python {REPO_PATH}/kaggle/resnet18_cifar10.py --optimizer SAM_SGD --rho-sweep '0.01,0.02,0.05,0.1,0.2' --epochs 20 --results-dir {DIRS['results']}/sam_sensitivity")

# [4] Ablation (Fixed: removed invalid arguments)
run_cmd("4. Component Ablation", 
    f"python {REPO_PATH}/scripts/run_nn_ablation.py --results-dir {DIRS['results']}/components --plots-dir {DIRS['plots']}")

# [5] Robustness
run_cmd("5. Robustness", 
    f"python {REPO_PATH}/src/experiments/run_initial_condition_robustness.py --results-dir {DIRS['results']}/robustness --plots-dir {DIRS['plots']}")

# [6] NLP
run_cmd("6. NLP Benchmark", 
    f"python {REPO_PATH}/kaggle/nlp_benchmark/run_nlp.py --epochs 5 --results-dir {DIRS['results']}/nlp --resume")

# [7] Medical
run_cmd("7. Medical Seg", 
    f"python {REPO_PATH}/kaggle/medical_benchmark/run_seg.py --epochs 40 --results-dir {DIRS['results']}/medical --resume")

# [8] Visualization (Dùng patch tối ưu)
print("\n>>> 🎨 Visualizing...")
# Tìm checkpoint tốt nhất
adam_best = next((f for f in glob.glob(f"{DIRS['checkpoints']}/*Adam*.pt")), None)
sam_best = next((f for f in glob.glob(f"{DIRS['checkpoints']}/*SAM*.pt")), None)

if adam_best and sam_best:
    run_cmd("8a. Adam Landscape", f"python {REPO_PATH}/patch_landscape.py --ckpt '{adam_best}' --output_dir {DIRS['plots']} --dataset CIFAR --compute_hessian")
    run_cmd("8b. SAM Landscape", f"python {REPO_PATH}/patch_landscape.py --ckpt '{sam_best}' --output_dir {DIRS['plots']} --dataset CIFAR --compute_hessian")

# [9] Reports
run_cmd("Summaries", f"python {REPO_PATH}/scripts/generate_summaries.py --results_dir {DIRS['results']} --output_dir {DIRS['tables']}")
run_cmd("LaTeX", f"python {REPO_PATH}/scripts/generate_latex_tables.py --data_dir {DIRS['tables']} --output_dir {DIRS['tables']}")

# ==================================================================================
# 8. ĐÓNG GÓI
# ==================================================================================
print(f"\n{'='*80}\n📦 ĐÓNG GÓI KẾT QUẢ CUỐI CÙNG...\n{'='*80}")
shutil.make_archive(f"{REPO_PATH}/THESIS_FULL_DONE", 'zip', BASE_OUT)
print(f"✅ HOÀN TẤT! Tải file 'THESIS_FULL_DONE.zip' về.")