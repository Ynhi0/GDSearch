
import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# 1. Thêm đường dẫn code vào hệ thống để import được
sys.path.append("/workspaces/GDSearch/kaggle/mnist_benchmark")

# 2. Import module gốc (không cần sửa file gốc)
try:
    import run_mnist
    print("✅ Đã import thành công module run_mnist từ repo.")
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
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
    
    print(f"🚀 Bắt đầu chạy MNIST Wrapper | Seeds: {seeds} | Opts: {len(optimizers)}")
    
    count_run = 0
    count_skip = 0
    
    for opt_name, lr in optimizers:
        for seed in seeds:
            # Tên file chuẩn khớp với run_mnist.py
            out_name = f"NN_SimpleMLP_MNIST_{opt_name}_lr{lr}_seed{seed}_benchmark.csv"
            out_path = results_dir / out_name
            
            if out_path.exists():
                # LOGIC BỎ QUA Ở ĐÂY
                print(f"⏩ SKIP: {out_name} (Đã có)")
                count_skip += 1
                continue
            
            print(f"\n▶️ RUN: {opt_name} | seed={seed}")
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
                print(f"❌ LỖI khi chạy {opt_name} seed {seed}: {e}")

    print(f"\n✅ HOÀN TẤT MNIST! Chạy mới: {count_run}, Bỏ qua: {count_skip}")

if __name__ == "__main__":
    main()
