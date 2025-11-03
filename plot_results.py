"""
Script để vẽ và trực quan hóa kết quả thí nghiệm.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from run_experiment import run_single_experiment


def plot_generalization_curves(df, title, save_path=None):
    """
    Plot generalization curves for neural network training runs.
    Shows (1) train_loss (avg per epoch) vs epochs along with test_loss
    and (2) test_accuracy vs epochs.

    Expects df with rows labeled by 'phase' ('train' or 'eval'),
    and columns: 'epoch', 'train_loss' (train rows), 'test_loss', 'test_accuracy' (eval rows).
    """
    # Aggregate training loss per epoch
    train_df = df[df.get('phase', 'train') == 'train']
    eval_df = df[df.get('phase', 'eval') == 'eval']

    train_epoch_loss = train_df.groupby('epoch')['train_loss'].mean() if not train_df.empty else pd.Series(dtype=float)
    eval_epoch_loss = eval_df.set_index('epoch')['test_loss'] if not eval_df.empty else pd.Series(dtype=float)
    eval_epoch_acc = eval_df.set_index('epoch')['test_accuracy'] if not eval_df.empty else pd.Series(dtype=float)

    epochs = sorted(set(train_epoch_loss.index).union(set(eval_epoch_loss.index)).union(set(eval_epoch_acc.index)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Loss curves
    if not train_epoch_loss.empty:
        ax1.plot(train_epoch_loss.index, train_epoch_loss.values, 'b-o', label='Train Loss', alpha=0.8)
    if not eval_epoch_loss.empty:
        ax1.plot(eval_epoch_loss.index, eval_epoch_loss.values, 'r-s', label='Test Loss', alpha=0.8)
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    if not eval_epoch_acc.empty:
        ax2.plot(eval_epoch_acc.index, eval_epoch_acc.values, 'g-^', label='Test Accuracy', alpha=0.9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epochs')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory(df, test_function, title, save_path=None):
    """
    Vẽ quỹ đạo tối ưu hóa trên đường đồng mức 2D của hàm kiểm tra.
    
    Args:
        df: DataFrame chứa lịch sử tối ưu hóa (với cột 'x', 'y')
        test_function: Đối tượng hàm kiểm tra (để tính giá trị hàm)
        title: Tiêu đề cho biểu đồ
        save_path: Đường dẫn để lưu hình (nếu None thì chỉ hiển thị)
    """
    # Lấy giới hạn vẽ đồ thị từ hàm kiểm tra
    (x_min, x_max), (y_min, y_max) = test_function.get_bounds()
    
    # Tạo lưới điểm
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Tính giá trị hàm trên lưới
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = test_function.compute(X[i, j], Y[i, j])
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Vẽ đường đồng mức với thang log
    levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max() + 1), 30)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6, linewidths=0.5)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    
    # Thêm colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Giá trị hàm', rotation=270, labelpad=20)
    
    # Vẽ quỹ đạo
    x_traj = df['x'].values
    y_traj = df['y'].values
    
    # Vẽ đường quỹ đạo
    ax.plot(x_traj, y_traj, 'r-', linewidth=2, alpha=0.7, label='Quỹ đạo')
    
    # Đánh dấu điểm bắt đầu và kết thúc
    ax.plot(x_traj[0], y_traj[0], 'go', markersize=12, label='Điểm bắt đầu', zorder=5)
    ax.plot(x_traj[-1], y_traj[-1], 'r*', markersize=15, label='Điểm kết thúc', zorder=5)
    
    # Vẽ một số điểm trung gian
    step = max(1, len(x_traj) // 10)
    ax.plot(x_traj[::step], y_traj[::step], 'ko', markersize=4, alpha=0.5, zorder=4)
    
    # Thiết lập nhãn và tiêu đề
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _compute_curvature_angles(df):
    """Compute curvature (turning angle in degrees) along a 2D trajectory from df columns x,y."""
    x = df['x'].values
    y = df['y'].values
    if len(x) < 3:
        return np.array([])
    # update vectors u_t = p_{t+1} - p_t for t=0..n-2
    u = np.stack([np.diff(x), np.diff(y)], axis=1)  # (n-1, 2)
    # angles between consecutive updates
    eps = 1e-12
    dots = np.sum(u[:-1] * u[1:], axis=1)
    norms = np.linalg.norm(u[:-1], axis=1) * np.linalg.norm(u[1:], axis=1)
    cos_vals = np.clip(dots / (norms + eps), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_vals))  # length n-2
    # align to iterations: set first angle as NaN for t=0, then angles for t=1..n-2, and NaN for last if desired
    angles_full = np.empty(len(x))
    angles_full[:] = np.nan
    angles_full[1:1+len(angles)] = angles
    return angles_full


def plot_dynamics_triplet(df, title, save_path=None):
    """
    Plot Update Norm, Gradient Norm, and Trajectory Curvature vs Iteration for GD logs.
    Curvature is the turning angle between consecutive update vectors.
    """
    iterations = df['iteration'].values
    update = df['update_norm'].values
    grad = df['grad_norm'].values
    curvature = _compute_curvature_angles(df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].semilogy(iterations, update, 'r-', linewidth=2)
    axes[0].set_ylabel('||Δθ|| (Update Norm)')
    axes[0].set_title('Update Step Norm vs Iteration')
    axes[0].grid(True, alpha=0.3, which='both')

    axes[1].semilogy(iterations, grad, 'g-', linewidth=2)
    axes[1].set_ylabel('||∇f|| (Grad Norm)')
    axes[1].set_title('Gradient Norm vs Iteration')
    axes[1].grid(True, alpha=0.3, which='both')

    axes[2].plot(iterations, curvature, 'b-', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Curvature (deg)')
    axes[2].set_title('Trajectory Curvature vs Iteration')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory_3d(df, test_function, title, save_path=None):
    """Plot 3D loss surface with trajectory overlay for a test function (e.g., Rosenbrock)."""
    (x_min, x_max), (y_min, y_max) = test_function.get_bounds()
    x_grid = np.linspace(x_min, x_max, 150)
    y_grid = np.linspace(y_min, y_max, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = test_function.compute(X[i, j], Y[i, j])

    x_traj = df['x'].values
    y_traj = df['y'].values
    z_traj = np.array([test_function.compute(x, y) for x, y in zip(x_traj, y_traj)])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
    ax.plot3D(x_traj, y_traj, z_traj, 'r-', linewidth=2)
    ax.scatter3D([x_traj[0]], [y_traj[0]], [z_traj[0]], color='g', s=50, label='Start')
    ax.scatter3D([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], color='r', s=60, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def trajectory_grid_adam(beta1_values, beta2_values, lr=0.01, a=1, b=100, initial=(-1.5, 2.0), num_iterations=2000, seed=42, save_path=None):
    """Generate a grid of trajectory plots for Adam with varying beta1 rows and beta2 columns on Rosenbrock."""
    func = Rosenbrock(a=a, b=b)
    nrows = len(beta1_values)
    ncols = len(beta2_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    (x_min, x_max), (y_min, y_max) = func.get_bounds()
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func.compute(X[i, j], Y[i, j])

    for r, beta1 in enumerate(beta1_values):
        for c, beta2 in enumerate(beta2_values):
            ax = axes[r, c]
            df = run_single_experiment(
                optimizer_config={'type': 'Adam', 'params': {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'epsilon': 1e-8}},
                function_config={'type': 'Rosenbrock', 'params': {'a': a, 'b': b}},
                initial_point=initial,
                num_iterations=num_iterations,
                seed=seed
            )
            levels = np.logspace(np.log10(Z.min()+1e-10), np.log10(Z.max()+1), 30)
            ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5, linewidths=0.5)
            ax.plot(df['x'].values, df['y'].values, 'r-', linewidth=1.5)
            ax.plot(df['x'].values[0], df['y'].values[0], 'go', ms=6)
            ax.plot(df['x'].values[-1], df['y'].values[-1], 'r*', ms=10)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_title(f"β1={beta1}, β2={beta2}")
            ax.grid(True, alpha=0.2)

    fig.suptitle(f"Adam Trajectory Grid on Rosenbrock (lr={lr})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def trajectory_series_momentum(betas, lr=0.01, a=1, b=100, initial=(-1.5, 2.0), num_iterations=2000, seed=42, save_path=None):
    """Generate a row of trajectory plots for SGD with Momentum for various beta values on Rosenbrock."""
    func = Rosenbrock(a=a, b=b)
    ncols = len(betas)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    if ncols == 1:
        axes = np.array([axes])

    (x_min, x_max), (y_min, y_max) = func.get_bounds()
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func.compute(X[i, j], Y[i, j])

    for c, beta in enumerate(betas):
        ax = axes[c]
        df = run_single_experiment(
            optimizer_config={'type': 'SGDMomentum', 'params': {'lr': lr, 'beta': beta}},
            function_config={'type': 'Rosenbrock', 'params': {'a': a, 'b': b}},
            initial_point=initial,
            num_iterations=num_iterations,
            seed=seed
        )
        levels = np.logspace(np.log10(Z.min()+1e-10), np.log10(Z.max()+1), 30)
        ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5, linewidths=0.5)
        ax.plot(df['x'].values, df['y'].values, 'r-', linewidth=1.5)
        ax.plot(df['x'].values[0], df['y'].values[0], 'go', ms=6)
        ax.plot(df['x'].values[-1], df['y'].values[-1], 'r*', ms=10)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_title(f"β={beta}")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"SGD Momentum Trajectories on Rosenbrock (lr={lr})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(df, title, save_path=None):
    """
    Vẽ các chỉ số tối ưu hóa theo thời gian.

    Hỗ trợ cả hai loại dữ liệu:
      - Kết quả hàm kiểm tra (GD) với cột 'iteration', 'loss', 'grad_norm', 'update_norm'
      - Kết quả mạng nơ-ron (NN) với cột 'phase'='train' và 'global_step', 'train_loss', 'grad_norm', 'update_norm'
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    if 'global_step' in df.columns and 'phase' in df.columns:
        # NN logs - use train phase and global_step
        train_df = df[df['phase'] == 'train']
        xvals = train_df['global_step'].values
        loss_vals = train_df['train_loss'].values
        grad_vals = train_df['grad_norm'].values
        upd_vals = train_df['update_norm'].values
        x_label = 'Global Step'
    else:
        # GD logs - use iteration
        xvals = df['iteration'].values
        loss_vals = df['loss'].values
        grad_vals = df['grad_norm'].values
        upd_vals = df['update_norm'].values
        x_label = 'Vòng lặp'

    # Subplot 1: Loss
    ax1 = axes[0]
    ax1.semilogy(xvals, loss_vals, 'b-', linewidth=2)
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Mất mát (Loss)', fontsize=11)
    ax1.set_title('Mất mát theo thời gian', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')

    # Subplot 2: Gradient Norm
    ax2 = axes[1]
    ax2.semilogy(xvals, grad_vals, 'g-', linewidth=2)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Chuẩn Gradient', fontsize=11)
    ax2.set_title('Chuẩn Gradient theo thời gian', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # Subplot 3: Update Norm
    ax3 = axes[2]
    ax3.semilogy(xvals, upd_vals, 'r-', linewidth=2)
    ax3.set_xlabel(x_label, fontsize=11)
    ax3.set_ylabel('Chuẩn Bước cập nhật', fontsize=11)
    ax3.set_title('Chuẩn Bước cập nhật theo thời gian', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(list_of_dfs, labels, metric, title, save_path=None):
    """
    Vẽ so sánh một chỉ số cụ thể giữa nhiều thí nghiệm.
    
    Args:
        list_of_dfs: Danh sách các DataFrame chứa lịch sử tối ưu hóa
        labels: Danh sách nhãn tương ứng với mỗi DataFrame
        metric: Tên cột chỉ số cần so sánh ('loss', 'grad_norm', 'update_norm')
        title: Tiêu đề cho biểu đồ
        save_path: Đường dẫn để lưu hình (nếu None thì chỉ hiển thị)
    """
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Định nghĩa màu sắc
    colors = plt.cm.tab10(np.linspace(0, 1, len(list_of_dfs)))

    # Determine x-axis based on metric type (NN vs GD)
    is_eval_metric = metric in ('test_accuracy', 'test_loss')
    x_label = 'Epoch' if is_eval_metric else 'Global Step'
    y_label_map = {
        'loss': 'Mất mát (Loss)',
        'train_loss': 'Mất mát Huấn luyện',
        'test_loss': 'Mất mát Kiểm tra',
        'grad_norm': 'Chuẩn Gradient',
        'update_norm': 'Chuẩn Bước cập nhật',
        'test_accuracy': 'Độ chính xác Kiểm tra',
    }

    # Vẽ từng đường
    for i, (df, label) in enumerate(zip(list_of_dfs, labels)):
        if is_eval_metric and 'phase' in df.columns:
            eval_df = df[df['phase'] == 'eval']
            x = eval_df['epoch'].values
            y = eval_df[metric].values
            ax.plot(x, y, linewidth=2, label=label, color=colors[i], alpha=0.9)
        else:
            # Prefer NN train logs when available, else GD logs
            if 'global_step' in df.columns and 'phase' in df.columns:
                train_df = df[df['phase'] == 'train']
                x = train_df['global_step'].values
                y = train_df.get(metric, pd.Series(index=train_df.index, dtype=float)).values if metric in train_df.columns else train_df['train_loss'].values
            else:
                x = df['iteration'].values
                y = df[metric].values
            ax.semilogy(x, y, linewidth=2, label=label, color=colors[i], alpha=0.8)

    # Thiết lập nhãn
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label_map.get(metric, metric), fontsize=12)
    
    # Tiêu đề
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_results(results_dir='results'):
    """
    Tải tất cả các file kết quả từ thư mục.
    
    Args:
        results_dir: Đường dẫn đến thư mục chứa kết quả
        
    Returns:
        Dictionary với key là tên file và value là DataFrame
    """
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Thư mục {results_dir} không tồn tại!")
        return results
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)
            results[filename] = df
    
    return results


def main():
    """Hàm chính để tạo các biểu đồ từ kết quả thí nghiệm."""
    # Tạo thư mục để lưu hình
    os.makedirs('plots', exist_ok=True)
    
    # Tải tất cả kết quả
    print("Đang tải kết quả thí nghiệm...")
    results = load_results('results')
    
    if not results:
        print("Không tìm thấy kết quả thí nghiệm!")
        print("Hãy chạy 'python run_experiment.py' trước.")
        return
    
    print(f"Đã tải {len(results)} file kết quả.")
    
    # Ví dụ 1: Vẽ quỹ đạo cho một số thí nghiệm cụ thể
    print("\nĐang tạo biểu đồ quỹ đạo...")
    
    # Tìm các file có cùng seed để so sánh
    seed = 42
    
    for filename, df in results.items():
        if f'seed{seed}' in filename:
            # Xác định hàm kiểm tra
            if 'Rosenbrock' in filename:
                test_func = Rosenbrock()
            elif 'IllConditionedQuadratic' in filename:
                test_func = IllConditionedQuadratic()
            elif 'SaddlePoint' in filename:
                test_func = SaddlePoint()
            else:
                continue
            
            # Tạo tiêu đề
            title = filename.replace('.csv', '').replace('_', ' ')
            
            # Tạo đường dẫn lưu
            save_path = os.path.join('plots', filename.replace('.csv', '_trajectory.png'))
            
            # Vẽ quỹ đạo
            plot_trajectory(df, test_func, title, save_path)
    
    # Ví dụ 2: Vẽ metrics cho từng thí nghiệm
    print("Đang tạo biểu đồ metrics...")
    
    for filename, df in results.items():
        if f'seed{seed}' in filename:
            # Tạo tiêu đề
            title = filename.replace('.csv', '').replace('_', ' ')
            
            # Tạo đường dẫn lưu
            save_path = os.path.join('plots', filename.replace('.csv', '_metrics.png'))
            
            # Vẽ metrics
            plot_metrics(df, title, save_path)
    
    # Ví dụ 3: So sánh các optimizer trên cùng một hàm
    print("Đang tạo biểu đồ so sánh...")
    
    # So sánh trên hàm Rosenbrock
    rosenbrock_results = {}
    for filename, df in results.items():
        if 'Rosenbrock' in filename and f'seed{seed}' in filename and 'lr0.001' in filename:
            # Lấy tên optimizer
            opt_name = filename.split('_')[0]
            rosenbrock_results[opt_name] = df
    
    if rosenbrock_results:
        dfs = list(rosenbrock_results.values())
        labels = list(rosenbrock_results.keys())
        
        # So sánh loss
        plot_comparison(
            dfs, labels, 'loss',
            f'So sánh Loss trên Rosenbrock (lr=0.001, seed={seed})',
            os.path.join('plots', f'comparison_Rosenbrock_loss_seed{seed}.png')
        )
        
        # So sánh gradient norm
        plot_comparison(
            dfs, labels, 'grad_norm',
            f'So sánh Chuẩn Gradient trên Rosenbrock (lr=0.001, seed={seed})',
            os.path.join('plots', f'comparison_Rosenbrock_gradnorm_seed{seed}.png')
        )
    
    # So sánh trên hàm IllConditionedQuadratic
    quad_results = {}
    for filename, df in results.items():
        if 'IllConditionedQuadratic' in filename and f'seed{seed}' in filename and 'lr0.001' in filename:
            opt_name = filename.split('_')[0]
            quad_results[opt_name] = df
    
    if quad_results:
        dfs = list(quad_results.values())
        labels = list(quad_results.keys())
        
        plot_comparison(
            dfs, labels, 'loss',
            f'So sánh Loss trên IllConditionedQuadratic (lr=0.001, seed={seed})',
            os.path.join('plots', f'comparison_IllCondQuad_loss_seed{seed}.png')
        )
    
    print("\nHoàn thành! Tất cả các biểu đồ được lưu trong thư mục 'plots/'")


if __name__ == '__main__':
    main()
