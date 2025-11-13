"""
Script chính để chạy các thí nghiệm so sánh thuật toán tối ưu hóa.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint, Ackley2D
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW, AMSGrad


def run_single_experiment(optimizer_config, function_config, initial_point, num_iterations, seed):
    """
    Chạy một thí nghiệm duy nhất với cấu hình được chỉ định.
    
    Args:
        optimizer_config: Dictionary cấu hình optimizer
            {'type': 'SGD'|'SGDMomentum'|'RMSProp'|'Adam', 'params': {...}}
        function_config: Dictionary cấu hình hàm kiểm tra
            {'type': 'Rosenbrock'|'IllConditionedQuadratic'|'SaddlePoint', 'params': {...}}
        initial_point: Tuple (x0, y0) - điểm bắt đầu
        num_iterations: Số vòng lặp
        seed: Seed cho random number generator
        
    Returns:
        DataFrame chứa lịch sử quá trình tối ưu hóa
    """
    # Thiết lập seed để đảm bảo tính tái tạo
    np.random.seed(seed)
    
    # Khởi tạo hàm kiểm tra
    func_type = function_config['type']
    func_params = function_config.get('params', {})
    
    if func_type == 'Rosenbrock':
        test_function = Rosenbrock(**func_params)
    elif func_type == 'IllConditionedQuadratic':
        test_function = IllConditionedQuadratic(**func_params)
    elif func_type == 'SaddlePoint':
        test_function = SaddlePoint(**func_params)
    elif func_type == 'Ackley':
        test_function = Ackley2D(**func_params)
    else:
        raise ValueError(f"Loại hàm kiểm tra không hợp lệ: {func_type}")
    
    # Khởi tạo optimizer
    opt_type = optimizer_config['type']
    opt_params = optimizer_config.get('params', {})
    
    if opt_type == 'SGD':
        optimizer = SGD(**opt_params)
    elif opt_type == 'SGDMomentum':
        optimizer = SGDMomentum(**opt_params)
    elif opt_type == 'SGDNesterov':
        optimizer = SGDNesterov(**opt_params)
    elif opt_type == 'RMSProp':
        optimizer = RMSProp(**opt_params)
    elif opt_type == 'Adam':
        optimizer = Adam(**opt_params)
    elif opt_type == 'AdamW':
        optimizer = AdamW(**opt_params)
    elif opt_type == 'AMSGrad':
        optimizer = AMSGrad(**opt_params)
    else:
        raise ValueError(f"Loại optimizer không hợp lệ: {opt_type}")
    
    # Reset optimizer về trạng thái ban đầu
    optimizer.reset()
    
    # Khởi tạo tham số
    current_x, current_y = initial_point
    
    # Danh sách lưu trữ lịch sử
    history = []

    # Bắt đầu đo thời gian và theo dõi GPU (nếu có)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Vòng lặp tối ưu hóa
    for i in range(num_iterations):
        # Tính toán giá trị hàm và gradient
        loss = test_function.compute(current_x, current_y)
        grad_x, grad_y = test_function.gradient(current_x, current_y)
        
        # Tính chuẩn của gradient
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute Hessian eigenvalues for curvature analysis
        hessian = test_function.hessian(current_x, current_y)
        eigenvalues = np.linalg.eigvalsh(hessian)  # Returns sorted eigenvalues
        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[1]
        condition_number = abs(lambda_max / lambda_min) if abs(lambda_min) > 1e-10 else np.inf
        
        # Thực hiện bước cập nhật
        new_x, new_y = optimizer.step((current_x, current_y), (grad_x, grad_y))
        
        # Tính chuẩn của bước cập nhật
        update_norm = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        
        # Lưu thông tin vào lịch sử (including Hessian eigenvalues)
        history.append({
            'iteration': i,
            'x': current_x,
            'y': current_y,
            'loss': loss,
            'grad_norm': grad_norm,
            'update_norm': update_norm,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'condition_number': condition_number
        })
        
        # Cập nhật tham số
        current_x, current_y = new_x, new_y
    
    # Chuyển đổi lịch sử thành DataFrame
    df = pd.DataFrame(history)

    # Kết thúc đo thời gian và ghi lại thống kê GPU
    elapsed_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    # Thêm thông tin thời gian và bộ nhớ vào DataFrame (hằng cho mọi hàng)
    df['elapsed_time'] = elapsed_time
    df['peak_memory_MB'] = peak_memory

    return df


def create_experiment_configs():
    """
    Tạo danh sách các cấu hình thí nghiệm theo Ma trận Thiết kế.
    
    Returns:
        List các dictionary cấu hình thí nghiệm
    """
    configs = []
    
    # Điểm bắt đầu cho các hàm
    initial_rosenbrock = (-1.5, 2.0)
    initial_quad = (1.0, 1.0)
    initial_saddle = (0.5, 0.5)
    
    # ========== SGD Momentum trên Rosenbrock ==========
    # SGDM-R-1: beta=0.5
    configs.append({
        'experiment_id': 'SGDM-R-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.5}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # SGDM-R-2: beta=0.9
    configs.append({
        'experiment_id': 'SGDM-R-2',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # SGDM-R-3: beta=0.99
    configs.append({
        'experiment_id': 'SGDM-R-3',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.99}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ========== Adam trên Rosenbrock ==========
    # ADAM-R-1: beta1=0.9, beta2=0.999 (default)
    configs.append({
        'experiment_id': 'ADAM-R-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ADAM-R-2: beta1=0.5, beta2=0.999
    configs.append({
        'experiment_id': 'ADAM-R-2',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.5, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ADAM-R-3: beta1=0.9, beta2=0.9
    configs.append({
        'experiment_id': 'ADAM-R-3',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ADAM-R-4: beta1=0.5, beta2=0.9
    configs.append({
        'experiment_id': 'ADAM-R-4',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.5, 'beta2': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== Nesterov trên Rosenbrock ==========
    configs.append({
        'experiment_id': 'NAG-R-1',
        'optimizer_config': {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    configs.append({
        'experiment_id': 'NAG-R-2',
        'optimizer_config': {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.5}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== AdamW trên Rosenbrock (so sánh weight decay) ==========
    for wd, exp_id in [(0.0, 'ADAMW-R-0'), (0.01, 'ADAMW-R-1'), (0.05, 'ADAMW-R-5')]:
        configs.append({
            'experiment_id': exp_id,
            'optimizer_config': {'type': 'AdamW', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'weight_decay': wd}},
            'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
            'initial_point': initial_rosenbrock,
            'num_iterations': 10000,
            'seed': 42
        })

    # ========== AMSGrad trên Rosenbrock ==========
    configs.append({
        'experiment_id': 'AMSG-R-1',
        'optimizer_config': {'type': 'AMSGrad', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ========== SGD trên các hàm khác ==========
    # SGD trên Rosenbrock
    configs.append({
        'experiment_id': 'SGD-R-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.001}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # SGD trên IllConditionedQuadratic
    configs.append({
        'experiment_id': 'SGD-Q-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.001}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # SGD trên SaddlePoint
    configs.append({
        'experiment_id': 'SGD-S-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.01}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ========== RMSProp trên các hàm ==========
    # RMSProp trên Rosenbrock
    configs.append({
        'experiment_id': 'RMS-R-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # RMSProp trên IllConditionedQuadratic
    configs.append({
        'experiment_id': 'RMS-Q-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # RMSProp trên SaddlePoint
    configs.append({
        'experiment_id': 'RMS-S-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # ========== Thêm thí nghiệm trên các hàm khác ==========
    # SGDMomentum trên IllConditionedQuadratic
    configs.append({
        'experiment_id': 'SGDM-Q-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # SGDMomentum trên SaddlePoint
    configs.append({
        'experiment_id': 'SGDM-S-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # Adam trên IllConditionedQuadratic
    configs.append({
        'experiment_id': 'ADAM-Q-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })
    
    # Adam trên SaddlePoint
    configs.append({
        'experiment_id': 'ADAM-S-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })
    
    return configs


def generate_filename(config):
    """
    Tạo tên file duy nhất cho kết quả thí nghiệm.
    
    Args:
        config: Dictionary cấu hình thí nghiệm đầy đủ
        
    Returns:
        Tên file (string)
    """
    # Sử dụng experiment_id nếu có, nếu không tạo từ các tham số
    if 'experiment_id' in config:
        exp_id = config['experiment_id']
        filename = f"{exp_id}.csv"
    else:
        # Fallback cho các thí nghiệm không có ID
        opt_type = config['optimizer_config']['type']
        func_type = config['function_config']['type']
        seed = config['seed']
        filename = f"{opt_type}_{func_type}_seed{seed}.csv"
    
    return filename


def main():
    """Hàm chính để chạy tất cả các thí nghiệm."""
    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs('results', exist_ok=True)
    
    # Tạo danh sách cấu hình thí nghiệm
    configs = create_experiment_configs()
    
    print(f"Tổng số thí nghiệm: {len(configs)}")
    print("Bắt đầu chạy thí nghiệm...\n")
    
    # Chạy tất cả các thí nghiệm
    for config in tqdm(configs, desc="Chạy thí nghiệm"):
        # Chạy thí nghiệm
        df = run_single_experiment(
            optimizer_config=config['optimizer_config'],
            function_config=config['function_config'],
            initial_point=config['initial_point'],
            num_iterations=config['num_iterations'],
            seed=config['seed']
        )
        
        # Tạo tên file
        filename = generate_filename(config)
        
        # Lưu kết quả
        filepath = os.path.join('results', filename)
        df.to_csv(filepath, index=False)
        
        # Thêm thông tin experiment_id vào metadata nếu có
        if 'experiment_id' in config:
            # Có thể thêm experiment_id vào DataFrame nếu cần
            pass
    
    print("\nHoàn thành tất cả các thí nghiệm!")
    print(f"Kết quả được lưu trong thư mục 'results/'")
    print(f"Tổng số file: {len(configs)}")


if __name__ == '__main__':
    main()
