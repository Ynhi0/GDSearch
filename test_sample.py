"""
Script chạy một vài thí nghiệm mẫu để test nhanh.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from optimizers import SGD, SGDMomentum, RMSProp, Adam
from run_experiment import run_single_experiment


def run_sample_experiments():
    """Chạy một vài thí nghiệm mẫu."""
    print("Chạy thí nghiệm mẫu để kiểm tra...")
    
    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs('results', exist_ok=True)
    
    # Định nghĩa một vài thí nghiệm
    configs = [
        {
            'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.001}},
            'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
            'initial_point': (-1.5, 2.5),
            'num_iterations': 500,
            'seed': 42
        },
        {
            'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01}},
            'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
            'initial_point': (-1.5, 2.5),
            'num_iterations': 500,
            'seed': 42
        },
        {
            'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9}},
            'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
            'initial_point': (1.0, 1.0),
            'num_iterations': 500,
            'seed': 42
        },
    ]
    
    # Chạy thí nghiệm
    for i, config in enumerate(tqdm(configs, desc="Thí nghiệm"), 1):
        df = run_single_experiment(
            optimizer_config=config['optimizer_config'],
            function_config=config['function_config'],
            initial_point=config['initial_point'],
            num_iterations=config['num_iterations'],
            seed=config['seed']
        )
        
        # Tạo tên file
        opt_name = config['optimizer_config']['type']
        func_name = config['function_config']['type']
        lr = config['optimizer_config']['params'].get('lr', 0)
        seed = config['seed']
        
        filename = f"sample_{opt_name}_{func_name}_lr{lr}_seed{seed}.csv"
        filepath = os.path.join('results', filename)
        
        # Lưu kết quả
        df.to_csv(filepath, index=False)
        
        # In thông tin
        final_loss = df['loss'].iloc[-1]
        print(f"  ✓ {opt_name} on {func_name}: final loss = {final_loss:.6f}")
    
    print(f"\n✅ Hoàn thành! Đã lưu {len(configs)} file kết quả trong 'results/'")


if __name__ == '__main__':
    run_sample_experiments()
