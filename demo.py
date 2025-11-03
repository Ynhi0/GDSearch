"""
Script demo để kiểm tra các module và chạy một thí nghiệm nhỏ.
"""

import numpy as np
import matplotlib.pyplot as plt
from test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint
from optimizers import SGD, SGDMomentum, RMSProp, Adam


def demo_test_functions():
    """Demo các hàm kiểm tra."""
    print("=" * 60)
    print("DEMO: Các Hàm Kiểm Tra")
    print("=" * 60)
    
    # Tạo các hàm
    functions = [
        Rosenbrock(a=1, b=100),
        IllConditionedQuadratic(kappa=100),
        SaddlePoint()
    ]
    
    # Điểm test
    x, y = 1.0, 1.0
    
    for func in functions:
        print(f"\n{func.name}:")
        print(f"  Tại điểm ({x}, {y}):")
        print(f"  - Giá trị hàm: {func.compute(x, y):.6f}")
        grad_x, grad_y = func.gradient(x, y)
        print(f"  - Gradient: ({grad_x:.6f}, {grad_y:.6f})")
        hessian = func.hessian(x, y)
        print(f"  - Hessian:")
        print(f"    [{hessian[0, 0]:8.2f}, {hessian[0, 1]:8.2f}]")
        print(f"    [{hessian[1, 0]:8.2f}, {hessian[1, 1]:8.2f}]")
        bounds = func.get_bounds()
        print(f"  - Giới hạn vẽ: x={bounds[0]}, y={bounds[1]}")


def demo_optimizers():
    """Demo các optimizer."""
    print("\n" + "=" * 60)
    print("DEMO: Các Thuật Toán Tối Ưu")
    print("=" * 60)
    
    # Tạo các optimizer
    optimizers_list = [
        SGD(lr=0.01),
        SGDMomentum(lr=0.01, beta=0.9),
        RMSProp(lr=0.01, decay_rate=0.9),
        Adam(lr=0.01)
    ]
    
    # Tham số và gradient test
    params = (1.0, 1.0)
    gradients = (2.0, -1.5)
    
    print(f"\nTham số ban đầu: {params}")
    print(f"Gradient: {gradients}")
    
    for opt in optimizers_list:
        opt.reset()
        new_params = opt.step(params, gradients)
        print(f"\n{opt.name}:")
        print(f"  Tham số mới: ({new_params[0]:.6f}, {new_params[1]:.6f})")


def demo_simple_optimization():
    """Demo một quá trình tối ưu hóa đơn giản."""
    print("\n" + "=" * 60)
    print("DEMO: Tối Ưu Hóa Đơn Giản")
    print("=" * 60)
    
    # Sử dụng hàm Rosenbrock và Adam
    func = Rosenbrock(a=1, b=100)
    opt = Adam(lr=0.01)
    
    # Điểm bắt đầu
    x, y = -1.0, 2.0
    
    print(f"\nHàm: {func.name}")
    print(f"Optimizer: {opt.name}")
    print(f"Điểm bắt đầu: ({x:.4f}, {y:.4f})")
    print(f"Điểm mục tiêu: (1, 1)")
    print(f"\nQuá trình tối ưu hóa:")
    print("-" * 60)
    
    num_iterations = 200
    history = []
    
    for i in range(num_iterations):
        loss = func.compute(x, y)
        grad_x, grad_y = func.gradient(x, y)
        
        history.append({
            'iteration': i,
            'x': x,
            'y': y,
            'loss': loss
        })
        
        # In thông tin mỗi 50 vòng lặp
        if i % 50 == 0:
            print(f"Iteration {i:3d}: x={x:7.4f}, y={y:7.4f}, loss={loss:10.6f}")
        
        # Cập nhật tham số
        x, y = opt.step((x, y), (grad_x, grad_y))
    
    # Kết quả cuối cùng
    final_loss = func.compute(x, y)
    print("-" * 60)
    print(f"Kết quả cuối cùng:")
    print(f"  Vị trí: ({x:.6f}, {y:.6f})")
    print(f"  Loss: {final_loss:.8f}")
    print(f"  Khoảng cách đến điểm tối ưu: {np.sqrt((x-1)**2 + (y-1)**2):.6f}")


def demo_comparison():
    """Demo so sánh các optimizer."""
    print("\n" + "=" * 60)
    print("DEMO: So Sánh Các Optimizer")
    print("=" * 60)
    
    # Hàm test
    func = IllConditionedQuadratic(kappa=100)
    
    # Các optimizer
    optimizers_list = [
        ('SGD', SGD(lr=0.001)),
        ('Momentum', SGDMomentum(lr=0.001, beta=0.9)),
        ('RMSProp', RMSProp(lr=0.01, decay_rate=0.9)),
        ('Adam', Adam(lr=0.01))
    ]
    
    # Điểm bắt đầu
    initial_x, initial_y = 1.0, 1.0
    num_iterations = 100
    
    print(f"\nHàm: {func.name}")
    print(f"Điểm bắt đầu: ({initial_x}, {initial_y})")
    print(f"Số vòng lặp: {num_iterations}")
    print(f"\nKết quả cuối cùng:")
    print("-" * 60)
    
    results = {}
    
    for name, opt in optimizers_list:
        opt.reset()
        x, y = initial_x, initial_y
        
        for i in range(num_iterations):
            loss = func.compute(x, y)
            grad_x, grad_y = func.gradient(x, y)
            x, y = opt.step((x, y), (grad_x, grad_y))
        
        final_loss = func.compute(x, y)
        distance = np.sqrt(x**2 + y**2)
        
        results[name] = {
            'position': (x, y),
            'loss': final_loss,
            'distance': distance
        }
        
        print(f"{name:10s}: loss={final_loss:10.6f}, distance={distance:8.6f}")
    
    print("-" * 60)
    
    # Tìm optimizer tốt nhất
    best_opt = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"\nOptimizer tốt nhất: {best_opt[0]}")


def main():
    """Hàm chính."""
    print("\n")
    print("#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  DEMO: GDSearch - So Sánh Thuật Toán Tối Ưu Hóa  ".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    try:
        # Chạy các demo
        demo_test_functions()
        demo_optimizers()
        demo_simple_optimization()
        demo_comparison()
        
        print("\n" + "=" * 60)
        print("DEMO HOÀN THÀNH!")
        print("=" * 60)
        print("\nĐể chạy thí nghiệm đầy đủ, sử dụng:")
        print("  python run_experiment.py")
        print("\nĐể tạo biểu đồ, sử dụng:")
        print("  python plot_results.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
