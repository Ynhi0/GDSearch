"""
Module định nghĩa các hàm kiểm tra (test functions) để đánh giá thuật toán tối ưu hóa.
"""

import numpy as np


class TestFunction:
    """Lớp cơ sở cho các hàm kiểm tra."""
    
    def __init__(self):
        """Khởi tạo hàm kiểm tra."""
        pass
    
    def compute(self, x, y):
        """
        Tính giá trị hàm tại điểm (x, y).
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            
        Returns:
            Giá trị hàm tại (x, y)
        """
        raise NotImplementedError("Phương thức compute phải được triển khai trong lớp con")
    
    def gradient(self, x, y):
        """
        Tính gradient của hàm tại điểm (x, y).
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            
        Returns:
            Tuple (grad_x, grad_y) - gradient theo x và y
        """
        raise NotImplementedError("Phương thức gradient phải được triển khai trong lớp con")
    
    def hessian(self, x, y):
        """
        Tính ma trận Hessian của hàm tại điểm (x, y).
        
        Args:
            x: Tọa độ x
            y: Tọa độ y
            
        Returns:
            Ma trận Hessian 2x2 dạng numpy array
        """
        raise NotImplementedError("Phương thức hessian phải được triển khai trong lớp con")
    
    def get_bounds(self):
        """
        Trả về giới hạn vẽ đồ thị.
        
        Returns:
            Tuple ((x_min, x_max), (y_min, y_max))
        """
        raise NotImplementedError("Phương thức get_bounds phải được triển khai trong lớp con")


class Rosenbrock(TestFunction):
    """
    Hàm Rosenbrock: f(x,y) = (a - x)^2 + b(y - x^2)^2
    
    Đây là một hàm kiểm tra cổ điển với một thung lũng hẹp.
    Điểm cực tiểu toàn cục tại (a, a^2) với giá trị 0.
    """
    
    def __init__(self, a=1, b=100):
        """
        Khởi tạo hàm Rosenbrock.
        
        Args:
            a: Tham số a (mặc định: 1)
            b: Tham số b (mặc định: 100)
        """
        super().__init__()
        self.a = a
        self.b = b
        self.name = f"Rosenbrock(a={a}, b={b})"
    
    def compute(self, x, y):
        """Tính giá trị hàm Rosenbrock."""
        return (self.a - x)**2 + self.b * (y - x**2)**2
    
    def gradient(self, x, y):
        """
        Tính gradient giải tích của hàm Rosenbrock.
        
        df/dx = -2(a - x) - 4bx(y - x^2)
        df/dy = 2b(y - x^2)
        """
        grad_x = -2 * (self.a - x) - 4 * self.b * x * (y - x**2)
        grad_y = 2 * self.b * (y - x**2)
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """
        Tính ma trận Hessian của hàm Rosenbrock.
        
        d²f/dx² = 2 - 4b(y - 3x^2)
        d²f/dxdy = -4bx
        d²f/dy² = 2b
        """
        h_xx = 2 - 4 * self.b * (y - 3 * x**2)
        h_xy = -4 * self.b * x
        h_yy = 2 * self.b
        return np.array([[h_xx, h_xy], [h_xy, h_yy]])
    
    def get_bounds(self):
        """Trả về giới hạn vẽ đồ thị cho hàm Rosenbrock."""
        return (-2, 2), (-1, 3)


class IllConditionedQuadratic(TestFunction):
    """
    Hàm Quadratic có điều kiện xấu: f(x,y) = 0.5 * (kappa * x^2 + y^2)
    
    Đây là một hàm bậc hai đơn giản với số điều kiện (condition number) được kiểm soát.
    Điểm cực tiểu toàn cục tại (0, 0) với giá trị 0.
    """
    
    def __init__(self, kappa=100):
        """
        Khởi tạo hàm Quadratic có điều kiện xấu.
        
        Args:
            kappa: Số điều kiện (condition number) - tỷ lệ giữa các trục (mặc định: 100)
        """
        super().__init__()
        self.kappa = kappa
        self.name = f"IllConditionedQuadratic(kappa={kappa})"
    
    def compute(self, x, y):
        """Tính giá trị hàm Quadratic có điều kiện xấu."""
        return 0.5 * (self.kappa * x**2 + y**2)
    
    def gradient(self, x, y):
        """
        Tính gradient của hàm Quadratic có điều kiện xấu.
        
        df/dx = kappa * x
        df/dy = y
        """
        grad_x = self.kappa * x
        grad_y = y
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """
        Tính ma trận Hessian của hàm Quadratic có điều kiện xấu.
        
        Hessian là ma trận đường chéo với các phần tử [kappa, 1].
        """
        return np.array([[self.kappa, 0], [0, 1]])
    
    def get_bounds(self):
        """Trả về giới hạn vẽ đồ thị cho hàm Quadratic có điều kiện xấu."""
        scale = max(1, np.sqrt(self.kappa) / 10)
        return (-scale, scale), (-scale * np.sqrt(self.kappa), scale * np.sqrt(self.kappa))


class SaddlePoint(TestFunction):
    """
    Hàm Saddle Point: f(x,y) = 0.5 * (x^2 - y^2)
    
    Đây là một hàm có điểm yên ngựa (saddle point) tại gốc tọa độ.
    Không có cực tiểu toàn cục (hàm không bị chặn dưới).
    """
    
    def __init__(self):
        """Khởi tạo hàm Saddle Point."""
        super().__init__()
        self.name = "SaddlePoint"
    
    def compute(self, x, y):
        """Tính giá trị hàm Saddle Point."""
        return 0.5 * (x**2 - y**2)
    
    def gradient(self, x, y):
        """
        Tính gradient của hàm Saddle Point.
        
        df/dx = x
        df/dy = -y
        """
        grad_x = x
        grad_y = -y
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """
        Tính ma trận Hessian của hàm Saddle Point.
        
        Hessian là ma trận đường chéo với các phần tử [1, -1].
        """
        return np.array([[1, 0], [0, -1]])
    
    def get_bounds(self):
        """Trả về giới hạn vẽ đồ thị cho hàm Saddle Point."""
        return (-2, 2), (-2, 2)
