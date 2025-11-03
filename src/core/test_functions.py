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


# ============================================================================
# High-Dimensional Test Functions (N-dimensional)
# ============================================================================


class HighDimensionalFunction:
    """Base class for high-dimensional test functions."""
    
    def __init__(self, dim=10):
        """
        Initialize high-dimensional function.
        
        Args:
            dim: Number of dimensions (default: 10)
        """
        self.dim = dim
        self.name = f"{self.__class__.__name__}(dim={dim})"
    
    def compute(self, x):
        """
        Compute function value at point x.
        
        Args:
            x: numpy array of shape (dim,)
            
        Returns:
            Function value at x
        """
        raise NotImplementedError
    
    def gradient(self, x):
        """
        Compute gradient at point x.
        
        Args:
            x: numpy array of shape (dim,)
            
        Returns:
            Gradient array of shape (dim,)
        """
        raise NotImplementedError
    
    def get_bounds(self):
        """
        Return search bounds.
        
        Returns:
            Tuple (lower_bound, upper_bound) for each dimension
        """
        raise NotImplementedError
    
    def get_optimum(self):
        """
        Return known global optimum.
        
        Returns:
            Tuple (x_opt, f_opt) - optimal point and value
        """
        raise NotImplementedError


class Rastrigin(HighDimensionalFunction):
    """
    Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    
    Highly multimodal function with many local minima.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    """
    
    def __init__(self, dim=10, A=10):
        """
        Initialize Rastrigin function.
        
        Args:
            dim: Number of dimensions (default: 10)
            A: Amplitude parameter (default: 10)
        """
        super().__init__(dim)
        self.A = A
        self.name = f"Rastrigin(dim={dim}, A={A})"
    
    def compute(self, x):
        """Compute Rastrigin function value."""
        x = np.asarray(x)
        return self.A * self.dim + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
    
    def gradient(self, x):
        """
        Compute gradient of Rastrigin function.
        
        df/dx_i = 2*x_i + 2*pi*A*sin(2*pi*x_i)
        """
        x = np.asarray(x)
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)
    
    def get_bounds(self):
        """Return search bounds for Rastrigin function."""
        return (-5.12, 5.12)
    
    def get_optimum(self):
        """Return known global optimum."""
        return np.zeros(self.dim), 0.0


class Ackley(HighDimensionalFunction):
    """
    Ackley function: f(x) = -a*exp(-b*sqrt(sum(x_i^2)/n)) - exp(sum(cos(c*x_i))/n) + a + e
    
    Characterized by nearly flat outer region and large hole at center.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    """
    
    def __init__(self, dim=10, a=20, b=0.2, c=2*np.pi):
        """
        Initialize Ackley function.
        
        Args:
            dim: Number of dimensions (default: 10)
            a: Amplitude parameter (default: 20)
            b: Width parameter (default: 0.2)
            c: Frequency parameter (default: 2*pi)
        """
        super().__init__(dim)
        self.a = a
        self.b = b
        self.c = c
        self.name = f"Ackley(dim={dim})"
    
    def compute(self, x):
        """Compute Ackley function value."""
        x = np.asarray(x)
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        return term1 + term2 + self.a + np.e
    
    def gradient(self, x):
        """
        Compute gradient of Ackley function.
        
        df/dx_i = (a*b / (n*sqrt(sum(x_j^2)/n))) * x_i * exp(-b*sqrt(sum(x_j^2)/n))
                  + (c / n) * sin(c*x_i) * exp(sum(cos(c*x_j))/n)
        """
        x = np.asarray(x)
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        sqrt_term = np.sqrt(sum_sq / n)
        
        # Term 1 derivative
        if sqrt_term > 1e-10:
            grad1 = (self.a * self.b / (n * sqrt_term)) * x * np.exp(-self.b * sqrt_term)
        else:
            grad1 = np.zeros_like(x)
        
        # Term 2 derivative
        grad2 = (self.c / n) * np.sin(self.c * x) * np.exp(sum_cos / n)
        
        return grad1 + grad2
    
    def get_bounds(self):
        """Return search bounds for Ackley function."""
        return (-32.768, 32.768)
    
    def get_optimum(self):
        """Return known global optimum."""
        return np.zeros(self.dim), 0.0


class Sphere(HighDimensionalFunction):
    """
    Sphere function: f(x) = sum(x_i^2)
    
    Simple convex function, easy to optimize.
    Global minimum at x = [0, 0, ..., 0] with f(x) = 0.
    """
    
    def __init__(self, dim=10):
        """
        Initialize Sphere function.
        
        Args:
            dim: Number of dimensions (default: 10)
        """
        super().__init__(dim)
        self.name = f"Sphere(dim={dim})"
    
    def compute(self, x):
        """Compute Sphere function value."""
        x = np.asarray(x)
        return np.sum(x**2)
    
    def gradient(self, x):
        """
        Compute gradient of Sphere function.
        
        df/dx_i = 2*x_i
        """
        x = np.asarray(x)
        return 2 * x
    
    def get_bounds(self):
        """Return search bounds for Sphere function."""
        return (-5.12, 5.12)
    
    def get_optimum(self):
        """Return known global optimum."""
        return np.zeros(self.dim), 0.0


class Schwefel(HighDimensionalFunction):
    """
    Schwefel function: f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
    
    Deceptive function where global minimum is far from local minima.
    Global minimum at x = [420.9687, ..., 420.9687] with f(x) ≈ 0.
    """
    
    def __init__(self, dim=10):
        """
        Initialize Schwefel function.
        
        Args:
            dim: Number of dimensions (default: 10)
        """
        super().__init__(dim)
        self.name = f"Schwefel(dim={dim})"
    
    def compute(self, x):
        """Compute Schwefel function value."""
        x = np.asarray(x)
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def gradient(self, x):
        """
        Compute gradient of Schwefel function.
        
        f(x) = 418.9829*n - sum(x_i * sin(sqrt(|x_i|)))
        df/dx_i = -sin(sqrt(|x_i|)) - x_i * cos(sqrt(|x_i|)) * sign(x_i) / (2*sqrt(|x_i|))
        """
        x = np.asarray(x)
        abs_x = np.abs(x)
        sqrt_abs_x = np.sqrt(abs_x)
        
        # Handle zero values
        grad = np.zeros_like(x, dtype=float)
        nonzero = abs_x > 1e-10
        
        grad[nonzero] = (-np.sin(sqrt_abs_x[nonzero]) - 
                         x[nonzero] * np.cos(sqrt_abs_x[nonzero]) * np.sign(x[nonzero]) / (2 * sqrt_abs_x[nonzero]))
        
        return grad
    
    def get_bounds(self):
        """Return search bounds for Schwefel function."""
        return (-500, 500)
    
    def get_optimum(self):
        """Return known global optimum."""
        return np.full(self.dim, 420.9687), 0.0
