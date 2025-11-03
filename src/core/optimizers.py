"""
Module định nghĩa các thuật toán tối ưu hóa (optimizers).
"""

import numpy as np


class Optimizer:
    """Lớp cơ sở cho các thuật toán tối ưu hóa."""
    
    def __init__(self):
        """Khởi tạo optimizer."""
        pass
    
    def step(self, params, gradients):
        """
        Thực hiện một bước cập nhật tham số.
        
        Args:
            params: Tuple (x, y) - tham số hiện tại
            gradients: Tuple (grad_x, grad_y) - gradient tại tham số hiện tại
            
        Returns:
            Tuple (new_x, new_y) - tham số sau khi cập nhật
        """
        raise NotImplementedError("Phương thức step phải được triển khai trong lớp con")
    
    def reset(self):
        """Reset trạng thái nội bộ của optimizer."""
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) cơ bản.
    
    Công thức cập nhật: θ_new = θ_old - lr * gradient
    """
    
    def __init__(self, lr=0.01):
        """
        Khởi tạo SGD optimizer.
        
        Args:
            lr: Learning rate (tốc độ học) (mặc định: 0.01)
        """
        super().__init__()
        self.lr = lr
        self.name = f"SGD(lr={lr})"
    
    def step(self, params, gradients):
        """Thực hiện một bước SGD."""
        x, y = params
        grad_x, grad_y = gradients
        
        new_x = x - self.lr * grad_x
        new_y = y - self.lr * grad_y
        
        return new_x, new_y
    
    def reset(self):
        """SGD không có trạng thái nội bộ."""
        pass


class SGDMomentum(Optimizer):
    """
    SGD với Momentum.
    
    Công thức cập nhật:
        v_new = beta * v_old + gradient
        θ_new = θ_old - lr * v_new
    """
    
    def __init__(self, lr=0.01, beta=0.9):
        """
        Khởi tạo SGD với Momentum optimizer.
        
        Args:
            lr: Learning rate (tốc độ học) (mặc định: 0.01)
            beta: Hệ số momentum (mặc định: 0.9)
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.name = f"SGDMomentum(lr={lr}, beta={beta})"
        
        # Khởi tạo velocity
        self.v_x = 0.0
        self.v_y = 0.0
    
    def step(self, params, gradients):
        """Thực hiện một bước SGD với Momentum."""
        x, y = params
        grad_x, grad_y = gradients
        
        # Cập nhật velocity
        self.v_x = self.beta * self.v_x + grad_x
        self.v_y = self.beta * self.v_y + grad_y
        
        # Cập nhật tham số
        new_x = x - self.lr * self.v_x
        new_y = y - self.lr * self.v_y
        
        return new_x, new_y
    
    def reset(self):
        """Reset velocity về 0."""
        self.v_x = 0.0
        self.v_y = 0.0


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation).
    
    Công thức cập nhật:
        s_new = decay_rate * s_old + (1 - decay_rate) * gradient^2
        θ_new = θ_old - lr * gradient / sqrt(s_new + epsilon)
    """
    
    def __init__(self, lr=0.01, decay_rate=0.9, epsilon=1e-8):
        """
        Khởi tạo RMSProp optimizer.
        
        Args:
            lr: Learning rate (tốc độ học) (mặc định: 0.01)
            decay_rate: Tỷ lệ suy giảm cho moving average (mặc định: 0.9)
            epsilon: Hằng số nhỏ để tránh chia cho 0 (mặc định: 1e-8)
        """
        super().__init__()
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.name = f"RMSProp(lr={lr}, decay={decay_rate})"
        
        # Khởi tạo squared gradient accumulator
        self.s_x = 0.0
        self.s_y = 0.0
    
    def step(self, params, gradients):
        """Thực hiện một bước RMSProp."""
        x, y = params
        grad_x, grad_y = gradients
        
        # Cập nhật squared gradient accumulator
        self.s_x = self.decay_rate * self.s_x + (1 - self.decay_rate) * grad_x**2
        self.s_y = self.decay_rate * self.s_y + (1 - self.decay_rate) * grad_y**2
        
        # Cập nhật tham số với adaptive learning rate
        new_x = x - self.lr * grad_x / (np.sqrt(self.s_x) + self.epsilon)
        new_y = y - self.lr * grad_y / (np.sqrt(self.s_y) + self.epsilon)
        
        return new_x, new_y
    
    def reset(self):
        """Reset squared gradient accumulator về 0."""
        self.s_x = 0.0
        self.s_y = 0.0


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation).
    
    Công thức cập nhật:
        m_new = beta1 * m_old + (1 - beta1) * gradient
        v_new = beta2 * v_old + (1 - beta2) * gradient^2
        m_hat = m_new / (1 - beta1^t)
        v_hat = v_new / (1 - beta2^t)
        θ_new = θ_old - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Khởi tạo Adam optimizer.
        
        Args:
            lr: Learning rate (tốc độ học) (mặc định: 0.001)
            beta1: Hệ số suy giảm cho moment bậc 1 (mặc định: 0.9)
            beta2: Hệ số suy giảm cho moment bậc 2 (mặc định: 0.999)
            epsilon: Hằng số nhỏ để tránh chia cho 0 (mặc định: 1e-8)
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = f"Adam(lr={lr})"
        
        # Khởi tạo moment estimates
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        
        # Bộ đếm timestep
        self.t = 0
    
    def step(self, params, gradients):
        """Thực hiện một bước Adam."""
        x, y = params
        grad_x, grad_y = gradients
        
        # Tăng timestep
        self.t += 1
        
        # Cập nhật biased first moment estimate
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
        
        # Cập nhật biased second moment estimate
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x**2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y**2
        
        # Tính bias-corrected moment estimates
        m_x_hat = self.m_x / (1 - self.beta1**self.t)
        m_y_hat = self.m_y / (1 - self.beta1**self.t)
        v_x_hat = self.v_x / (1 - self.beta2**self.t)
        v_y_hat = self.v_y / (1 - self.beta2**self.t)
        
        # Cập nhật tham số
        new_x = x - self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
        new_y = y - self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)
        
        return new_x, new_y
    
    def reset(self):
        """Reset moment estimates và timestep về 0."""
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.t = 0
