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
        # Hỗ trợ cả tuple (x,y) và numpy array
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            new_x = x - self.lr * grad_x
            new_y = y - self.lr * grad_y
            return new_x, new_y
        else:
            # Xử lý array (cho neural networks)
            return params - self.lr * gradients
    
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
        self.v = None  # Cho neural networks
    
    def step(self, params, gradients):
        """Thực hiện một bước SGD với Momentum."""
        # Hỗ trợ cả tuple (x,y) và numpy array
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            
            # Cập nhật velocity
            self.v_x = self.beta * self.v_x + grad_x
            self.v_y = self.beta * self.v_y + grad_y
            
            # Cập nhật tham số
            new_x = x - self.lr * self.v_x
            new_y = y - self.lr * self.v_y
            
            return new_x, new_y
        else:
            # Xử lý array (cho neural networks)
            if self.v is None:
                self.v = np.zeros_like(params)
            
            # Cập nhật velocity
            self.v = self.beta * self.v + gradients
            
            # Cập nhật tham số
            return params - self.lr * self.v
    
    def reset(self):
        """Reset velocity về 0."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None


class SGDNesterov(Optimizer):
    """
    SGD with Nesterov Accelerated Gradient (NAG).

    Update rule (PyTorch-style formulation using current gradient g_t):
        v_t = beta * v_{t-1} + g_t
        d_t = g_t + beta * v_t
        theta_new = theta_old - lr * d_t

    This approximates the lookahead gradient without requiring function access.
    """

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.name = f"SGDNesterov(lr={lr}, beta={beta})"

        # State
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None  # array state for NN

    def step(self, params, gradients):
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            # update velocity
            self.v_x = self.beta * self.v_x + grad_x
            self.v_y = self.beta * self.v_y + grad_y
            # nesterov accelerated gradient
            d_x = grad_x + self.beta * self.v_x
            d_y = grad_y + self.beta * self.v_y
            new_x = x - self.lr * d_x
            new_y = y - self.lr * d_y
            return new_x, new_y
        else:
            if self.v is None:
                self.v = np.zeros_like(params)
            self.v = self.beta * self.v + gradients
            d = gradients + self.beta * self.v
            return params - self.lr * d

    def reset(self):
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None


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
        self.s = None  # Cho neural networks
    
    def step(self, params, gradients):
        """Thực hiện một bước RMSProp."""
        # Hỗ trợ cả tuple (x,y) và numpy array
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            
            # Cập nhật squared gradient accumulator
            self.s_x = self.decay_rate * self.s_x + (1 - self.decay_rate) * grad_x**2
            self.s_y = self.decay_rate * self.s_y + (1 - self.decay_rate) * grad_y**2
            
            # Cập nhật tham số với adaptive learning rate
            new_x = x - self.lr * grad_x / (np.sqrt(self.s_x) + self.epsilon)
            new_y = y - self.lr * grad_y / (np.sqrt(self.s_y) + self.epsilon)
            
            return new_x, new_y
        else:
            # Xử lý array (cho neural networks)
            if self.s is None:
                self.s = np.zeros_like(params)
            
            # Cập nhật squared gradient accumulator
            self.s = self.decay_rate * self.s + (1 - self.decay_rate) * gradients**2
            
            # Cập nhật tham số với adaptive learning rate
            return params - self.lr * gradients / (np.sqrt(self.s) + self.epsilon)
    
    def reset(self):
        """Reset squared gradient accumulator về 0."""
        self.s_x = 0.0
        self.s_y = 0.0
        self.s = None


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
        self.m = None  # Cho neural networks
        self.v = None  # Cho neural networks
        
        # Bộ đếm timestep
        self.t = 0
    
    def step(self, params, gradients):
        """Thực hiện một bước Adam."""
        # Tăng timestep
        self.t += 1
        
        # Hỗ trợ cả tuple (x,y) và numpy array
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            
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
        else:
            # Xử lý array (cho neural networks)
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            
            # Cập nhật biased first moment estimate
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
            
            # Cập nhật biased second moment estimate
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
            
            # Tính bias-corrected moment estimates
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            # Cập nhật tham số
            return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        """Reset moment estimates và timestep về 0."""
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None
        self.v = None
        self.t = 0


class AdamW(Optimizer):
    """
    Adam with decoupled weight decay (AdamW).

    Same moments as Adam, but applies weight decay directly to parameters:
        theta = theta - lr * ( m_hat / (sqrt(v_hat) + eps) ) - lr * weight_decay * theta
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = f"AdamW(lr={lr}, wd={weight_decay})"

        # moments
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, gradients):
        self.t += 1
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients

            # update moments
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (grad_x ** 2)
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * (grad_y ** 2)

            m_x_hat = self.m_x / (1 - self.beta1 ** self.t)
            m_y_hat = self.m_y / (1 - self.beta1 ** self.t)
            v_x_hat = self.v_x / (1 - self.beta2 ** self.t)
            v_y_hat = self.v_y / (1 - self.beta2 ** self.t)

            # Adam step
            step_x = self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
            step_y = self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)

            # Decoupled weight decay
            x = x - self.lr * self.weight_decay * x
            y = y - self.lr * self.weight_decay * y

            new_x = x - step_x
            new_y = y - step_y
            return new_x, new_y
        else:
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # Decoupled weight decay
            params = params - self.lr * self.weight_decay * params
            return params - step

    def reset(self):
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None
        self.v = None
        self.t = 0


class AMSGrad(Optimizer):
    """
    AMSGrad variant of Adam: uses maximum of past second-moment estimates (v_hat)
    to ensure non-increasing effective step sizes.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = f"AMSGrad(lr={lr})"

        # moments and max trackers (tuple mode)
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.vhat_max_x = 0.0
        self.vhat_max_y = 0.0

        # array mode states
        self.m = None
        self.v = None
        self.vhat_max = None

        self.t = 0

    def step(self, params, gradients):
        self.t += 1
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (grad_x ** 2)
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * (grad_y ** 2)

            m_x_hat = self.m_x / (1 - self.beta1 ** self.t)
            m_y_hat = self.m_y / (1 - self.beta1 ** self.t)
            v_x_hat = self.v_x / (1 - self.beta2 ** self.t)
            v_y_hat = self.v_y / (1 - self.beta2 ** self.t)

            # Update running max of v_hat
            self.vhat_max_x = max(self.vhat_max_x, v_x_hat)
            self.vhat_max_y = max(self.vhat_max_y, v_y_hat)

            new_x = x - self.lr * m_x_hat / (np.sqrt(self.vhat_max_x) + self.epsilon)
            new_y = y - self.lr * m_y_hat / (np.sqrt(self.vhat_max_y) + self.epsilon)
            return new_x, new_y
        else:
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
                self.vhat_max = np.zeros_like(params)
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            self.vhat_max = np.maximum(self.vhat_max, v_hat)
            step = self.lr * m_hat / (np.sqrt(self.vhat_max) + self.epsilon)
            return params - step

    def reset(self):
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.vhat_max_x = 0.0
        self.vhat_max_y = 0.0
        self.m = None
        self.v = None
        self.vhat_max = None
        self.t = 0


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    
    SAM finds flatter minima by minimizing both the loss and the sharpness
    (worst-case loss in a neighborhood around the current point).
    
    Paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    (Foret et al., ICLR 2021)
    
    NOTE: This base implementation is primarily for 2D function optimization.
    For neural network training, use SAMWrapper in pytorch_optimizers.py
    which properly handles the closure for computing adversarial gradients.
    
    Algorithm:
    1. Compute gradient at current point: g(θ)
    2. Take adversarial step: θ_adv = θ + ρ * ||g(θ)||_2 * g(θ) / ||g(θ)||_2
    3. Compute gradient at adversarial point: g(θ_adv)
    4. Take actual update step using g(θ_adv)
    """
    
    def __init__(self, lr=0.01, rho=0.05, base_optimizer='SGD', **base_kwargs):
        """
        Initialize SAM optimizer.
        
        Args:
            lr: Learning rate for the base optimizer
            rho: Neighborhood size (sharpness radius)
            base_optimizer: Base optimizer to wrap ('SGD', 'Adam', etc.)
            **base_kwargs: Keyword arguments for base optimizer
        """
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.base_optimizer_name = base_optimizer
        
        # Initialize base optimizer
        if base_optimizer == 'SGD':
            self.base_opt = SGD(lr=lr, **base_kwargs)
        elif base_optimizer == 'SGDMomentum':
            self.base_opt = SGDMomentum(lr=lr, **base_kwargs)
        elif base_optimizer == 'Adam':
            self.base_opt = Adam(lr=lr, **base_kwargs)
        elif base_optimizer == 'AdamW':
            self.base_opt = AdamW(lr=lr, **base_kwargs)
        elif base_optimizer == 'RMSProp':
            self.base_opt = RMSProp(lr=lr, **base_kwargs)
        else:
            raise ValueError(f"Unsupported base optimizer: {base_optimizer}")
            
        self.name = f"SAM({base_optimizer}, lr={lr}, rho={rho})"
        
        # SAM-specific state
        self.perturbation_x = 0.0
        self.perturbation_y = 0.0
        self.perturbation = None
    
    def _compute_adversarial_step(self, params, gradients):
        """
        Compute the adversarial step for SAM.
        
        Args:
            params: Current parameters
            gradients: Current gradients
            
        Returns:
            Adversarial parameters (perturbed point)
        """
        if isinstance(params, tuple):
            # 2D case
            x, y = params
            grad_x, grad_y = gradients
            
            # Compute gradient norm
            grad_norm = np.sqrt(grad_x**2 + grad_y**2)
            if grad_norm < 1e-12:
                return params
                
            # Normalize gradient direction
            grad_dir_x = grad_x / grad_norm
            grad_dir_y = grad_y / grad_norm
            
            # Adversarial step: θ + ρ * (g / ||g||)
            adv_x = x + self.rho * grad_dir_x
            adv_y = y + self.rho * grad_dir_y
            
            # Store perturbation for later use
            self.perturbation_x = self.rho * grad_dir_x
            self.perturbation_y = self.rho * grad_dir_y
            
            return adv_x, adv_y
        else:
            # Array case (neural networks)
            grad_norm = np.linalg.norm(gradients)
            if grad_norm < 1e-12:
                return params
                
            # Normalize gradient direction
            grad_dir = gradients / grad_norm
            
            # Adversarial step
            adv_params = params + self.rho * grad_dir
            
            # Store perturbation
            self.perturbation = self.rho * grad_dir
            
            return adv_params
    
    def step(self, params, gradients, loss_fn=None, adversarial_gradients=None):
        """
        Perform SAM update step.
        
        Args:
            params: Current parameters
            gradients: Gradients at current parameters
            loss_fn: Loss function (needed for 2D case to compute adversarial gradients)
            adversarial_gradients: Pre-computed gradients at adversarial point (optional)
            
        Returns:
            Updated parameters
        """
        if adversarial_gradients is not None:
            # Use pre-computed adversarial gradients (for PyTorch integration)
            return self.base_opt.step(params, adversarial_gradients)
        elif loss_fn is not None:
            # Compute adversarial gradients for 2D case
            adv_params = self._compute_adversarial_step(params, gradients)
            adv_gradients = loss_fn(adv_params)  # loss_fn should return gradients
            return self.base_opt.step(params, adv_gradients)
        else:
            # Fallback for backward compatibility (not correct SAM)
            print("Warning: SAM without adversarial gradients - using base optimizer only")
            return self.base_opt.step(params, gradients)
    
    def reset(self):
        """Reset optimizer state."""
        self.base_opt.reset()
        self.perturbation_x = 0.0
        self.perturbation_y = 0.0
        self.perturbation = None


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    
    Lookahead maintains two sets of weights: slow weights (for stability) 
    and fast weights (for exploration). The fast weights are updated normally,
    while slow weights follow the fast weights with a delay.
    
    Paper: "Lookahead Optimizer: k steps forward, 1 step back"
    (Zhang et al., NeurIPS 2019)
    """
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        """
        Initialize Lookahead wrapper.
        
        Args:
            base_optimizer: Base optimizer instance to wrap
            k: Number of fast steps before slow update
            alpha: Interpolation factor between slow and fast weights
        """
        super().__init__()
        self.base_opt = base_optimizer
        self.k = k
        self.alpha = alpha
        self.name = f"Lookahead({base_optimizer.name}, k={k}, alpha={alpha})"
        
        # Warning about adaptive optimizers
        if 'Adam' in base_optimizer.name or 'RMSProp' in base_optimizer.name:
            print(f"⚠️  WARNING: Lookahead with {base_optimizer.name} may interfere with internal optimizer state (running averages).")
            print("   Consider using Lookahead only with SGD for reliable behavior.")
            print("   This is mentioned in the thesis for educational purposes but not recommended for production use.")
        
        # State
        self.step_count = 0
        self.slow_params_x = None
        self.slow_params_y = None
        self.slow_params = None
        
    def _initialize_slow_weights(self, params):
        """Initialize slow weights to match current parameters."""
        if isinstance(params, tuple):
            self.slow_params_x, self.slow_params_y = params
        else:
            self.slow_params = params.copy()
    
    def _update_slow_weights(self, params):
        """Update slow weights by interpolating with fast weights."""
        if isinstance(params, tuple):
            x, y = params
            self.slow_params_x = self.alpha * self.slow_params_x + (1 - self.alpha) * x
            self.slow_params_y = self.alpha * self.slow_params_y + (1 - self.alpha) * y
            return self.slow_params_x, self.slow_params_y
        else:
            self.slow_params = self.alpha * self.slow_params + (1 - self.alpha) * params
            return self.slow_params
    
    def step(self, params, gradients):
        """
        Perform Lookahead update.
        
        Args:
            params: Current parameters (fast weights)
            gradients: Gradients
            
        Returns:
            Updated parameters (slow weights after k steps, fast weights otherwise)
        """
        # Initialize slow weights if needed
        if self.slow_params_x is None and isinstance(params, tuple):
            self._initialize_slow_weights(params)
        elif self.slow_params is None:
            self._initialize_slow_weights(params)
        
        # Update fast weights with base optimizer
        fast_params = self.base_opt.step(params, gradients)
        
        # Increment step counter
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            return self._update_slow_weights(fast_params)
        else:
            return fast_params
    
    def reset(self):
        """Reset optimizer state."""
        self.base_opt.reset()
        self.step_count = 0
        self.slow_params_x = None
        self.slow_params_y = None
        self.slow_params = None

