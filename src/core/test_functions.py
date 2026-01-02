"""
Module defining test functions for evaluating optimization algorithms.
"""

import numpy as np
from typing import Tuple


class TestFunction:
    """Base class for 2D test functions."""
    __test__ = False

    def __init__(self) -> None:
        pass

    def compute(self, x: float, y: float) -> float:
        """Compute function value at point (x, y)."""
        raise NotImplementedError("The compute method must be implemented in subclass")

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Compute gradient of function at point (x, y). Returns (grad_x, grad_y)."""
        raise NotImplementedError("The gradient method must be implemented in subclass")

    def hessian(self, x: float, y: float) -> np.ndarray:
        """Compute 2x2 Hessian matrix at point (x, y)."""
        raise NotImplementedError("The hessian method must be implemented in subclass")

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return plotting bounds as ((x_min, x_max), (y_min, y_max))."""
        raise NotImplementedError("The get_bounds method must be implemented in subclass")


class Rosenbrock(TestFunction):
    """
    Rosenbrock function: f(x,y) = (a - x)^2 + b(y - x^2)^2
    
    This is a classic test function with a narrow valley.
    Global minimum at (a, a^2) with value 0.
    """
    
    def __init__(self, a: float = 1, b: float = 100) -> None:
        """
        Initialize Rosenbrock function.
        
        Args:
            a: Parameter a (default: 1)
            b: Parameter b (default: 100)
        """
        super().__init__()
        self.a = a
        self.b = b
        self.name = f"Rosenbrock(a={a}, b={b})"
    
    def compute(self, x: float, y: float) -> float:
        """Compute value of Rosenbrock function."""
        return float((self.a - x)**2 + self.b * (y - x**2)**2)
    
    def gradient(self, x: float, y: float, noise_std: float = 0.0, noise_type: str = 'additive', batch_size: int = 1) -> Tuple[float, float]:
        """
        Compute analytical gradient of Rosenbrock function with optional stochastic noise.
        
        df/dx = -2(a - x) - 4bx(y - x^2)
        df/dy = 2b(y - x^2)
        
        Args:
            x, y: Point at which to compute gradient
            noise_std: Base standard deviation of gradient noise (0 = deterministic)
            noise_type: 'additive' (Gaussian) or 'multiplicative' (scales with gradient magnitude)
            batch_size: Simulated batch size (noise variance scales as 1/batch_size)
        
        Note: Adding noise simulates Stochastic Gradient Descent (SGD) behavior.
              Without noise, this is standard Gradient Descent (GD).
              
        Scientific Note on Batch Size:
              - In real SGD, gradient variance \u221d 1/B where B is batch size
              - actual_noise_std = noise_std / sqrt(batch_size)
              - This allows studying batch size effects on convergence and saddle escape
        """
        grad_x = float(-2 * (self.a - x) - 4 * self.b * x * (y - x**2))
        grad_y = float(2 * self.b * (y - x**2))
        
        if noise_std > 0:
            # Scale noise by batch size: variance \u221d 1/B
            actual_noise_std = noise_std / np.sqrt(batch_size)
            
            if noise_type == 'multiplicative':
                # Multiplicative noise: noise vanishes at stationary points (mimics real SGD)
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                noise_scale = actual_noise_std * grad_norm
                noise_x = np.random.normal(0, noise_scale)
                noise_y = np.random.normal(0, noise_scale)
            else:  # additive
                # Additive Gaussian noise (simple but less realistic)
                noise_x = np.random.normal(0, actual_noise_std)
                noise_y = np.random.normal(0, actual_noise_std)
            
            grad_x += noise_x
            grad_y += noise_y
        
        return grad_x, grad_y
    
    def hessian(self, x: float, y: float) -> np.ndarray:
        """
        Compute Hessian matrix of Rosenbrock function.
        
        d²f/dx² = 2 - 4b(y - 3x^2)
        d²f/dxdy = -4bx
        d²f/dy² = 2b
        """
        h_xx = 2 - 4 * self.b * (y - 3 * x**2)
        h_xy = -4 * self.b * x
        h_yy = 2 * self.b
        return np.array([[h_xx, h_xy], [h_xy, h_yy]])
    
    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return plotting bounds for Rosenbrock function."""
        return (-2, 2), (-1, 3)


class IllConditionedQuadratic(TestFunction):
    """
    Ill-conditioned Quadratic function: f(x,y) = 0.5 * (kappa * x^2 + y^2)
    
    This is a simple quadratic function with controlled condition number.
    Global minimum at (0, 0) with value 0.
    
    NOTE: Neural networks typically have condition numbers in range 1000-100000.
          Default kappa=100 is "easy mode" and may not reflect real optimization difficulty.
          Use kappa >= 1000 for realistic experiments.
    """
    
    def __init__(self, kappa=100):
        """
        Initialize Ill-conditioned Quadratic function.
        
        Args:
            kappa: Condition number - ratio between axes (default: 100)
                   Recommended: kappa=1000 or kappa=10000 for realistic NN simulation
        """
        super().__init__()
        self.kappa = kappa
        self.name = f"IllConditionedQuadratic(kappa={kappa})"
    
    def compute(self, x, y):
        """Compute value of ill-conditioned quadratic function."""
        return 0.5 * (self.kappa * x**2 + y**2)
    
    def gradient(self, x, y):
        """
        Compute gradient of ill-conditioned quadratic function.
        
        df/dx = kappa * x
        df/dy = y
        """
        grad_x = self.kappa * x
        grad_y = y
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """
        Compute Hessian matrix of ill-conditioned quadratic function.
        
        Hessian is diagonal matrix with elements [kappa, 1].
        """
        return np.array([[self.kappa, 0], [0, 1]])
    
    def get_bounds(self):
        """Return plotting bounds for ill-conditioned quadratic function."""
        scale = max(1, np.sqrt(self.kappa) / 10)
        return (-scale, scale), (-scale * np.sqrt(self.kappa), scale * np.sqrt(self.kappa))


class SaddlePoint(TestFunction):
    """
    Saddle Point function: f(x,y) = 0.5 * (x^2 - y^2)
    
    This function has a saddle point at the origin.
    No global minimum (function is unbounded below).
    """
    
    def __init__(self):
        """Initialize Saddle Point function."""
        super().__init__()
        self.name = "SaddlePoint"
    
    def compute(self, x, y):
        """Compute value of Saddle Point function."""
        return 0.5 * (x**2 - y**2)
    
    def gradient(self, x, y, noise_std: float = 0.0, noise_type: str = 'additive', batch_size: int = 1):
        """
        Compute gradient of Saddle Point function with optional stochastic noise.
        
        df/dx = x
        df/dy = -y
        
        Args:
            noise_std: Base standard deviation of gradient noise (0 = deterministic GD)
            noise_type: 'additive' or 'multiplicative'
            batch_size: Simulated batch size (noise variance scales as 1/batch_size)
        """
        grad_x = x
        grad_y = -y
        
        # Add stochastic noise for SGD simulation
        if noise_std > 0:
            # Scale noise by batch size
            actual_noise_std = noise_std / np.sqrt(batch_size)
            
            if noise_type == 'multiplicative':
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                if grad_norm > 0:  # Avoid division by zero at stationary point
                    noise_scale = actual_noise_std * grad_norm
                    grad_x += np.random.normal(0, noise_scale)
                    grad_y += np.random.normal(0, noise_scale)
            else:  # additive
                grad_x += np.random.normal(0, actual_noise_std)
                grad_y += np.random.normal(0, actual_noise_std)
        
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """
        Compute Hessian matrix of Saddle Point function.
        
        Hessian is diagonal matrix with elements [1, -1].
        """
        return np.array([[1, 0], [0, -1]])
    
    def get_bounds(self):
        """Return plotting bounds for Saddle Point function."""
        return (-2, 2), (-2, 2)


class Ackley2D(TestFunction):
    """
    Ackley function (2D):
        f(x, y) = -a * exp(-b * sqrt(0.5 * (x^2 + y^2)))
                  - exp(0.5 * (cos(c x) + cos(c y))) + a + e

    Default: a=20, b=0.2, c=2π. Global minimum at (0,0) with f=0.
    """

    def __init__(self, a=20.0, b=0.2, c=2 * np.pi):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.name = "Ackley2D"

    def compute(self, x, y):
        x = float(x)
        y = float(y)
        r = np.sqrt(0.5 * (x * x + y * y))
        term1 = -self.a * np.exp(-self.b * r)
        term2 = -np.exp(0.5 * (np.cos(self.c * x) + np.cos(self.c * y)))
        return term1 + term2 + self.a + np.e

    def gradient(self, x, y, noise_std: float = 0.0, noise_type: str = 'additive', batch_size: int = 1):
        """Compute gradient with optional stochastic noise for SGD simulation.
        
        Args:
            noise_std: Base standard deviation of gradient noise
            noise_type: 'additive' or 'multiplicative'
            batch_size: Simulated batch size (noise variance scales as 1/batch_size)
        """
        x = float(x)
        y = float(y)
        r = np.sqrt(0.5 * (x * x + y * y))
        if r == 0.0:
            d1x = 0.0
            d1y = 0.0
        else:
            common = self.a * self.b * np.exp(-self.b * r) / (2.0 * r)
            d1x = common * x
            d1y = common * y
        exp2 = np.exp(0.5 * (np.cos(self.c * x) + np.cos(self.c * y)))
        d2x = 0.5 * self.c * np.sin(self.c * x) * exp2
        d2y = 0.5 * self.c * np.sin(self.c * y) * exp2
        
        grad_x = d1x + d2x
        grad_y = d1y + d2y
        
        # Add stochastic noise for SGD simulation
        if noise_std > 0:
            # Scale noise by batch size: variance ∝ 1/B
            actual_noise_std = noise_std / np.sqrt(batch_size)
            
            if noise_type == 'multiplicative':
                grad_norm = np.sqrt(grad_x**2 + grad_y**2)
                noise_scale = actual_noise_std * grad_norm
                grad_x += np.random.normal(0, noise_scale)
                grad_y += np.random.normal(0, noise_scale)
            else:  # additive
                grad_x += np.random.normal(0, actual_noise_std)
                grad_y += np.random.normal(0, actual_noise_std)
        
        return grad_x, grad_y

    def hessian(self, x, y):
        # Numerical Hessian (central difference)
        x = float(x)
        y = float(y)
        eps = 1e-4
        # Removed unused: gx, gy = self.gradient(x, y)
        gx_xp, _ = self.gradient(x + eps, y)
        gx_xm, _ = self.gradient(x - eps, y)
        gx_yp, _ = self.gradient(x, y + eps)
        gx_ym, _ = self.gradient(x, y - eps)
        _, gy_xp = self.gradient(x + eps, y)
        _, gy_xm = self.gradient(x - eps, y)
        _, gy_yp = self.gradient(x, y + eps)
        _, gy_ym = self.gradient(x, y - eps)
        f_xx = (gx_xp - gx_xm) / (2 * eps)
        f_xy = (gx_yp - gx_ym) / (2 * eps)
        f_yx = (gy_xp - gy_xm) / (2 * eps)
        f_yy = (gy_yp - gy_ym) / (2 * eps)
        return np.array([[f_xx, 0.5 * (f_xy + f_yx)], [0.5 * (f_xy + f_yx), f_yy]], dtype=float)

    def get_bounds(self):
        return ((-5, 5), (-5, 5))


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


class BealeFunction(TestFunction):
    """
    Beale function - ill-conditioned with narrow curved valley.
    
    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    
    Global minimum: f(3, 0.5) = 0
    Search domain: x, y in [-4.5, 4.5]
    
    This function tests optimizer's ability to navigate tight curvatures
    and avoid getting stuck in the narrow valley.
    
    Reference:
        "Test functions for optimization" - Beale (1958)
    """
    
    def compute(self, x, y):
        """Compute Beale function value at (x, y)."""
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3
    
    def gradient(self, x, y):
        """
        Compute gradient of Beale function.
        
        df/dx = 2*(1.5 - x + xy) * (-1 + y) + 2*(2.25 - x + xy^2) * (-1 + y^2) 
                + 2*(2.625 - x + xy^3) * (-1 + y^3)
        df/dy = 2*(1.5 - x + xy) * x + 2*(2.25 - x + xy^2) * 2xy 
                + 2*(2.625 - x + xy^3) * 3xy^2
        """
        term1 = 1.5 - x + x*y
        term2 = 2.25 - x + x*y**2
        term3 = 2.625 - x + x*y**3
        
        grad_x = (2*term1*(-1 + y) + 
                  2*term2*(-1 + y**2) + 
                  2*term3*(-1 + y**3))
        
        grad_y = (2*term1*x + 
                  2*term2*2*x*y + 
                  2*term3*3*x*y**2)
        
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """Compute Hessian matrix at (x, y)."""
        # For exact Hessian computation
        term1 = 1.5 - x + x*y
        term2 = 2.25 - x + x*y**2
        term3 = 2.625 - x + x*y**3
        
        # Second derivatives
        d2f_dx2 = 2*(-1 + y)**2 + 2*(-1 + y**2)**2 + 2*(-1 + y**3)**2
        
        d2f_dy2 = (2*x**2 + 
                   2*term2*2*x + 2*(2*x*y)**2 + 
                   2*term3*6*x*y + 2*(3*x*y**2)**2)
        
        d2f_dxdy = (2*(-1 + y) + 2*term1 + 
                    2*(-1 + y**2)*2*x*y + 2*term2*2*y + 
                    2*(-1 + y**3)*3*x*y**2 + 2*term3*3*y**2)
        
        return np.array([[d2f_dx2, d2f_dxdy],
                         [d2f_dxdy, d2f_dy2]])
    
    def get_bounds(self):
        """Return search bounds for Beale function."""
        return ((-4.5, 4.5), (-4.5, 4.5))


class StyblinskiTang(TestFunction):
    """
    Styblinski-Tang function - highly multi-modal with many weak local minima.
    
    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
    
    For 2D: Global minimum: f(-2.903534, -2.903534) ≈ -78.332
    Search domain: x, y in [-5, 5]
    
    This function tests optimizer's global exploration vs local exploitation.
    Contains many local minima that can trap gradient-based optimizers.
    
    Reference:
        Styblinski, M. A., & Tang, T.-S. (1990). Experiments in nonconvex optimization.
    """
    
    def compute(self, x, y):
        """Compute Styblinski-Tang function value at (x, y)."""
        return 0.5 * ((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y))
    
    def gradient(self, x, y):
        """
        Compute gradient of Styblinski-Tang function.
        
        df/dx_i = 0.5 * (4*x_i^3 - 32*x_i + 5)
        """
        grad_x = 0.5 * (4*x**3 - 32*x + 5)
        grad_y = 0.5 * (4*y**3 - 32*y + 5)
        return grad_x, grad_y
    
    def hessian(self, x, y):
        """Compute Hessian matrix at (x, y)."""
        d2f_dx2 = 0.5 * (12*x**2 - 32)
        d2f_dy2 = 0.5 * (12*y**2 - 32)
        d2f_dxdy = 0.0  # No cross terms
        
        return np.array([[d2f_dx2, d2f_dxdy],
                         [d2f_dxdy, d2f_dy2]])
    
    def get_bounds(self):
        """Return search bounds for Styblinski-Tang function."""
        return ((-5, 5), (-5, 5))
