"""                                  
Module defining optimization algorithms (optimizers).
"""

import numpy as np
import logging
from typing import Tuple, Union, Any, Callable


class Optimizer:
    """Base class for optimization algorithms."""

    def __init__(self) -> None:
        """Initialize optimizer."""
        # History of parameters (for 2D / tuple case store list of (x,y), for arrays store copies)
        self.history_params: list = []

    def _append_history(self, params: Any) -> None:
        """Append parameters to history in a safe, copy-on-write manner."""
        try:
            if isinstance(params, tuple):
                x, y = params
                self.history_params.append((float(x), float(y)))
            else:
                # For array-like params, store a copy
                self.history_params.append(np.array(params, copy=True))
        except (TypeError, ValueError, AttributeError):
            # Never raise during logging of history
            try:
                self.history_params.append(params)
            except (TypeError, ValueError, AttributeError):
                pass

    def step(self, params: Union[Tuple[float, float], Any], gradients: Union[Tuple[float, float], Any], **kwargs: Any) -> Union[Tuple[float, float], Any]:
        """
        Perform one parameter update step.

        Args:
            params: Tuple (x, y) - current parameters
            gradients: Tuple (grad_x, grad_y) - gradient at current parameters
            **kwargs: Additional optimizer-specific arguments (e.g., loss_fn for SAM)

        Returns:
            Tuple (new_x, new_y) - parameters after update

        Note:
            Subclasses may extend the signature with optimizer-specific kwargs.
            This allows SAM to accept `loss_fn` and `adversarial_gradients`
            without violating the Liskov Substitution Principle.
        """
        raise NotImplementedError("The step method must be implemented in subclass")

    def set_lr(self, lr: float) -> None:
        """
        Update learning rate (for scheduler compatibility).

        This enables learning rate scheduling in 2D optimization experiments,
        matching the scheduler support in PyTorch neural network training.

        Args:
            lr: New learning rate value

        Note:
            This method allows simulating Cosine Annealing, OneCycle, etc.
            in 2D test function experiments to maintain consistency with
            neural network training experiments.
        """
        if hasattr(self, 'lr'):
            self.lr = lr
        else:
            logging.warning("%s does not have 'lr' attribute", self.__class__.__name__)

    def get_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate, or 0.0 if not defined
        """
        return getattr(self, 'lr', 0.0)

    def reset(self) -> None:
        """Reset internal optimizer state."""
        # Default: no state to reset
        self.history_params = []

    def _dispatch_step(
        self,
        params: Union[Tuple[float, float], np.ndarray],
        gradients: Union[Tuple[float, float], np.ndarray],
        tuple_handler: Callable[[Tuple[float, float], Tuple[float, float]], Tuple[float, float]],
        array_handler: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> Union[Tuple[float, float], np.ndarray]:
        """
        Generic dispatcher for tuple vs array params (M1: Code Deduplication).
        
        This helper method eliminates boilerplate in optimizer step() implementations
        by providing a single dispatch point for handling both tuple (x, y) and
        array-based parameter formats.
        
        Args:
            params: Parameters (tuple or array)
            gradients: Gradients (tuple or array)
            tuple_handler: Callable to handle tuple case (x, y) -> (new_x, new_y)
            array_handler: Callable to handle array case
            
        Returns:
            Updated parameters in same format as input
            
        Example:
            def step(self, params, gradients, **kwargs):
                return self._dispatch_step(
                    params, gradients,
                    self._step_tuple,
                    self._step_array
                )
        """
        if isinstance(params, tuple):
            return tuple_handler(params, gradients)
        else:
            return array_handler(params, gradients)


class SGD(Optimizer):
    """
    Basic Stochastic Gradient Descent (SGD) with optional L2 regularization.

    Update formula: θ_new = θ_old - lr * gradient - lr * weight_decay * θ_old

    Note: The weight_decay term here is classic L2 regularization applied to the
    gradients. This matches standard SGD weight decay but differs from the
    decoupled weight decay strategy used in AdamW.
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0) -> None:
        """
        Initialize SGD optimizer.

        Args:
            lr: Learning rate (default: 0.01)
            weight_decay: L2 regularization coefficient (default: 0.0). This is
                         applied directly to the gradient term, not decoupled.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        if weight_decay > 0:
            self.name = f"SGD(lr={lr}, wd={weight_decay})"
        else:
            self.name = f"SGD(lr={lr})"

    def step(self, params: Union[Tuple[float, float], Any], gradients: Union[Tuple[float, float], Any], **kwargs: Any) -> Union[Tuple[float, float], Any]:
        """Perform one SGD step with optional weight decay.
        
        Returns:
            Updated parameters (same type as input)
        """
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)
    
    def _step_tuple(self, params: Tuple[float, float], gradients: Tuple[float, float]) -> Tuple[float, float]:
        """SGD step for tuple (x, y) parameters."""
        x, y = params
        grad_x, grad_y = gradients

        # Apply weight decay (L2 regularization)
        if self.weight_decay > 0:
            grad_x += self.weight_decay * x
            grad_y += self.weight_decay * y

        new_x = x - self.lr * grad_x
        new_y = y - self.lr * grad_y
        self._append_history((new_x, new_y))
        return new_x, new_y
    
    def _step_array(self, params: Any, gradients: Any) -> Any:
        """SGD step for array parameters."""
        # Handle array (for neural networks)
        effective_grad = np.array(gradients) if isinstance(gradients, (tuple, list)) else gradients.copy()

        # Apply weight decay (L2 regularization)
        if self.weight_decay > 0:
            effective_grad += self.weight_decay * params

        updated = params - self.lr * effective_grad
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        """SGD has no internal state - no action needed."""
        # Stateless optimizer - no state to reset


class SGDMomentum(Optimizer):
    """
    SGD with Momentum with optional L2 regularization.

    Update formula:
        v_new = beta * v_old + gradient + weight_decay * params  # L2 regularization
        θ_new = θ_old - lr * v_new

    Note: This uses coupled L2 weight decay (applied to gradient), matching
    standard PyTorch SGD behavior. This differs from decoupled weight decay
    used in AdamW.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0) -> None:
        """
        Initialize SGD with Momentum optimizer.

        Args:
            lr: Learning rate (default: 0.01)
            beta: Momentum coefficient (default: 0.9)
            weight_decay: L2 regularization coefficient (default: 0.0)
        """
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        if weight_decay > 0:
            self.name = f"SGDMomentum(lr={lr}, beta={beta}, wd={weight_decay})"
        else:
            self.name = f"SGDMomentum(lr={lr}, beta={beta})"

        # Initialize velocity
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None  # For neural networks

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one SGD with Momentum step with optional weight decay."""
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Apply weight decay (L2 regularization) to gradients
        if self.weight_decay > 0:
            grad_x += self.weight_decay * x
            grad_y += self.weight_decay * y

        # Update velocity
        self.v_x = self.beta * self.v_x + grad_x
        self.v_y = self.beta * self.v_y + grad_y

        # Update parameters
        new_x = x - self.lr * self.v_x
        new_y = y - self.lr * self.v_y
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        # Initialize velocity on first call
        if self.v is None:
            self.v = np.zeros_like(params)

        # Skip updates when gradients contain NaN or Inf
        if not np.isfinite(gradients).all():
            logging.warning("SGDMomentum: Non-finite gradients detected, skipping update")
            return params

        # Apply weight decay (L2 regularization)
        grad_array = np.asarray(gradients)
        effective_grad = grad_array.copy()
        if self.weight_decay > 0:
            effective_grad += self.weight_decay * params

        # Update velocity
        self.v = self.beta * self.v + effective_grad

        # Update parameters
        updated = params - self.lr * self.v
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        """Reset velocity to 0."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None
        super().reset()

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            'v_x': self.v_x,
            'v_y': self.v_y,
            'v': self.v.copy() if self.v is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")
        
        try:
            self.v_x = float(state_dict.get('v_x', 0.0))
            self.v_y = float(state_dict.get('v_y', 0.0))
            v_state = state_dict.get('v')
            self.v = np.array(v_state, dtype=np.float32) if v_state is not None else None
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid state_dict values: {e}") from e


class SGDNesterov(Optimizer):
    """
    SGD with Nesterov Accelerated Gradient (NAG).

    Update rule (PyTorch-style formulation using current gradient g_t):
        v_t = beta * v_{t-1} + g_t
        d_t = g_t + beta * v_t
        theta_new = theta_old - lr * d_t

    This approximates the lookahead gradient without requiring function access.
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.name = f"SGDNesterov(lr={lr}, beta={beta})"

        # State
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None  # array state for NN

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one Nesterov Accelerated Gradient step."""
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients
        
        # Update velocity
        self.v_x = self.beta * self.v_x + grad_x
        self.v_y = self.beta * self.v_y + grad_y
        
        # Nesterov accelerated gradient
        d_x = grad_x + self.beta * self.v_x
        d_y = grad_y + self.beta * self.v_y
        
        new_x = x - self.lr * d_x
        new_y = y - self.lr * d_y
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.v is None:
            self.v = np.zeros_like(params)
        elif self.v.shape != params.shape:
            logging.warning("SGDNesterov: Parameter shape changed from %s to %s. Resizing state.", self.v.shape, params.shape)
            self.v = np.zeros_like(params)
        
        self.v = self.beta * self.v + gradients
        d = gradients + self.beta * self.v
        updated = params - self.lr * d
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            'v_x': self.v_x,
            'v_y': self.v_y,
            'v': self.v.copy() if self.v is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        self.v_x = state_dict.get('v_x', 0.0)
        self.v_y = state_dict.get('v_y', 0.0)
        v_state = state_dict.get('v')
        self.v = np.array(v_state, dtype=np.float32) if v_state is not None else None


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation).

    Update formula:
        s_new = decay_rate * s_old + (1 - decay_rate) * gradient^2
        θ_new = θ_old - lr * gradient / sqrt(s_new + epsilon)
    """

    def __init__(self, lr: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8) -> None:
        """
        Initialize RMSProp optimizer.

        Args:
            lr: Learning rate (default: 0.01)
            decay_rate: Decay rate for moving average (default: 0.9)
            epsilon: Small constant to avoid division by zero (default: 1e-8)
        """
        super().__init__()
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.name = f"RMSProp(lr={lr}, decay={decay_rate})"

        # Initialize squared gradient accumulator
        self.s_x = 0.0
        self.s_y = 0.0
        self.s = None  # For neural networks

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one RMSProp step."""
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Update squared gradient accumulator
        self.s_x = self.decay_rate * self.s_x + (1 - self.decay_rate) * grad_x**2
        self.s_y = self.decay_rate * self.s_y + (1 - self.decay_rate) * grad_y**2

        # Update parameters with adaptive learning rate
        new_x = x - self.lr * grad_x / (np.sqrt(self.s_x) + self.epsilon)
        new_y = y - self.lr * grad_y / (np.sqrt(self.s_y) + self.epsilon)
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        # Initialize state on first call
        if self.s is None:
            self.s = np.zeros_like(params)
        elif self.s.shape != params.shape:
            logging.warning("RMSProp: Parameter shape changed from %s to %s. Resizing state.", self.s.shape, params.shape)
            self.s = np.zeros_like(params)

        # Update squared gradient accumulator
        grad_array = np.asarray(gradients)
        self.s = self.decay_rate * self.s + (1 - self.decay_rate) * grad_array**2

        # Update parameters with adaptive learning rate
        updated = params - self.lr * grad_array / (np.sqrt(self.s) + self.epsilon)
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        """Reset squared gradient accumulator to 0."""
        self.s_x = 0.0
        self.s_y = 0.0
        self.s = None

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            's_x': self.s_x,
            's_y': self.s_y,
            's': self.s.copy() if self.s is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")
        
        try:
            self.s_x = float(state_dict.get('s_x', 0.0))
            self.s_y = float(state_dict.get('s_y', 0.0))
            s_state = state_dict.get('s')
            self.s = np.array(s_state, dtype=np.float32) if s_state is not None else None
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid state_dict values: {e}") from e


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) with optional L2 regularization.

    CRITICAL NOTE: This implements L2 regularization (grad += wd * param),
    NOT decoupled weight decay. Use AdamW for decoupled weight decay.

    The L2 variant is included to demonstrate WHY AdamW was necessary:
    L2 regularization interacts poorly with adaptive learning rates.

    Update formula:
        grad_effective = grad + weight_decay * param  (L2 reg, if enabled)
        m_new = beta1 * m_old + (1 - beta1) * grad_effective
        v_new = beta2 * v_old + (1 - beta2) * grad_effective^2
        m_hat = m_new / (1 - beta1^t)
        v_hat = v_new / (1 - beta2^t)
        θ_new = θ_old - lr * m_hat / (sqrt(v_hat) + epsilon)

    For proper weight decay with Adam, use AdamW instead.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
        """
        Initialize Adam optimizer.

        Args:
            lr: Learning rate (default: 0.001)
            beta1: Decay coefficient for first moment (default: 0.9)
            beta2: Decay coefficient for second moment (default: 0.999)
            epsilon: Small constant to avoid division by zero (default: 1e-8)
            weight_decay: L2 regularization coefficient (default: 0.0)
                         WARNING: This is coupled L2, not decoupled decay.
                         Use AdamW for decoupled weight decay.
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        if weight_decay > 0:
            self.name = f"Adam(lr={lr}, L2_wd={weight_decay})"
        else:
            self.name = f"Adam(lr={lr})"

        # Initialize moment estimates
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None  # For neural networks
        self.v = None  # For neural networks

        # Timestep counter
        self.t = 0

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one Adam step with optional L2 regularization."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Apply L2 regularization (coupled to gradient, the 'wrong' way)
        if self.weight_decay > 0:
            grad_x = grad_x + self.weight_decay * x
            grad_y = grad_y + self.weight_decay * y

        # Update biased first moment estimate
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # Update biased second moment estimate
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x**2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y**2

        # Robust bias correction with timestep capping
        effective_t = min(self.t, 1_000_000)
        bias_correction1 = max(1.0 - self.beta1**effective_t, 1e-8)
        bias_correction2 = max(1.0 - self.beta2**effective_t, 1e-8)
        
        # Warn if bias correction approaches numerical limits
        if self.t <= 10 and (bias_correction1 < 1e-6 or bias_correction2 < 1e-6):
            logging.warning(
                "Adam: Bias correction approaching zero at t=%d (beta1=%.4f, beta2=%.4f). "
                "Consider beta values < 1.0 for numerical stability.",
                self.t, self.beta1, self.beta2
            )
        
        # Compute bias-corrected moment estimates
        m_x_hat = self.m_x / bias_correction1
        m_y_hat = self.m_y / bias_correction1
        v_x_hat = self.v_x / bias_correction2
        v_y_hat = self.v_y / bias_correction2

        # Update parameters
        new_x = x - self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
        new_y = y - self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        # Initialize state on first call
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        elif self.m.shape != params.shape:
            logging.warning("Adam: Parameter shape changed from %s to %s. Resizing state.", self.m.shape, params.shape)
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Skip updates when gradients contain NaN or Inf
        if not np.isfinite(gradients).all():
            logging.warning("Adam: Non-finite gradients detected, skipping update")
            return params

        # Apply L2 regularization (coupled to gradient)
        grad_array = np.asarray(gradients)
        if self.weight_decay > 0:
            grad_array = grad_array + self.weight_decay * params

        # Type safety check
        if self.m is None or self.v is None:
            raise TypeError("Optimizer state not initialized properly. This should not happen after initialization check.")
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array**2

        # Robust bias correction with adaptive timestep capping
        # Cap timestep based on beta values to prevent underflow
        # When (1 - beta^t) < 1e-6, bias correction becomes numerically unstable
        max_safe_t_beta1 = int(-np.log(1e-6) / max(np.log(self.beta1), 1e-10))
        max_safe_t_beta2 = int(-np.log(1e-6) / max(np.log(self.beta2), 1e-10))
        effective_t = min(self.t, max_safe_t_beta1, max_safe_t_beta2, 1_000_000)
        bias_correction1 = max(1.0 - self.beta1**effective_t, 1e-6)
        bias_correction2 = max(1.0 - self.beta2**effective_t, 1e-6)
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2

        # Update parameters
        updated = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        """Reset moment estimates and timestep to 0."""
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None
        self.v = None
        self.t = 0

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            'm_x': self.m_x,
            'm_y': self.m_y,
            'v_x': self.v_x,
            'v_y': self.v_y,
            'm': self.m.copy() if self.m is not None else None,
            'v': self.v.copy() if self.v is not None else None,
            't': self.t,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")
        
        try:
            self.m_x = float(state_dict.get('m_x', 0.0))
            self.m_y = float(state_dict.get('m_y', 0.0))
            self.v_x = float(state_dict.get('v_x', 0.0))
            self.v_y = float(state_dict.get('v_y', 0.0))
            m_state = state_dict.get('m')
            v_state = state_dict.get('v')
            self.m = np.array(m_state, dtype=np.float32) if m_state is not None else None
            self.v = np.array(v_state, dtype=np.float32) if v_state is not None else None
            self.t = int(state_dict.get('t', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid state_dict values: {e}") from e


class AdamW(Optimizer):
    """
    Adam with decoupled weight decay (AdamW).

    Same moments as Adam, but applies weight decay directly to parameters:
        theta = theta - lr * ( m_hat / (sqrt(v_hat) + eps) ) - lr * weight_decay * theta
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
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

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one AdamW step with decoupled weight decay."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Update moments
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (grad_x ** 2)
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * (grad_y ** 2)

        m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
        m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
        v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
        v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

        # Adam step (computed from original params)
        step_x = self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
        step_y = self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)

        # Decoupled weight decay: apply to original params
        new_x = x - step_x - self.lr * self.weight_decay * x
        new_y = y - step_y - self.lr * self.weight_decay * y
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        elif self.m.shape != params.shape:
            logging.warning("AdamW: Parameter shape changed from %s to %s. Resizing state.", self.m.shape, params.shape)
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Skip updates when gradients contain NaN or Inf
        if not np.isfinite(gradients).all():
            logging.warning("AdamW: Non-finite gradients detected, skipping update")
            return params

        # Type safety check
        if self.m is None or self.v is None:
            raise TypeError(
                "AdamW optimizer state not initialized properly. This should not happen after initialization check.\n"
                f"Debug info: m={type(self.m).__name__ if self.m is not None else 'None'}, "
                f"v={type(self.v).__name__ if self.v is not None else 'None'}, "
                f"params shape={getattr(params, 'shape', 'unknown')}"
            )
        
        grad_array = np.asarray(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_array ** 2)
        
        # Robust bias correction with adaptive timestep capping
        # Cap timestep based on beta values to prevent underflow
        # When (1 - beta^t) < 1e-6, bias correction becomes numerically unstable
        max_safe_t_beta1 = int(-np.log(1e-6) / max(np.log(self.beta1), 1e-10))
        max_safe_t_beta2 = int(-np.log(1e-6) / max(np.log(self.beta2), 1e-10))
        effective_t = min(self.t, max_safe_t_beta1, max_safe_t_beta2, 1_000_000)
        bias_correction1 = max(1.0 - self.beta1**effective_t, 1e-6)
        bias_correction2 = max(1.0 - self.beta2**effective_t, 1e-6)
        
        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Decoupled weight decay: apply to original params
        updated = params - step - self.lr * self.weight_decay * params
        self._append_history(updated)
        return updated

    def reset(self) -> None:
        self.m_x = 0.0
        self.m_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.m = None
        self.v = None
        self.t = 0

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            'm_x': self.m_x,
            'm_y': self.m_y,
            'v_x': self.v_x,
            'v_y': self.v_y,
            'm': self.m.copy() if self.m is not None else None,
            'v': self.v.copy() if self.v is not None else None,
            't': self.t,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        self.m_x = state_dict.get('m_x', 0.0)
        self.m_y = state_dict.get('m_y', 0.0)
        self.v_x = state_dict.get('v_x', 0.0)
        self.v_y = state_dict.get('v_y', 0.0)
        m_state = state_dict.get('m')
        v_state = state_dict.get('v')
        self.m = np.array(m_state, dtype=np.float32) if m_state is not None else None
        self.v = np.array(v_state, dtype=np.float32) if v_state is not None else None
        self.t = state_dict.get('t', 0)


class AMSGrad(Optimizer):
    """
    AMSGrad variant of Adam: uses maximum of past second-moment estimates (v_hat)
    to ensure non-increasing effective step sizes.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
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

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one AMSGrad step."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients
        
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * (grad_x ** 2)
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * (grad_y ** 2)

        m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
        m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
        v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
        v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

        # Update running max of v_hat
        self.vhat_max_x = max(self.vhat_max_x, v_x_hat)
        self.vhat_max_y = max(self.vhat_max_y, v_y_hat)

        new_x = x - self.lr * m_x_hat / (np.sqrt(self.vhat_max_x) + self.epsilon)
        new_y = y - self.lr * m_y_hat / (np.sqrt(self.vhat_max_y) + self.epsilon)
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.vhat_max = np.zeros_like(params)
        elif self.m.shape != params.shape:
            # Parameter shape change violates AMSGrad convergence guarantees
            raise RuntimeError(
                f"AMSGrad CRITICAL ERROR: Parameter shape changed during training from {self.m.shape} to {params.shape}. "
                f"Shape changes violate AMSGrad's convergence guarantees and indicate a bug in the model or training loop. "
                f"AMSGrad requires vhat_max to track the maximum across ALL updates - resetting it breaks the convergence proof. "
                f"ABORTING to prevent silent state corruption and invalid scientific results."
            )
        
        # Type safety check
        if self.m is None or self.v is None or self.vhat_max is None:
            raise TypeError("AMSGrad optimizer state not initialized properly.")
        
        grad_array = np.asarray(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_array ** 2)
        
        # Robust bias correction with adaptive timestep capping
        # Cap timestep based on beta values to prevent underflow
        # When (1 - beta^t) < 1e-6, bias correction becomes numerically unstable
        max_safe_t_beta1 = int(-np.log(1e-6) / max(np.log(self.beta1), 1e-10))
        max_safe_t_beta2 = int(-np.log(1e-6) / max(np.log(self.beta2), 1e-10))
        effective_t = min(self.t, max_safe_t_beta1, max_safe_t_beta2, 1_000_000)
        bias_correction1 = max(1.0 - self.beta1**effective_t, 1e-6)
        bias_correction2 = max(1.0 - self.beta2**effective_t, 1e-6)
        
        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2
        
        # AMSGrad: use maximum of biased second moment
        self.vhat_max = np.maximum(self.vhat_max, v_hat)
        step = self.lr * m_hat / (np.sqrt(self.vhat_max) + self.epsilon)
        updated = params - step
        self._append_history(updated)
        return updated

    def reset(self) -> None:
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

    def state_dict(self) -> dict:
        """Save optimizer state for checkpointing."""
        return {
            'm_x': self.m_x,
            'm_y': self.m_y,
            'v_x': self.v_x,
            'v_y': self.v_y,
            'vhat_max_x': self.vhat_max_x,
            'vhat_max_y': self.vhat_max_y,
            'm': self.m.copy() if self.m is not None else None,
            'v': self.v.copy() if self.v is not None else None,
            'vhat_max': self.vhat_max.copy() if self.vhat_max is not None else None,
            't': self.t,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state from checkpoint."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected dict, got {type(state_dict)}")
        
        try:
            self.m_x = float(state_dict.get('m_x', 0.0))
            self.m_y = float(state_dict.get('m_y', 0.0))
            self.v_x = float(state_dict.get('v_x', 0.0))
            self.v_y = float(state_dict.get('v_y', 0.0))
            self.vhat_max_x = float(state_dict.get('vhat_max_x', 0.0))
            self.vhat_max_y = float(state_dict.get('vhat_max_y', 0.0))
            m_state = state_dict.get('m')
            v_state = state_dict.get('v')
            vhat_max_state = state_dict.get('vhat_max')
            self.m = np.array(m_state, dtype=np.float32) if m_state is not None else None
            self.v = np.array(v_state, dtype=np.float32) if v_state is not None else None
            self.vhat_max = np.array(vhat_max_state, dtype=np.float32) if vhat_max_state is not None else None
            self.t = int(state_dict.get('t', 0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid state_dict values: {e}") from e


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
            # NUMERICAL STABILITY FIX: Use np.hypot to avoid overflow
            grad_norm = np.hypot(grad_x, grad_y)
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
            # NUMERICAL STABILITY FIX: Use safe norm computation to prevent overflow
            # For large arrays, np.linalg.norm can overflow before taking sqrt
            # Use np.sqrt(np.sum(gradients**2)) with overflow check, or scale first
            max_abs = np.max(np.abs(gradients))
            if max_abs < 1e-12:
                return params

            # Scale gradients to prevent overflow in norm computation
            scaled_grad = gradients / max_abs
            scaled_norm = np.linalg.norm(scaled_grad)
            grad_norm = scaled_norm * max_abs  # Actual norm

            if grad_norm < 1e-12:
                return params

            # Normalize gradient direction
            grad_dir = gradients / grad_norm

            # Adversarial step
            adv_params = params + self.rho * grad_dir

            # Store perturbation
            self.perturbation = self.rho * grad_dir

            return adv_params

    def step(self, params, gradients, loss_fn=None, adversarial_gradients=None, **kwargs) -> Union[Tuple[float, float], Any]:
        """
        Perform SAM update step.

        Args:
            params: Current parameters
            gradients: Gradients at current parameters
            loss_fn: Loss function (needed for 2D case to compute adversarial gradients)
            adversarial_gradients: Pre-computed gradients at adversarial point (optional)
            **kwargs: Additional arguments (unused, for signature compatibility)

        Returns:
            Updated parameters
            
        LOGIC REVIEW NOTE:
            SAM update must be from ORIGINAL parameters, not adversarial ones.
            The algorithm is:
            1. Compute g(θ) at current θ
            2. Compute adversarial point: θ_adv = θ + ρ*(g/||g||)
            3. Compute g(θ_adv) at adversarial point
            4. Update from ORIGINAL θ: θ_new = θ - lr*g(θ_adv)
        
        TYPE SAFETY FIX: Validate API contract - SAM requires either adversarial_gradients or loss_fn
        """
        if adversarial_gradients is not None:
            # LOGIC FIX: When adversarial_gradients are pre-computed and passed in,
            # the params argument is ALREADY at the original position (not perturbed).
            # SAM algorithm: θ_new = θ_original - lr * g(θ_adv)
            # The caller (e.g., SAMWrapper) computes g(θ_adv) and passes it here.
            # DO NOT subtract perturbation - params are already correct.
            return self.base_opt.step(params, adversarial_gradients)
        elif loss_fn is not None:
            # Compute adversarial gradients for 2D case
            adv_params = self._compute_adversarial_step(params, gradients)
            adv_gradients = loss_fn(adv_params)  # loss_fn should return gradients
            # Update from ORIGINAL params (not adv_params)
            return self.base_opt.step(params, adv_gradients)
        else:
            # TYPE SAFETY FIX: Fail-fast with clear error message for SAM API contract
            # SAM requires either pre-computed adversarial gradients (from a closure/second forward pass)
            # or a loss function to compute gradients at the adversarial point.
            raise ValueError(
                "SAM optimizer requires either 'adversarial_gradients' or 'loss_fn' parameter.\n"
                "For neural network training, use SAMWrapper from pytorch_optimizers.py.\n"
                "Example usage:\n"
                "  # With SAMWrapper (recommended for PyTorch):\n"
                "  optimizer = SAMWrapper(model.parameters(), base_optimizer='SGD', lr=0.1)\n"
                "  loss = optimizer.step(closure=lambda: loss_fn(model(data), targets))\n"
                "\n"
                "  # With custom optimizer (2D functions):\n"
                "  new_params = sam.step(params, grads, loss_fn=gradient_function)\n"
                "\n"
                "Silently falling back to base optimizer would disable SAM and break experiments."
            )

    def reset(self) -> None:
        """Reset optimizer state."""
        self.base_opt.reset()
        self.perturbation_x = 0.0
        self.perturbation_y = 0.0
        self.perturbation = None

    def state_dict(self) -> dict:
        """Save SAM and base optimizer state for checkpointing."""
        base_state = {}
        if hasattr(self.base_opt, 'state_dict'):
            base_state = self.base_opt.state_dict()
        return {
            'base_optimizer': base_state,
            'rho': self.rho,
            'base_optimizer_name': self.base_optimizer_name,
            'perturbation_x': self.perturbation_x,
            'perturbation_y': self.perturbation_y,
            'perturbation': self.perturbation.copy() if self.perturbation is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore SAM and base optimizer state from checkpoint."""
        if hasattr(self.base_opt, 'load_state_dict'):
            base_state = state_dict.get('base_optimizer', {})
            self.base_opt.load_state_dict(base_state)
        self.rho = state_dict.get('rho', self.rho)
        self.perturbation_x = state_dict.get('perturbation_x', 0.0)
        self.perturbation_y = state_dict.get('perturbation_y', 0.0)
        pert_state = state_dict.get('perturbation')
        self.perturbation = np.array(pert_state, dtype=np.float32) if pert_state is not None else None


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

        # Note: Lookahead was tested with Adam in the original paper and works well.
        # However, slow weights don't accumulate momentum/adaptive state, which may
        # affect convergence speed. Tune k and alpha for best results.
        if 'Adam' in base_optimizer.name or 'RMSProp' in base_optimizer.name:
            logging.info(
                "Lookahead wrapping %s: slow weights updated every k=%d steps with alpha=%.2f. "
                "Note: slow weights don't benefit from adaptive LR. Tune k/alpha if needed.",
                base_optimizer.name, self.k, self.alpha
            )

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
        """Update slow weights by interpolating with fast weights.

        Per Lookahead paper: slow = slow + alpha * (fast - slow)
        Which equals: slow = (1 - alpha) * slow + alpha * fast
        """
        if isinstance(params, tuple):
            x, y = params
            # Ensure slow params initialized
            assert self.slow_params_x is not None and self.slow_params_y is not None, "Slow params must be initialized before update"
            # Lookahead: slow += alpha * (fast - slow)
            self.slow_params_x = (1 - self.alpha) * self.slow_params_x + self.alpha * x
            self.slow_params_y = (1 - self.alpha) * self.slow_params_y + self.alpha * y
            return self.slow_params_x, self.slow_params_y
        else:
            assert self.slow_params is not None, "Slow params must be initialized before update"
            self.slow_params = (1 - self.alpha) * self.slow_params + self.alpha * params
            return self.slow_params

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """
        Perform Lookahead update.

        Args:
            params: Current parameters (fast weights)
            gradients: Gradients

        Returns:
            Updated parameters (slow weights after k steps, fast weights otherwise)
        """
        # Initialize slow weights if needed
        if isinstance(params, tuple):
            if self.slow_params_x is None or self.slow_params_y is None:
                self._initialize_slow_weights(params)
        else:
            if self.slow_params is None:
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

    def reset(self) -> None:
        """Reset optimizer state."""
        self.base_opt.reset()
        self.step_count = 0
        self.slow_params_x = None
        self.slow_params_y = None
        self.slow_params = None

    def state_dict(self) -> dict:
        """Save Lookahead and base optimizer state for checkpointing."""
        base_state = {}
        if hasattr(self.base_opt, 'state_dict'):
            base_state = self.base_opt.state_dict()
        return {
            'base_optimizer': base_state,
            'k': self.k,
            'alpha': self.alpha,
            'step_count': self.step_count,
            'slow_params_x': self.slow_params_x,
            'slow_params_y': self.slow_params_y,
            'slow_params': self.slow_params.copy() if self.slow_params is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore Lookahead and base optimizer state from checkpoint."""
        if hasattr(self.base_opt, 'load_state_dict'):
            base_state = state_dict.get('base_optimizer', {})
            self.base_opt.load_state_dict(base_state)
        self.k = state_dict.get('k', self.k)
        self.alpha = state_dict.get('alpha', self.alpha)
        self.step_count = state_dict.get('step_count', 0)
        self.slow_params_x = state_dict.get('slow_params_x')
        self.slow_params_y = state_dict.get('slow_params_y')
        slow_state = state_dict.get('slow_params')
        self.slow_params = np.array(slow_state, dtype=np.float32) if slow_state is not None else None


class AdaBound(Optimizer):
    """
    AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate.

    Combines benefits of adaptive methods and SGD by dynamically bounding the learning rate.
    Reference: https://arxiv.org/abs/1902.09843

    Formula:
        m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

        lr_t = clip(lr / sqrt(v_hat), final_lr * (1 - 1/t), final_lr * (1 + 1/t))
        theta_new = theta - lr_t * m_hat
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        final_lr: float = 0.1,
        epsilon: float = 1e-8,
        gamma: float = 1e-3
    ) -> None:
        """
        Initialize AdaBound optimizer.

        Args:
            lr: Initial learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            final_lr: Final (SGD) learning rate
            epsilon: Small constant for numerical stability
            gamma: Convergence speed parameter
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.final_lr = final_lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.name = f"AdaBound(lr={lr}, final_lr={final_lr})"

        # Initialize moments
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one AdaBound step."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Update biased first moment
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # Update biased second moment
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x ** 2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y ** 2

        # Bias-corrected moments
        m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
        m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
        v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
        v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

        # Dynamic bounds
        lower_bound = self.final_lr * (1.0 - 1.0 / (self.gamma * self.t + 1.0))
        upper_bound = self.final_lr * (1.0 + 1.0 / (self.gamma * self.t))

        # Adaptive learning rate with bounds
        step_x = np.clip(self.lr / (np.sqrt(v_x_hat) + self.epsilon), lower_bound, upper_bound)
        step_y = np.clip(self.lr / (np.sqrt(v_y_hat) + self.epsilon), lower_bound, upper_bound)

        new_x = x - step_x * m_x_hat
        new_y = y - step_y * m_y_hat
        self._append_history((new_x, new_y))
        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        elif self.m.shape != params.shape:
            logging.warning("AdaBound: Parameter shape changed. Resizing state.")
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Type safety check
        if self.m is None or self.v is None:
            raise TypeError("AdaBound optimizer state not initialized properly.")

        grad_array = np.asarray(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array ** 2

        m_hat = self.m / max(1 - self.beta1 ** self.t, 1e-8)
        v_hat = self.v / max(1 - self.beta2 ** self.t, 1e-8)

        # Dynamic bounds
        lower_bound = self.final_lr * (1.0 - 1.0 / (self.gamma * self.t + 1.0))
        upper_bound = self.final_lr * (1.0 + 1.0 / (self.gamma * self.t))

        # Adaptive step size with bounds
        step_size = self.lr / (np.sqrt(v_hat) + self.epsilon)
        step_size = np.clip(step_size, lower_bound, upper_bound)

        return params - step_size * m_hat

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0


class RAdam(Optimizer):
    """
    RAdam: Rectified Adam optimizer.

    Addresses the bad convergence problem of Adam by rectifying the adaptive learning rate.
    Reference: https://arxiv.org/abs/1908.03265

    Key idea: Use warmup heuristic based on variance of adaptive learning rate.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        """Initialize RAdam optimizer."""
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = f"RAdam(lr={lr})"

        # Initialize moments
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0

        # Compute rho_inf (maximum length of approximated SMA)
        self.rho_inf = 2.0 / (1.0 - self.beta2) - 1.0

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one RAdam step."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Update biased first moment
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # Update biased second moment
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x ** 2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y ** 2

        # Bias correction for first moment
        m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
        m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)

        # Compute length of the approximated SMA
        rho_t = self.rho_inf - 2.0 * self.t * (self.beta2 ** self.t) / max(1.0 - self.beta2 ** self.t, 1e-8)

        # Check if variance is tractable
        if rho_t > 4.0:
            # Rectified update with bias correction
            v_x_hat = self.v_x / (1 - self.beta2 ** self.t)
            v_y_hat = self.v_y / (1 - self.beta2 ** self.t)

            r_t = np.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf) /
                         ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))

            new_x = x - self.lr * r_t * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
            new_y = y - self.lr * r_t * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)
        else:
            # Use un-adapted update (like SGD with momentum)
            new_x = x - self.lr * m_x_hat
            new_y = y - self.lr * m_y_hat

        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.m is None or self.m.shape != params.shape:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        if self.m is None or self.v is None:
            raise TypeError("RAdam optimizer state not initialized properly.")
        
        grad_array = np.asarray(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array ** 2

        m_hat = self.m / max(1 - self.beta1 ** self.t, 1e-8)
        rho_t = self.rho_inf - 2.0 * self.t * (self.beta2 ** self.t) / max(1.0 - self.beta2 ** self.t, 1e-8)

        if rho_t > 4.0:
            v_hat = self.v / max(1 - self.beta2 ** self.t, 1e-8)
            r_t = np.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf) /
                         ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))
            return params - self.lr * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            return params - self.lr * m_hat

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0


class LAMB(Optimizer):
    """
    LAMB: Layer-wise Adaptive Moments optimizer for Batch training.

    Designed for large batch training, uses layer-wise adaptation.
    Reference: https://arxiv.org/abs/1904.00962

    Key idea: Trust ratio based on layer-wise norms for better large-batch training.
    
    NOTE: This implementation uses a simplified global trust ratio for 2D optimization.
    For neural network training with true layer-wise adaptation, use pytorch_optimizers.LAMBWrapper.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        """Initialize LAMB optimizer."""
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = f"LAMB(lr={lr}, wd={weight_decay})"

        # Initialize moments
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0

    def step(
        self,
        params: Union[Tuple[float, float], Any],
        gradients: Union[Tuple[float, float], Any],
        **kwargs: Any
    ) -> Union[Tuple[float, float], Any]:
        """Perform one LAMB step."""
        self.t += 1
        return self._dispatch_step(params, gradients, self._step_tuple, self._step_array)

    def _step_tuple(
        self,
        params: Tuple[float, float],
        gradients: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Handle tuple parameters (2D test functions)."""
        x, y = params
        grad_x, grad_y = gradients

        # Update biased first moment
        self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
        self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

        # Update biased second moment
        self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x ** 2
        self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y ** 2

        # Bias correction
        m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
        m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
        v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
        v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

        # Adam update (before trust ratio)
        update_x = m_x_hat / (np.sqrt(v_x_hat) + self.epsilon) + self.weight_decay * x
        update_y = m_y_hat / (np.sqrt(v_y_hat) + self.epsilon) + self.weight_decay * y

        # Compute trust ratio (numerical stability with hypot)
        param_norm = np.hypot(x, y)
        update_norm = np.hypot(update_x, update_y)

        if param_norm > self.epsilon and update_norm > self.epsilon:
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0

        # Apply trust ratio
        new_x = x - self.lr * trust_ratio * update_x
        new_y = y - self.lr * trust_ratio * update_y

        return new_x, new_y

    def _step_array(self, params: Any, gradients: Any) -> Any:
        """Handle array parameters (neural networks)."""
        if self.m is None or self.m.shape != params.shape:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        # Type safety check
        if self.m is None or self.v is None:
            raise TypeError("LAMB optimizer state not initialized properly.")
        
        grad_array = np.asarray(gradients)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        update = m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params

        # Compute trust ratio with numerical stability
        param_norm = np.linalg.norm(params)
        update_norm = np.linalg.norm(update)

        if param_norm > self.epsilon and update_norm > self.epsilon:
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0

        return params - self.lr * trust_ratio * update

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m_x, self.m_y = 0.0, 0.0
        self.v_x, self.v_y = 0.0, 0.0
        self.m, self.v = None, None
        self.t = 0


# -----------------------------------------------------------------------------
# Optimizer factory
# -----------------------------------------------------------------------------

def create_optimizer_instance(name: str, **kwargs) -> Optimizer:
    """Create an optimizer instance given a (possibly non-canonical) name.

    This function accepts names like 'SGDMomentum' or 'SGD_Momentum' and
    normalizes them before instantiation. It provides a single place to
    centralize any alias handling for the simple optimizers used in 2D tests.
    """
    try:
        from src.core.optimizer_registry import normalize_optimizer_name
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logging.debug("optimizer_registry import failed: %s", e, exc_info=True)
        # Minimal fallback normalization
        def normalize_optimizer_name(name: str) -> str:
            return name.replace(' ', '_').replace('-', '_')

    canon = normalize_optimizer_name(name)

    # Map canonical names to classes
    if canon == 'SGD':
        return SGD(**kwargs)
    elif canon == 'SGD_Momentum':
        # Accept 'momentum' as alias for 'beta' (SGDMomentum expects 'beta')
        params = kwargs.copy()
        if 'momentum' in params and 'beta' not in params:
            params['beta'] = params.pop('momentum')
        return SGDMomentum(**params)
    elif canon in ('SGDNesterov', 'SGD_Nesterov'):
        params = kwargs.copy()
        if 'momentum' in params and 'beta' not in params:
            params['beta'] = params.pop('momentum')
        return SGDNesterov(**params)
    elif canon.lower() == 'rmsprop' or canon == 'RMSProp':
        return RMSProp(**kwargs)
    elif canon == 'Adam':
        return Adam(**kwargs)
    elif canon == 'AdamW':
        return AdamW(**kwargs)
    elif canon == 'AMSGrad':
        return AMSGrad(**kwargs)
    elif canon == 'LAMB':
        return LAMB(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer name for factory: {name} (normalized: {canon})")


