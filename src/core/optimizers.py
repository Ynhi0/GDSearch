# -*- coding: utf-8 -*-
"""
Module defining optimization algorithms (optimizers).
"""

import numpy as np
import logging
from typing import Tuple, Union, Any


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
        """Perform one SGD step with optional weight decay."""
        # Supports both tuple (x,y) and numpy array
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients

            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad_x += self.weight_decay * x
                grad_y += self.weight_decay * y

            new_x = x - self.lr * grad_x
            new_y = y - self.lr * grad_y
            # Track history
            self._append_history((new_x, new_y))
            return new_x, new_y
        else:
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

    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one SGD with Momentum step with optional weight decay."""
        # Supports both tuple (x,y) and numpy array
        if isinstance(params, tuple):
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
            # Track history
            self._append_history((new_x, new_y))
            return new_x, new_y
        else:
            # Handle array (for neural networks)
            if self.v is None:
                self.v = np.zeros_like(params)

            # Skip updates when gradients contain NaN or Inf
            if not np.isfinite(gradients).all():
                logging.warning("SGDMomentum: Non-finite gradients detected, skipping update")
                return params

            # Apply weight decay (L2 regularization)
            # Type safety: gradients is array-like here (not tuple)
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

    def reset(self):
        """Reset velocity to 0."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None
        super().reset()


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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
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
            self._append_history((new_x, new_y))
            return new_x, new_y
        else:
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

    def reset(self):
        self.v_x = 0.0
        self.v_y = 0.0
        self.v = None


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation).

    Update formula:
        s_new = decay_rate * s_old + (1 - decay_rate) * gradient^2
        θ_new = θ_old - lr * gradient / sqrt(s_new + epsilon)
    """

    def __init__(self, lr=0.01, decay_rate=0.9, epsilon=1e-8):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one RMSProp step."""
        # Supports both tuple (x,y) and numpy array
        if isinstance(params, tuple):
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
        else:
            # Handle array (for neural networks)
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

    def reset(self):
        """Reset squared gradient accumulator to 0."""
        self.s_x = 0.0
        self.s_y = 0.0
        self.s = None


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

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one Adam step with optional L2 regularization."""
        # Increment timestep
        self.t += 1

        # Supports both tuple (x,y) and numpy array
        if isinstance(params, tuple):
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

            # Compute bias-corrected moment estimates
            m_x_hat = self.m_x / max(1 - self.beta1**self.t, 1e-8)
            m_y_hat = self.m_y / max(1 - self.beta1**self.t, 1e-8)
            v_x_hat = self.v_x / max(1 - self.beta2**self.t, 1e-8)
            v_y_hat = self.v_y / max(1 - self.beta2**self.t, 1e-8)

            # Update parameters
            new_x = x - self.lr * m_x_hat / (np.sqrt(v_x_hat) + self.epsilon)
            new_y = y - self.lr * m_y_hat / (np.sqrt(v_y_hat) + self.epsilon)
            self._append_history((new_x, new_y))
            return new_x, new_y
        else:
            # Handle array (for neural networks)
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

            # Update biased first moment estimate
            assert self.m is not None
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array

            # Update biased second moment estimate
            assert self.v is not None
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array**2

            # Compute bias-corrected moment estimates
            m_hat = self.m / max(1 - self.beta1**self.t, 1e-8)
            v_hat = self.v / max(1 - self.beta2**self.t, 1e-8)

            # Update parameters
            updated = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self._append_history(updated)
            return updated

    def reset(self):
        """Reset moment estimates and timestep to 0."""
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        self.t += 1
        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients

            # update moments
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
        else:
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

            assert self.m is not None
            assert self.v is not None
            grad_array = np.asarray(gradients)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_array ** 2)
            m_hat = self.m / max(1 - self.beta1 ** self.t, 1e-8)
            v_hat = self.v / max(1 - self.beta2 ** self.t, 1e-8)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # Decoupled weight decay: apply to original params
            return params - step - self.lr * self.weight_decay * params

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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        self.t += 1
        if isinstance(params, tuple):
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
        else:
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
                self.vhat_max = np.zeros_like(params)
            elif self.m.shape != params.shape:
                logging.warning("AMSGrad: Parameter shape changed from %s to %s. Resizing state.", self.m.shape, params.shape)
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
                self.vhat_max = np.zeros_like(params)
            assert self.m is not None
            assert self.v is not None
            assert self.vhat_max is not None
            grad_array = np.asarray(gradients)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_array ** 2)
            m_hat = self.m / max(1 - self.beta1 ** self.t, 1e-8)
            v_hat = self.v / max(1 - self.beta2 ** self.t, 1e-8)
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
            # Fail-fast: SAM requires either pre-computed adversarial gradients
            # (from a closure/second forward pass) or a loss function capable of
            # computing gradients at the adversarial point. Silently falling back
            # to the base optimizer can disable SAM undetected and break experiments.
            raise RuntimeError(
                "SAM.step requires `adversarial_gradients` or `loss_fn` to compute "
                "the SAM adversarial update; refusing to proceed without them. "
                "In PyTorch use SAMWrapper which provides the required closure/gradients."
            )

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
            logging.warning("Lookahead with %s may interfere with internal optimizer state (running averages).", base_optimizer.name)
            logging.warning("Consider using Lookahead only with SGD for reliable behavior.")
            logging.warning("This is mentioned in the thesis for educational purposes but not recommended for production use.")

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

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, epsilon=1e-8, gamma=1e-3):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one AdaBound step."""
        self.t += 1

        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients

            # Update biased first moment
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

            # Update biased second moment
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x ** 2
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y ** 2

            # Compute bias-corrected moments
            # Add epsilon guard for numerical stability
            m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
            m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
            v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
            v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

            # Compute dynamic bounds
            final_lr_t = self.final_lr * self.lr / self.gamma
            lower_bound = final_lr_t * (1.0 - 1.0 / (self.gamma * self.t + 1.0))
            upper_bound = final_lr_t * (1.0 + 1.0 / (self.gamma * self.t))

            # Compute step sizes with bounds
            step_size_x = self.lr / (np.sqrt(v_x_hat) + self.epsilon)
            step_size_y = self.lr / (np.sqrt(v_y_hat) + self.epsilon)

            step_size_x = np.clip(step_size_x, lower_bound, upper_bound)
            step_size_y = np.clip(step_size_y, lower_bound, upper_bound)

            # Update parameters
            new_x = x - step_size_x * m_x_hat
            new_y = y - step_size_y * m_y_hat
            return new_x, new_y
        else:
            # Array version
            if self.m is None or self.m.shape != params.shape:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)

            assert self.m is not None
            assert self.v is not None
            grad_array = np.asarray(gradients)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array ** 2

            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            final_lr_t = self.final_lr * self.lr / self.gamma
            lower_bound = final_lr_t * (1.0 - 1.0 / (self.gamma * self.t + 1.0))
            upper_bound = final_lr_t * (1.0 + 1.0 / (self.gamma * self.t))

            step_size = self.lr / (np.sqrt(v_hat) + self.epsilon)
            step_size = np.clip(step_size, lower_bound, upper_bound)

            return params - step_size * m_hat

    def reset(self):
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

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one RAdam step."""
        self.t += 1

        if isinstance(params, tuple):
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
            rho_t = self.rho_inf - 2.0 * self.t * (self.beta2 ** self.t) / (1.0 - self.beta2 ** self.t)

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
        else:
            # Array version
            if self.m is None or self.m.shape != params.shape:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)

            assert self.m is not None
            assert self.v is not None
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

    def reset(self):
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
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
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

    def step(self, params, gradients, **kwargs) -> Union[Tuple[float, float], Any]:
        """Perform one LAMB step."""
        self.t += 1

        if isinstance(params, tuple):
            x, y = params
            grad_x, grad_y = gradients

            # Update biased first moment
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y

            # Update biased second moment
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x ** 2
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y ** 2

            # Bias correction
            # Add epsilon guard for numerical stability
            m_x_hat = self.m_x / max(1 - self.beta1 ** self.t, 1e-8)
            m_y_hat = self.m_y / max(1 - self.beta1 ** self.t, 1e-8)
            v_x_hat = self.v_x / max(1 - self.beta2 ** self.t, 1e-8)
            v_y_hat = self.v_y / max(1 - self.beta2 ** self.t, 1e-8)

            # Adam update (before trust ratio)
            update_x = m_x_hat / (np.sqrt(v_x_hat) + self.epsilon) + self.weight_decay * x
            update_y = m_y_hat / (np.sqrt(v_y_hat) + self.epsilon) + self.weight_decay * y

            # Compute trust ratio
            # Add epsilon for numerical stability in edge cases
            param_norm = np.sqrt(x**2 + y**2)
            update_norm = np.sqrt(update_x**2 + update_y**2)

            if param_norm > self.epsilon and update_norm > self.epsilon:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0

            # Apply trust ratio
            new_x = x - self.lr * trust_ratio * update_x
            new_y = y - self.lr * trust_ratio * update_y

            return new_x, new_y
        else:
            # Array version
            if self.m is None or self.m.shape != params.shape:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)

            # Ensure arrays initialized for safe arithmetic
            assert self.m is not None and self.v is not None, "Internal state arrays must be initialized"
            grad_array = np.asarray(gradients)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_array
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad_array ** 2

            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            update = m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params

            # Use epsilon for numerical stability in norm comparison
            param_norm = np.linalg.norm(params)
            update_norm = np.linalg.norm(update)

            if param_norm > self.epsilon and update_norm > self.epsilon:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0

            return params - self.lr * trust_ratio * update

    def reset(self):
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


