"""
PyTorch-native optimizer implementations with zero-copy GPU execution.

This module replaces the inefficient numpy-based wrappers in pytorch_optimizers.py
with true PyTorch tensor operations for maximum performance.

All optimizers inherit from torch.optim.Optimizer and use in-place operations
to avoid CPU-GPU data transfer overhead.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


class TorchSGDMomentum(Optimizer):
    """
    Native PyTorch SGD with Momentum implementation.
    
    Uses pure tensor operations - no numpy conversions.
    Supports GPU acceleration with zero overhead.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(TorchSGDMomentum, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(TorchSGDMomentum, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Optional loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # Apply momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                
                # Update parameters (in-place, zero-copy on GPU)
                p.add_(buf, alpha=-lr)
        
        return loss


class TorchAdam(Optimizer):
    """
    Native PyTorch Adam implementation.
    
    Implements Algorithm 1 from "Adam: A Method for Stochastic Optimization"
    (Kingma & Ba, 2015) with pure tensor operations.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(TorchAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(TorchAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay (L2 regularization - coupled variant)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                # Update parameters (in-place)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class TorchAdamW(Optimizer):
    """
    Native PyTorch AdamW implementation with decoupled weight decay.
    
    Implements "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    with pure tensor operations for maximum GPU performance.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(TorchAdamW, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(TorchAdamW, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                # AdamW: Decoupled weight decay (applied AFTER gradient update)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply weight decay directly to parameters
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
        
        return loss


class TorchSAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer with native PyTorch operations.
    
    Implements "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    (Foret et al., 2021) without numpy overhead.
    
    SAM performs two passes:
    1. Ascent step: move to worst-case perturbation
    2. Descent step: compute gradient at perturbed location and update
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.SGD)
            rho: Neighborhood size for perturbation (default: 0.05)
            **kwargs: Arguments for base optimizer
        """
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        
        defaults = dict(rho=rho, **kwargs)
        super(TorchSAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: compute and apply adversarial perturbation.
        
        Args:
            zero_grad: Whether to zero gradients after this step
        """
        # Compute gradient norm
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Save original parameters
                self.state[p]['old_p'] = p.data.clone()
                
                # Adversarial perturbation
                e_w = p.grad * scale
                p.add_(e_w)  # Move to perturbed location
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: update parameters using gradient at perturbed location.
        
        Args:
            zero_grad: Whether to zero gradients after this step
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Restore original parameters
                p.data = self.state[p]['old_p']
        
        # Update using base optimizer
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        """
        Single step combining both SAM phases.
        Requires closure for re-computing gradients.
        """
        if closure is None:
            raise ValueError("SAM requires closure for gradient re-computation")
        
        # First forward-backward pass (for perturbation)
        loss = closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass (at perturbed location)
        closure()
        self.second_step()
        
        return loss
    
    def _grad_norm(self):
        """Compute L2 norm of gradients."""
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class TorchLookahead(Optimizer):
    """
    Lookahead optimizer with native PyTorch operations.
    
    Implements "Lookahead Optimizer: k steps forward, 1 step back"
    (Zhang et al., 2019) without numpy overhead.
    """
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        """
        Args:
            base_optimizer: Inner optimizer instance
            k: Number of lookahead steps (default: 5)
            alpha: Slow weights update coefficient (default: 0.5)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if k < 1:
            raise ValueError(f"Invalid k value: {k}")
        
        self.base_optimizer = base_optimizer
        # Initialize parent with base optimizer's param_groups
        super(TorchLookahead, self).__init__(base_optimizer.param_groups, {})
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        
        # Cache for slow weights
        # 🐛 AUDIT FIX: Use id(p) as key instead of tensor p (tensors are unhashable)
        self.slow_weights = {}
        for group in self.param_groups:
            for p in group['params']:
                self.slow_weights[id(p)] = p.data.clone()
    
    def __getstate__(self):
        return {
            'base_optimizer': self.base_optimizer,
            'param_groups': self.param_groups,
            'k': self.k,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'slow_weights': self.slow_weights,
        }
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    @property
    def state(self):
        return self.base_optimizer.state
    
    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # Update fast weights
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1
        
        # Update slow weights every k steps
        if self.step_counter % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    # 🐛 AUDIT FIX: Use id(p) as key
                    p_id = id(p)
                    if p_id in self.slow_weights:
                        # Interpolate: slow = slow + alpha * (fast - slow)
                        self.slow_weights[p_id].add_(p.data - self.slow_weights[p_id], alpha=self.alpha)
                        # Copy slow weights to fast weights
                        p.data.copy_(self.slow_weights[p_id])
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# Convenience factory functions
def create_sgd_momentum(params, lr=0.01, momentum=0.9, weight_decay=0.0):
    """Factory for SGD with Momentum."""
    return TorchSGDMomentum(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def create_adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Factory for Adam."""
    return TorchAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def create_adamw(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
    """Factory for AdamW."""
    return TorchAdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def create_sam(params, base_optimizer_class=torch.optim.SGD, rho=0.05, **kwargs):
    """Factory for SAM."""
    return TorchSAM(params, base_optimizer_class, rho=rho, **kwargs)


def create_lookahead(base_optimizer, k=5, alpha=0.5):
    """Factory for Lookahead."""
    return TorchLookahead(base_optimizer, k=k, alpha=alpha)
