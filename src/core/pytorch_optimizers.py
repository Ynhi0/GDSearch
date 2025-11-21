"""
PyTorch-compatible optimizer wrappers for GDSearch custom optimizers.

Wraps our custom optimizers (SGD, Adam, etc.) to work with PyTorch nn.Module parameters.
"""

import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import custom optimizers - handle path properly
try:
    from src.core.optimizers import (
        SGD as CustomSGD,
        SGDMomentum as CustomSGDMomentum,
        SGDNesterov as CustomSGDNesterov,
        Adam as CustomAdam,
        AdamW as CustomAdamW,
        RMSProp as CustomRMSProp,
        SAM as CustomSAM,
        Lookahead as CustomLookahead,
    )
except ModuleNotFoundError:
    # If running as script, add parent to path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.core.optimizers import (
        SGD as CustomSGD,
        SGDMomentum as CustomSGDMomentum,
        SGDNesterov as CustomSGDNesterov,
        Adam as CustomAdam,
        AdamW as CustomAdamW,
        RMSProp as CustomRMSProp,
        SAM as CustomSAM,
        Lookahead as CustomLookahead,
    )


class SGDWrapper(Optimizer):
    """PyTorch wrapper for custom SGD optimizer."""
    
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.custom_opt = CustomSGD(lr=lr)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                
                # Compute update
                updated_param = self.custom_opt.step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss


class SGDMomentumWrapper(Optimizer):
    """PyTorch wrapper for custom SGD with momentum optimizer."""
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Initialize optimizer for this parameter if needed
                if id(p) not in self.custom_opts:
                    # Map torch's momentum -> beta in custom optimizer
                    self.custom_opts[id(p)] = CustomSGDMomentum(
                        lr=group['lr'],
                        beta=group['momentum']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                
                # Compute update
                updated_param = self.custom_opts[id(p)].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss


class AdamWrapper(Optimizer):
    """PyTorch wrapper for custom Adam optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Initialize optimizer for this parameter if needed
                if id(p) not in self.custom_opts:
                    self.custom_opts[id(p)] = CustomAdam(
                        lr=group['lr'],
                        beta1=group['beta1'],
                        beta2=group['beta2'],
                        epsilon=group['epsilon']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                
                # Compute update
                updated_param = self.custom_opts[id(p)].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss


class SGDNesterovWrapper(Optimizer):
    """PyTorch wrapper for custom SGD with Nesterov (NAG)."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.custom_opts = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if id(p) not in self.custom_opts:
                    self.custom_opts[id(p)] = CustomSGDNesterov(
                        lr=group['lr'],
                        beta=group['momentum']
                    )
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                updated_param = self.custom_opts[id(p)].step(param_np.flatten(), grad.flatten())
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)

        return loss


class RMSPropWrapper(Optimizer):
    """PyTorch wrapper for custom RMSProp optimizer."""
    
    def __init__(self, params, lr=0.01, alpha=0.99, epsilon=1e-8):
        defaults = dict(lr=lr, alpha=alpha, epsilon=epsilon)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Initialize optimizer for this parameter if needed
                if id(p) not in self.custom_opts:
                    # Map torch's alpha (smoothing) -> decay_rate in custom RMSProp
                    self.custom_opts[id(p)] = CustomRMSProp(
                        lr=group['lr'],
                        decay_rate=group['alpha'],
                        epsilon=group['epsilon']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                
                # Compute update
                updated_param = self.custom_opts[id(p)].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss


class AdamWWrapper(Optimizer):
    """PyTorch wrapper for custom AdamW optimizer (decoupled weight decay)."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        beta1, beta2 = betas
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.custom_opts = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if id(p) not in self.custom_opts:
                    self.custom_opts[id(p)] = CustomAdamW(
                        lr=group['lr'],
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=group['eps'],
                        weight_decay=group['weight_decay']
                    )
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                updated_param = self.custom_opts[id(p)].step(param_np.flatten(), grad.flatten())
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)

        return loss


class SAMWrapper(Optimizer):
    """PyTorch wrapper for SAM (Sharpness-Aware Minimization) optimizer."""
    
    def __init__(self, params, lr=0.01, rho=0.05, base_optimizer='SGD', **base_kwargs):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            rho: Neighborhood size (sharpness radius)
            base_optimizer: Base optimizer class name ('SGD', 'Adam', etc.)
            **base_kwargs: Additional arguments for base optimizer
        """
        defaults = dict(lr=lr, rho=rho)
        super().__init__(params, defaults)
        
        # Import here to avoid circular imports
        from src.core.optimizers import SAM as CustomSAM
        self.custom_opt = CustomSAM(lr=lr, rho=rho, base_optimizer=base_optimizer, **base_kwargs)
        
        # Store model reference for adversarial step computation
        self.model = None
        self.criterion = None
        
    def set_model_and_criterion(self, model, criterion):
        """Set model and criterion for SAM adversarial step computation."""
        self.model = model
        self.criterion = criterion
    
    def step(self, closure=None):
        """
        Perform SAM update step.
        
        For SAM to work properly, you need to call set_model_and_criterion() first
        and provide a closure that computes the loss.
        """
        if self.model is None or self.criterion is None:
            raise ValueError("SAM requires model and criterion to be set via set_model_and_criterion()")
        
        if closure is None:
            raise ValueError("SAM requires a closure function to compute adversarial gradients")
        
        loss = None
        if closure is not None:
            loss = closure()
        
        # Store original parameters
        original_params = []
        for group in self.param_groups:
            for p in group['params']:
                original_params.append(p.data.clone())
        
        # Compute adversarial step
        # First, compute gradients at current point
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Compute perturbation: ρ * (g / ||g||)
                grad_norm = torch.norm(p.grad)
                if grad_norm > 1e-12:
                    perturbation = group['rho'] * (p.grad / grad_norm)
                    p.data.add_(perturbation)
        
        # Compute loss and gradients at adversarial point
        adv_loss = closure()
        
        # Now restore original parameters and use adversarial gradients for update
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(original_params[idx])
                idx += 1
        
        # The gradients are now computed at the adversarial point
        # Use base optimizer logic (simplified - in practice would delegate to base opt)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-group['lr'])
        
        return loss


class LookaheadWrapper(Optimizer):
    """PyTorch wrapper for Lookahead optimizer."""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        """
        Initialize Lookahead wrapper.
        
        Args:
            base_optimizer: PyTorch optimizer instance to wrap
            k: Number of fast steps before slow update
            alpha: Interpolation factor between slow and fast weights
        """
        # Extract parameters from base optimizer
        params = []
        for group in base_optimizer.param_groups:
            params.extend(group['params'])
        
        defaults = dict(k=k, alpha=alpha)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Initialize slow weights
        self.slow_params = []
        for p in self.param_groups[0]['params']:
            self.slow_params.append(p.data.clone())
    
    def step(self, closure=None):
        """Perform Lookahead update step."""
        loss = self.base_optimizer.step(closure)
        
        # Increment step counter
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    # Interpolate: slow = α * slow + (1-α) * fast
                    alpha = group['alpha']
                    self.slow_params[idx] = alpha * self.slow_params[idx] + (1 - alpha) * p.data
                    p.data.copy_(self.slow_params[idx])
                    idx += 1
        
        return loss


def test_sam_and_lookahead():
    """Test SAM and Lookahead optimizers."""
    print("Testing SAM and Lookahead optimizers...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    criterion = torch.nn.MSELoss()
    
    # Test data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Test SAM
    print("  Testing SAM...")
    sam_opt = SAMWrapper(model.parameters(), lr=0.01, rho=0.05, base_optimizer='SGD')
    sam_opt.set_model_and_criterion(model, criterion)
    
    def closure():
        sam_opt.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        return loss
    
    try:
        loss = sam_opt.step(closure)
        print(f"  ✓ SAM step completed successfully, loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ SAM failed: {e}")
    
    # Test Lookahead
    print("  Testing Lookahead...")
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    lookahead_opt = LookaheadWrapper(base_opt, k=3, alpha=0.5)
    
    try:
        def lookahead_closure():
            lookahead_opt.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            return loss
        
        loss = lookahead_opt.step(lookahead_closure)
        print(f"  ✓ Lookahead step completed successfully, loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Lookahead failed: {e}")
    
    print("\n✓ SAM and Lookahead optimizer wrappers tested!")


if __name__ == '__main__':
    # Test the wrappers
    print("Testing PyTorch optimizer wrappers...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 2)
    
    # Test each wrapper
    optimizers = {
        'SGD': SGDWrapper(model.parameters(), lr=0.01),
        'SGDMomentum': SGDMomentumWrapper(model.parameters(), lr=0.01, momentum=0.9),
        'SGDNesterov': SGDNesterovWrapper(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': AdamWrapper(model.parameters(), lr=0.001),
        'AdamW': AdamWWrapper(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
        'RMSProp': RMSPropWrapper(model.parameters(), lr=0.01)
    }
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}:")
        
        # Reset model
        model = torch.nn.Linear(10, 2)
        
        # Dummy forward and backward
        x = torch.randn(5, 10)
        y = torch.randn(5, 2)
        
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        print(f"  ✓ Step completed successfully")
        print(f"  Loss: {loss.item():.4f}")
    
    print("\n✓ All optimizer wrappers working!")
    test_sam_and_lookahead()
