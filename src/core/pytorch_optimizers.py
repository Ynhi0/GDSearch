"""
PyTorch-compatible optimizer wrappers for GDSearch custom optimizers.

Wraps our custom optimizers (SGD, Adam, etc.) to work with PyTorch nn.Module parameters.
"""

import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import custom optimizers - handle path properly
try:
    from src.core.optimizers import SGD as CustomSGD, SGDMomentum as CustomSGDMomentum
    from src.core.optimizers import Adam as CustomAdam, RMSProp as CustomRMSProp
except ModuleNotFoundError:
    # If running as script, add parent to path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.core.optimizers import SGD as CustomSGD, SGDMomentum as CustomSGDMomentum
    from src.core.optimizers import Adam as CustomAdam, RMSProp as CustomRMSProp


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
                    self.custom_opts[id(p)] = CustomSGDMomentum(
                        lr=group['lr'],
                        momentum=group['momentum']
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
                    self.custom_opts[id(p)] = CustomRMSProp(
                        lr=group['lr'],
                        alpha=group['alpha'],
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


if __name__ == '__main__':
    # Test the wrappers
    print("Testing PyTorch optimizer wrappers...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 2)
    
    # Test each wrapper
    optimizers = {
        'SGD': SGDWrapper(model.parameters(), lr=0.01),
        'SGDMomentum': SGDMomentumWrapper(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': AdamWrapper(model.parameters(), lr=0.001),
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
