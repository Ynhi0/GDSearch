"""
PyTorch-compatible optimizer wrappers for GDSearch custom optimizers.

Wraps our custom optimizers (SGD, Adam, etc.) to work with PyTorch nn.Module parameters.
"""

import torch
import logging
import numpy as np
from torch.optim.optimizer import Optimizer
import math
from collections import OrderedDict
from typing import Any, cast

# Import custom optimizers - handle path properly
try:
    from src.core.optimizers import (
        SGD as CustomSGD,
        SGDMomentum as CustomSGDMomentum,
        SGDNesterov as CustomSGDNesterov,
        Adam as CustomAdam,
        AdamW as CustomAdamW,
        RMSProp as CustomRMSProp,
        AdaBound as CustomAdaBound,
        RAdam as CustomRAdam,
        LAMB as CustomLAMB,
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
        AdaBound as CustomAdaBound,
        RAdam as CustomRAdam,
        LAMB as CustomLAMB,
    )


class SGDWrapper(Optimizer):
    """PyTorch wrapper for custom SGD optimizer."""
    
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.custom_opt = CustomSGD(lr=lr)
        
        # Flag for OOM handler - SGD does not require closure
        self.requires_closure = False
        
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
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                
                # Validate gradients before step to fail fast on NaN/Inf
                if not np.isfinite(grad).all():
                    raise ValueError(
                        "SGDWrapper: Non-finite gradient detected before step.\\n"
                        "  Gradient contains NaN or Inf values.\\n"
                        "  This indicates numerical instability in the forward/backward pass.\\n"
                        "  Consider: gradient clipping, smaller learning rate, or mixed precision training."
                    )
                
                # Compute update
                updated_param = self.custom_opt.step(param_np.flatten(), grad.flatten())
                if not isinstance(updated_param, np.ndarray):
                    raise TypeError(f"SGDWrapper: custom optimizer step() must return numpy.ndarray, got {type(updated_param).__name__}")
                
                # Validate shape before reshaping
                if updated_param.size != param_np.size:
                    raise ValueError(
                        f"SGDWrapper: Shape mismatch:\\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\\n"
                        f"  Returned size: {updated_param.size}\\n"
                        f"  Param device: {p.device}, dtype: {original_dtype}\\n"
                        f"  This indicates a bug in the custom optimizer's step() method."
                    )
                
                # Reshape and update parameter preserving dtype/device
                try:
                    updated_tensor = torch.from_numpy(updated_param.reshape(original_shape))
                    p.data.copy_(updated_tensor.to(original_dtype).to(p.device))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to update parameter with shape {original_shape}, "
                        f"dtype {original_dtype}: {e}"
                    ) from e
        
        return loss


class SGDMomentumWrapper(Optimizer):
    """PyTorch wrapper for custom SGD with momentum optimizer."""
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
        # Flag for OOM handler - SGDMomentum does not require closure
        self.requires_closure = False
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Use (group_idx, param_idx) as key for cross-process safety
                key = (group_idx, param_idx)
                
                # Initialize optimizer for this parameter if needed
                if key not in self.custom_opts:
                    # Map torch's momentum -> beta in custom optimizer
                    self.custom_opts[key] = CustomSGDMomentum(
                        lr=group['lr'],
                        beta=group['momentum']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                original_shape = param_np.shape
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # FIXED: Validate shape before reshaping to prevent silent corruption
                if updated_param.size != param_np.size:
                    param_name = f"group{group_idx}_param{param_idx}"
                    raise ValueError(
                        f"SGDMomentumWrapper: Shape mismatch in {param_name}:\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\n"
                        f"  Returned size: {updated_param.size}\n"
                        f"  Param device: {p.device}, dtype: {p.dtype}\n"
                        f"  This indicates a bug in the custom optimizer's step() method.\n"
                        f"  Check src/core/optimizers.py SGDMomentum implementation."
                    )
                
                # Reshape and update parameter preserving dtype/device
                updated_tensor = torch.from_numpy(updated_param.reshape(param_np.shape))
                p.data.copy_(updated_tensor.to(p.data.dtype).to(p.device))
        
        return loss
    
    def state_dict(self):
        """Save custom optimizer states with index-based keys for cross-process safety.

        Uses numpy arrays directly instead of .tolist() to keep serialization lightweight.
        Torch handles numpy array serialization efficiently.
        """
        base_state = super().state_dict()
        # Serialize custom_opts: map (group_idx, param_idx) to optimizer state (v=velocity, not m)
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            # Keep as numpy array; torch.save handles this efficiently
            custom_state[key_str] = {
                'v': opt.v if opt.v is not None else None,  # Keep as numpy, no .tolist()
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """Restore custom optimizer states from checkpoint using index mapping."""
        import numpy as np
        custom_state = state_dict.pop('custom_opts', {})
        super().load_state_dict(state_dict)
        
        # Reconstruct custom_opts from serialized state using index mapping
        self.custom_opts = {}
        for key_str, opt_state in custom_state.items():
            # Parse string key back to tuple
            group_idx, param_idx = map(int, key_str.split(','))
            key = (group_idx, param_idx)
            
            # Validate indices
            if group_idx < len(self.param_groups):
                group = self.param_groups[group_idx]
                if param_idx < len(group['params']):
                    opt = CustomSGDMomentum(lr=group['lr'], beta=group['momentum'])
                    opt.v = np.array(opt_state['v'], dtype=np.float32) if opt_state['v'] is not None else None
                    self.custom_opts[key] = opt


class AdamWrapper(Optimizer):
    """PyTorch wrapper for custom Adam optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
        # Flag for OOM handler - Adam does not require closure
        self.requires_closure = False
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Use (group_idx, param_idx) as key for cross-process safety
                key = (group_idx, param_idx)
                
                # Initialize optimizer for this parameter if needed
                if key not in self.custom_opts:
                    self.custom_opts[key] = CustomAdam(
                        lr=group['lr'],
                        beta1=group['beta1'],
                        beta2=group['beta2'],
                        epsilon=group['epsilon']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                original_shape = param_np.shape
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # FIXED: Validate shape before reshaping to prevent silent corruption
                if updated_param.size != param_np.size:
                    param_name = f"group{group_idx}_param{param_idx}"
                    raise ValueError(
                        f"AdamWrapper: Shape mismatch in {param_name}:\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\n"
                        f"  Returned size: {updated_param.size}\n"
                        f"  Param device: {p.device}, dtype: {p.dtype}\n"
                        f"  Optimizer state - m: {self.custom_opts[key].m is not None}, v: {self.custom_opts[key].v is not None}, t: {self.custom_opts[key].t}\n"
                        f"  This indicates a bug in the custom optimizer's step() method.\n"
                        f"  Check src/core/optimizers.py Adam implementation."
                    )
                
                # Reshape and update parameter preserving dtype/device
                updated_tensor = torch.from_numpy(updated_param.reshape(param_np.shape))
                p.data.copy_(updated_tensor.to(p.data.dtype).to(p.device))
        
        return loss
    
    def state_dict(self):
        """Save custom Adam optimizer states with index-based keys.

        Keeps numpy arrays instead of .tolist() for efficient serialization.
        """
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            # Keep as numpy arrays for efficient torch.save
            custom_state[key_str] = {
                'm': opt.m if opt.m is not None else None,
                'v': opt.v if opt.v is not None else None,
                't': opt.t
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """Restore custom Adam states using index mapping."""
        import numpy as np
        custom_state = state_dict.pop('custom_opts', {})
        super().load_state_dict(state_dict)
        
        self.custom_opts = {}
        for key_str, opt_state in custom_state.items():
            # Parse string key back to tuple
            group_idx, param_idx = map(int, key_str.split(','))
            key = (group_idx, param_idx)
            
            # Validate indices
            if group_idx < len(self.param_groups):
                group = self.param_groups[group_idx]
                if param_idx < len(group['params']):
                    opt = CustomAdam(
                        lr=group['lr'], beta1=group['beta1'],
                        beta2=group['beta2'], epsilon=group['epsilon']
                    )
                    opt.m = np.array(opt_state['m'], dtype=np.float32) if opt_state['m'] is not None else None
                    opt.v = np.array(opt_state['v'], dtype=np.float32) if opt_state['v'] is not None else None
                    opt.t = opt_state['t']
                    self.custom_opts[key] = opt


class SGDNesterovWrapper(Optimizer):
    """PyTorch wrapper for custom SGD with Nesterov (NAG)."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.custom_opts = {}
        
        # Flag for OOM handler - SGDNesterov does not require closure
        self.requires_closure = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Use (group_idx, param_idx) as key for cross-process safety
                key = (group_idx, param_idx)
                
                if key not in self.custom_opts:
                    self.custom_opts[key] = CustomSGDNesterov(
                        lr=group['lr'],
                        beta=group['momentum']
                    )
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Validate shape before reshaping
                if updated_param.size != param_np.size:
                    param_name = f"group{group_idx}_param{param_idx}"
                    raise ValueError(
                        f"SGDNesterovWrapper: Shape mismatch in {param_name}:\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\n"
                        f"  Returned size: {updated_param.size}\n"
                        f"  Param device: {p.device}, dtype: {original_dtype}\n"
                        f"  Optimizer state - v: {self.custom_opts[key].v is not None}\n"
                        f"  This indicates a bug in the custom optimizer's step() method.\n"
                        f"  Check src/core/optimizers.py SGDNesterov implementation."
                    )
                
                # Reshape and update parameter preserving dtype/device
                try:
                    updated_tensor = torch.from_numpy(updated_param.reshape(original_shape))
                    p.data.copy_(updated_tensor.to(original_dtype).to(p.device))
                except Exception as e:
                    raise RuntimeError(f"Failed to update parameter: {e}") from e

        return loss
    
    def state_dict(self):
        """Save Nesterov momentum states with index-based keys.

        Keeps numpy arrays for efficient serialization.
        """
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'v': opt.v if opt.v is not None else None,
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """Restore Nesterov states using index mapping."""
        import numpy as np
        custom_state = state_dict.pop('custom_opts', {})
        super().load_state_dict(state_dict)
        
        self.custom_opts = {}
        for key_str, opt_state in custom_state.items():
            # Parse string key back to tuple
            group_idx, param_idx = map(int, key_str.split(','))
            key = (group_idx, param_idx)
            
            # Validate indices
            if group_idx < len(self.param_groups):
                group = self.param_groups[group_idx]
                if param_idx < len(group['params']):
                    opt = CustomSGDNesterov(lr=group['lr'], beta=group['momentum'])
                    opt.v = np.array(opt_state['v'], dtype=np.float32) if opt_state['v'] is not None else None
                    self.custom_opts[key] = opt


class RMSPropWrapper(Optimizer):
    """PyTorch wrapper for custom RMSProp optimizer."""
    
    def __init__(self, params, lr=0.01, alpha=0.99, epsilon=1e-8):
        defaults = dict(lr=lr, alpha=alpha, epsilon=epsilon)
        super().__init__(params, defaults)
        # Create one optimizer per parameter (they have state)
        self.custom_opts = {}
        
        # Flag for OOM handler - RMSProp does not require closure
        self.requires_closure = False
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Use (group_idx, param_idx) as key for cross-process safety
                key = (group_idx, param_idx)
                
                # Initialize optimizer for this parameter if needed
                if key not in self.custom_opts:
                    # Map torch's alpha (smoothing) -> decay_rate in custom RMSProp
                    self.custom_opts[key] = CustomRMSProp(
                        lr=group['lr'],
                        decay_rate=group['alpha'],
                        epsilon=group['epsilon']
                    )
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Validate shape before reshaping
                if updated_param.size != param_np.size:
                    param_name = f"group{group_idx}_param{param_idx}"
                    raise ValueError(
                        f"RMSPropWrapper: Shape mismatch in {param_name}:\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\n"
                        f"  Returned size: {updated_param.size}\n"
                        f"  Param device: {p.device}, dtype: {original_dtype}\n"
                        f"  Optimizer state - s: {self.custom_opts[key].s is not None}\n"
                        f"  This indicates a bug in the custom optimizer's step() method.\n"
                        f"  Check src/core/optimizers.py RMSProp implementation."
                    )
                
                # Reshape and update parameter preserving dtype/device
                try:
                    updated_tensor = torch.from_numpy(updated_param.reshape(original_shape))
                    p.data.copy_(updated_tensor.to(original_dtype).to(p.device))
                except Exception as e:
                    raise RuntimeError(f"Failed to update parameter: {e}") from e
        
        return loss
    
    def state_dict(self):
        """Save RMSProp states with index-based keys.

        Keeps numpy arrays for efficient serialization.
        """
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                's': opt.s if opt.s is not None else None
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """Restore RMSProp states using index mapping."""
        import numpy as np
        custom_state = state_dict.pop('custom_opts', {})
        super().load_state_dict(state_dict)
        
        self.custom_opts = {}
        for key_str, opt_state in custom_state.items():
            # Parse string key back to tuple
            group_idx, param_idx = map(int, key_str.split(','))
            key = (group_idx, param_idx)
            
            # Validate indices
            if group_idx < len(self.param_groups):
                group = self.param_groups[group_idx]
                if param_idx < len(group['params']):
                    opt = CustomRMSProp(
                        lr=group['lr'], decay_rate=group['alpha'],
                        epsilon=group['epsilon']
                    )
                    opt.s = np.array(opt_state['s'], dtype=np.float32) if opt_state['s'] is not None else None
                    self.custom_opts[key] = opt
            else:
                import logging
                logging.warning("Skipping invalid optimizer state for key %s (indices out of bounds)", key_str)


class AdamWWrapper(Optimizer):
    """PyTorch wrapper for custom AdamW optimizer (decoupled weight decay)."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.custom_opts = {}
        
        # Flag for OOM handler - AdamW does not require closure
        self.requires_closure = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Use (group_idx, param_idx) as key for cross-process safety
                key = (group_idx, param_idx)
                
                if key not in self.custom_opts:
                    self.custom_opts[key] = CustomAdamW(
                        lr=group['lr'],
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=group['eps'],
                        weight_decay=group['weight_decay']
                    )
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy()
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Validate shape before reshaping
                if updated_param.size != param_np.size:
                    param_name = f"group{group_idx}_param{param_idx}"
                    raise ValueError(
                        f"AdamWWrapper: Shape mismatch in {param_name}:\n"
                        f"  Expected size: {param_np.size} (shape {original_shape})\n"
                        f"  Returned size: {updated_param.size}\n"
                        f"  Param device: {p.device}, dtype: {original_dtype}\n"
                        f"  Optimizer state - m: {self.custom_opts[key].m is not None}, v: {self.custom_opts[key].v is not None}, t: {self.custom_opts[key].t}\n"
                        f"  This indicates a bug in the custom optimizer's step() method.\n"
                        f"  Check src/core/optimizers.py AdamW implementation."
                    )
                
                # Reshape and update parameter preserving dtype/device
                try:
                    updated_tensor = torch.from_numpy(updated_param.reshape(original_shape))
                    p.data.copy_(updated_tensor.to(original_dtype).to(p.device))
                except Exception as e:
                    raise RuntimeError(f"Failed to update parameter: {e}") from e

        return loss
    
    def state_dict(self):
        """Save AdamW states with index-based keys.

        Keeps numpy arrays for efficient serialization.
        """
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'm': opt.m if opt.m is not None else None,
                'v': opt.v if opt.v is not None else None,
                't': opt.t
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """Restore AdamW states using index mapping."""
        import numpy as np
        custom_state = state_dict.pop('custom_opts', {})
        super().load_state_dict(state_dict)
        
        self.custom_opts = {}
        for key_str, opt_state in custom_state.items():
            # Parse string key back to tuple
            group_idx, param_idx = map(int, key_str.split(','))
            key = (group_idx, param_idx)
            
            # Validate indices
            if group_idx < len(self.param_groups):
                group = self.param_groups[group_idx]
                beta1, beta2 = group['betas']
                if param_idx < len(group['params']):
                    opt = CustomAdamW(
                        lr=group['lr'], beta1=beta1, beta2=beta2,
                        epsilon=group['eps'], weight_decay=group['weight_decay']
                    )
                    opt.m = np.array(opt_state['m'], dtype=np.float32) if opt_state['m'] is not None else None
                    opt.v = np.array(opt_state['v'], dtype=np.float32) if opt_state['v'] is not None else None
                    opt.t = opt_state['t']
                    self.custom_opts[key] = opt


class SAMWrapper(Optimizer):
    """
    Unified SAM (Sharpness-Aware Minimization) wrapper for PyTorch optimizers.
    
    Compatible with any base optimizer (SGD, Adam, AdamW, etc.) and supports
    both closure-based and standard step() interfaces.
    
    Usage:
        # With torch optimizers
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = SAMWrapper(base_opt, rho=0.05)
        
        # Training loop
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
    
    Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently 
               Improving Generalization", ICLR 2021
    """
    
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):  # pylint: disable=super-init-not-called
        """
        Initialize SAM optimizer wrapper.
        
        Args:
            base_optimizer: Any PyTorch optimizer instance (SGD, Adam, etc.)
            rho: Neighborhood size for sharpness (default: 0.05)
            adaptive: Use adaptive SAM variant (default: False)
        
        Note:
            Does not call super().__init__() because SAM is a wrapper that 
            delegates all optimization to base_optimizer. param_groups and 
            state are inherited by reference from the base optimizer.
        """
        # SAM wraps an existing optimizer - no super().__init__() needed
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        
        # Flag for OOM handler - SAM requires closure
        self.requires_closure = True
        
        # Inherit param_groups and state from base optimizer (reference, not copy)
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        
        # Initialize Optimizer parent without calling __init__ to avoid empty param error
        # We manually set the required attributes that Optimizer expects
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        
        # Local container for adversarial perturbations to avoid mutating
        # base_optimizer.state entries, which can interfere with lazy
        # initialization of base optimizers (e.g., Adam's 'exp_avg').
        self._perturbations = {}
        
        # Track sharpness metric for telemetry
        self.sharpness_history = []  # List of (step, sharpness) tuples
        self._step_count = 0
    
    @torch.no_grad()
    def _get_grad_norm(self):
        """Compute gradient norm across all parameters."""
        # Collect per-parameter gradient norms
        grads = []
        shared_device = None
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if shared_device is None:
                        shared_device = p.device
                    grads.append(p.grad.detach().view(-1))
        
        # Guard against empty list (no gradients present)
        if not grads:
            return torch.tensor(0.0, device=shared_device if shared_device else torch.device('cpu'), dtype=torch.float32)
        
        # Concatenate all gradients and compute overall norm
        all_grads = torch.cat(grads)
        norm = torch.norm(all_grads, p=2)
        
        return norm
    
    @torch.no_grad()
    def _ascent_step(self):
        """Take adversarial step in direction of gradient."""
        grad_norm = self._get_grad_norm()
        
        # Properly handle scale tensor with correct device/dtype
        # Avoid division by zero with small epsilon
        scale = self.rho / (grad_norm + 1e-12)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Ensure scale matches parameter device/dtype
                scale_p = scale.to(device=p.device, dtype=p.dtype)
                
                # Compute perturbation with proper dtype handling
                if self.adaptive:
                    # Adaptive SAM: weight perturbations by parameter magnitude
                    e_w = (torch.abs(p).pow(2)) * p.grad * scale_p
                else:
                    # Standard SAM: uniform perturbation direction
                    e_w = p.grad * scale_p
                
                p.add_(e_w)  # Move to adversarial point
                # Store perturbation for later restoration in local map
                # (do NOT write into base optimizer state dict)
                self._perturbations[id(p)] = e_w
    
    @torch.no_grad()
    def _descent_step(self):
        """Restore parameters and apply base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameters using locally stored perturbations
                e_w = self._perturbations.pop(id(p), None)
                if e_w is not None:
                    p.sub_(e_w)
        
        # Apply base optimizer update with adversarial gradients
        self.base_optimizer.step()
    
    def step(self, closure=None):
        """
        Perform SAM optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
                     REQUIRED for SAM to compute adversarial gradients.
        
        Returns:
            loss value from closure
        """
        if closure is None:
            raise ValueError(
                "SAMWrapper: closure is required but was None.\\n"
                "\\nSAM requires a closure function to compute adversarial gradients.\\n"
                "\\nEXAMPLE USAGE:\\n"
                "  def closure():\\n"
                "      optimizer.zero_grad()\\n"
                "      output = model(input)\\n"
                "      loss = criterion(output, target)\\n"
                "      loss.backward()\\n"
                "      return loss\\n"
                "\\n"
                "  loss = optimizer.step(closure)\\n"
                "\\nSee PyTorch LBFGS optimizer docs for closure examples."
            )
        
        # Validate closure is callable
        if not callable(closure):
            raise TypeError(
                f"SAMWrapper: closure must be callable, got {type(closure).__name__}"
            )
        
        # First forward-backward pass (compute gradients at current point)
        try:
            loss = closure()
        except Exception as e:
            raise RuntimeError(
                f"SAMWrapper: closure() failed during first forward pass: {e}"
            ) from e
        
        # Normalize loss to a Python float (accept Tensors or numeric scalars)
        from src.utils.num_utils import safe_to_float
        loss_at_current = safe_to_float(loss)
        if math.isnan(loss_at_current):
            raise RuntimeError(
                f"SAMWrapper: closure must return a Tensor or numeric scalar, got {type(loss).__name__}"
            ) from None
        
        # Check for non-finite loss
        if not math.isfinite(loss_at_current):
            raise ValueError(
                f"SAMWrapper: Non-finite loss detected at current point: {loss_at_current}\\n"
                "This indicates numerical instability. Consider: gradient clipping, smaller LR, or AMP."
            )
        
        # Save current parameters and take adversarial step
        self._ascent_step()
        
        # Second forward-backward pass (compute gradients at adversarial point)
        try:
            loss_adv = closure()  # Recompute loss and gradients at perturbed parameters
        except Exception as e:
            # Restore parameters before raising
            self._descent_step()
            raise RuntimeError(
                f"SAMWrapper: closure() failed during adversarial forward pass: {e}"
            ) from e
        
        # Normalize adversarial loss to Python float using safe coercion
        loss_at_adversarial = safe_to_float(loss_adv)
        if math.isnan(loss_at_adversarial):
            # Restore parameters before raising
            for group in self.param_groups:
                for p in group["params"]:
                    e_w = self._perturbations.pop(id(p), None)
                    if e_w is not None:
                        p.sub_(e_w)
            raise RuntimeError(
                f"SAMWrapper: closure must return a Tensor or numeric scalar at adversarial point, got {type(loss_adv).__name__}"
            )

        # Check for non-finite adversarial loss
        if not math.isfinite(loss_at_adversarial):
            # Restore parameters
            for group in self.param_groups:
                for p in group["params"]:
                    e_w = self._perturbations.pop(id(p), None)
                    if e_w is not None:
                        p.sub_(e_w)
            raise ValueError(
                f"SAMWrapper: Non-finite loss at adversarial point: {loss_at_adversarial}\\n"
                "SAM perturbation may have caused overflow. Try smaller rho parameter."
            )
        
        # Track sharpness (loss difference between adversarial and current point)
        sharpness = abs(loss_at_adversarial - loss_at_current)
        self._step_count += 1
        self.sharpness_history.append((self._step_count, sharpness))
        
        # Restore parameters and apply update
        self._descent_step()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Delegate to base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_sharpness_history(self):
        """Get sharpness tracking history for analysis.
        
        Returns:
            List of (step, sharpness) tuples tracking loss landscape sharpness
        """
        return self.sharpness_history.copy()
    
    def get_average_sharpness(self, last_n_steps=None):
        """Get average sharpness over recent steps.
        
        Args:
            last_n_steps: Number of recent steps to average (None = all steps)
            
        Returns:
            Average sharpness value
        """
        if not self.sharpness_history:
            return 0.0
        
        history_slice = self.sharpness_history[-last_n_steps:] if last_n_steps else self.sharpness_history
        return sum(s for _, s in history_slice) / len(history_slice) if history_slice else 0.0
    
    def state_dict(self):
        """Return state dict including base optimizer state."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'rho': self.rho,
            'adaptive': self.adaptive,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.rho = state_dict['rho']
        self.adaptive = state_dict.get('adaptive', False)


# Legacy aliases for backward compatibility
class SAMSGDWrapper(SAMWrapper):
    """
    DEPRECATED: Use SAMWrapper(torch.optim.SGD(...), rho=0.05) instead.
    
    Legacy wrapper for SAM with SGD base optimizer.
    Maintained for backward compatibility only.
    """
    def __init__(self, params, lr=0.01, momentum=0.0, rho=0.05, weight_decay=0.0):
        import warnings
        warnings.warn(
            "SAMSGDWrapper is deprecated. Use: "
            "SAMWrapper(torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay), rho=rho)",
            DeprecationWarning,
            stacklevel=2
        )
        base_opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(base_opt, rho=rho)


class SAMAdamWrapper(SAMWrapper):
    """
    DEPRECATED: Use SAMWrapper(torch.optim.Adam(...), rho=0.05) instead.
    
    Legacy wrapper for SAM with Adam base optimizer.
    Maintained for backward compatibility only.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, rho=0.05, weight_decay=0.0):
        import warnings
        warnings.warn(
            "SAMAdamWrapper is deprecated. Use: "
            "SAMWrapper(torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay), rho=rho)",
            DeprecationWarning,
            stacklevel=2
        )
        base_opt = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(base_opt, rho=rho)


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
        
        # Flag for OOM handler - Lookahead does not require closure
        self.requires_closure = False
        
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
        
        # Handle None returns gracefully: many PyTorch optimizers (e.g., SGD) return None
        # We normalize to numeric scalar for downstream consumers
        if loss is None:
            logging.debug("LookaheadWrapper: base optimizer.step returned None; normalizing to tensor(0.0)")
            try:
                loss_value = torch.tensor(0.0)
            except (RuntimeError, TypeError):
                loss_value = 0.0
        else:
            loss_value = loss
        
        # Increment step counter
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    # Lookahead: slow = slow + α * (fast - slow) = (1-α) * slow + α * fast
                    alpha = group['alpha']
                    self.slow_params[idx] = (1 - alpha) * self.slow_params[idx] + alpha * p.data
                    p.data.copy_(self.slow_params[idx])
                    idx += 1
        
        return loss_value
    
    def state_dict(self):
        """Return state dict including base optimizer and slow params state."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'slow_params': [p.clone() for p in self.slow_params],
            'step_count': self.step_count,
            'k': self.k,
            'alpha': self.alpha,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict and restore slow params."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.slow_params = [p.clone() for p in state_dict['slow_params']]
        self.step_count = state_dict['step_count']
        self.k = state_dict['k']
        self.alpha = state_dict['alpha']


def test_sam_and_lookahead():
    """Test SAM and Lookahead optimizers."""
    logging.info("Testing SAM and Lookahead optimizers...")    
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    criterion = torch.nn.MSELoss()
    
    # Test data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Test SAM
    logging.info("  Testing SAM...")
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sam_opt = SAMWrapper(base_optimizer, rho=0.05)
    
    def closure():
        sam_opt.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        return loss
    
    try:
        loss = sam_opt.step(closure)
        # Normalize to float safely in case loss is a Tensor or Python number
        if isinstance(loss, torch.Tensor):
            loss_val = float(loss.item())
        else:
            try:
                loss_val = float(cast(Any, loss))
            except (ValueError, TypeError, RuntimeError):
                logging.warning("SAM step returned non-numeric loss: %s", type(loss))
                loss_val = float('nan')
        logging.info("  SAM step completed successfully, loss: %.4f", loss_val)
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("  SAM failed: %s", e, exc_info=True)
    
    # Test Lookahead
    logging.info("  Testing Lookahead...")
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
        if isinstance(loss, torch.Tensor):
            loss_val = float(loss.item())
        else:
            try:
                loss_val = float(cast(Any, loss))
            except (ValueError, TypeError, RuntimeError):
                logging.warning("Lookahead step returned non-numeric loss: %s", type(loss))
                loss_val = float('nan')
        logging.info("  Lookahead step completed successfully, loss: %.4f", loss_val)
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("  Lookahead failed: %s", e, exc_info=True)
    
    logging.info("\nSAM and Lookahead optimizer wrappers tested!")

class AdaBoundWrapper(Optimizer):
    """PyTorch wrapper for custom AdaBound optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, epsilon=1e-8, gamma=1e-3):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, final_lr=final_lr, epsilon=epsilon, gamma=gamma)
        super().__init__(params, defaults)
        
        # Flag for OOM handler - AdaBound does not require closure
        self.requires_closure = False
        
        # Create separate optimizer for each parameter group
        self.custom_opts = {}
        for group_id, group in enumerate(self.param_groups):
            self.custom_opts[group_id] = CustomAdaBound(
                lr=group['lr'],
                beta1=group['beta1'],
                beta2=group['beta2'],
                final_lr=group['final_lr'],
                epsilon=group['epsilon'],
                gamma=group['gamma']
            )
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_id, group in enumerate(self.param_groups):
            opt = self.custom_opts[group_id]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy().copy()
                
                # Flatten for optimizer
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Validate size before reshape
                if new_param_flat.size != param_flat.size:
                    raise ValueError(
                        f"AdaBound optimizer returned wrong size: expected {param_flat.size}, "
                        f"got {new_param_flat.size}. Original shape: {original_shape}"
                    )
                
                # Reshape and copy back
                try:
                    new_param = new_param_flat.reshape(original_shape)
                    p.data.copy_(torch.from_numpy(new_param).to(p.device, dtype=original_dtype))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reshape AdaBound output: {e}. "
                        f"Original shape: {original_shape}, flat size: {new_param_flat.size}"
                    )
        
        return loss


class RAdamWrapper(Optimizer):
    """PyTorch wrapper for custom RAdam optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)
        
        # Flag for OOM handler - RAdam does not require closure
        self.requires_closure = False
        
        # Create separate optimizer for each parameter group
        self.custom_opts = {}
        for group_id, group in enumerate(self.param_groups):
            self.custom_opts[group_id] = CustomRAdam(
                lr=group['lr'],
                beta1=group['beta1'],
                beta2=group['beta2'],
                epsilon=group['epsilon']
            )
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_id, group in enumerate(self.param_groups):
            opt = self.custom_opts[group_id]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy().copy()
                
                # Flatten for optimizer
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Validate size before reshape
                if new_param_flat.size != param_flat.size:
                    raise ValueError(
                        f"RAdam optimizer returned wrong size: expected {param_flat.size}, "
                        f"got {new_param_flat.size}. Original shape: {original_shape}"
                    )
                
                # Reshape and copy back
                try:
                    new_param = new_param_flat.reshape(original_shape)
                    p.data.copy_(torch.from_numpy(new_param).to(p.device, dtype=original_dtype))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reshape RAdam output: {e}. "
                        f"Original shape: {original_shape}, flat size: {new_param_flat.size}"
                    )
        
        return loss


class LAMBWrapper(Optimizer):
    """PyTorch wrapper for custom LAMB optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Flag for OOM handler - LAMB does not require closure
        self.requires_closure = False
        
        # Create separate optimizer for each parameter group
        self.custom_opts = {}
        for group_id, group in enumerate(self.param_groups):
            self.custom_opts[group_id] = CustomLAMB(
                lr=group['lr'],
                beta1=group['beta1'],
                beta2=group['beta2'],
                epsilon=group['epsilon'],
                weight_decay=group['weight_decay']
            )
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_id, group in enumerate(self.param_groups):
            opt = self.custom_opts[group_id]
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient as numpy
                grad = p.grad.data.cpu().numpy()
                param_np = p.data.cpu().numpy().copy()
                
                # Flatten for optimizer
                original_shape = param_np.shape
                original_dtype = p.data.dtype
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Validate size before reshape
                if new_param_flat.size != param_flat.size:
                    raise ValueError(
                        f"LAMB optimizer returned wrong size: expected {param_flat.size}, "
                        f"got {new_param_flat.size}. Original shape: {original_shape}"
                    )
                
                # Reshape and copy back
                try:
                    new_param = new_param_flat.reshape(original_shape)
                    p.data.copy_(torch.from_numpy(new_param).to(p.device, dtype=original_dtype))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reshape LAMB output: {e}. "
                        f"Original shape: {original_shape}, flat size: {new_param_flat.size}"
                    )
        
        return loss


if __name__ == '__main__':
    # Test the wrappers
    logging.info("Testing PyTorch optimizer wrappers...")    
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
        logging.info("\nTesting %s:", name)        
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
        
    logging.info("\\nAll optimizer wrappers work correctly!")
    
    # Test new optimizers
    logging.info("\\n%s", "="*60)
    logging.info("Testing new optimizer wrappers (AdaBound, RAdam, LAMB)...")
    logging.info("%s", "="*60)
    
    model = torch.nn.Linear(10, 2)
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    new_optimizers = {
        'AdaBound': AdaBoundWrapper(model.parameters(), lr=0.001, final_lr=0.1),
        'RAdam': RAdamWrapper(model.parameters(), lr=0.001),
        'LAMB': LAMBWrapper(model.parameters(), lr=0.001, weight_decay=0.01)
    }
    
    for name, optimizer in new_optimizers.items():
        logging.info("\n  Testing %s...", name)
        model_test = torch.nn.Linear(10, 2)
        opt_test = new_optimizers[name].__class__(model_test.parameters(), lr=0.001)
        
        try:
            opt_test.zero_grad()
            output = model_test(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            opt_test.step()
            logging.info("    %s step completed successfully, loss: %.4f", name, loss.item())
        except (RuntimeError, ValueError, TypeError) as e:
            logging.info("    %s failed: %s", name, e)
    
    logging.info("\nNew optimizer wrappers tested!")
    logging.info("\nAll optimizer wrappers working!")
    
    test_sam_and_lookahead()


# =============================================================================
# PUBLIC API ALIASES
# =============================================================================
# Provide clean names for external imports
SAM = SAMWrapper
Lookahead = LookaheadWrapper
AdaBound = AdaBoundWrapper
RAdam = RAdamWrapper
LAMB = LAMBWrapper

# All available optimizer wrappers
__all__ = [
    'SGDWrapper', 'SGDMomentumWrapper', 'AdamWrapper', 'SGDNesterovWrapper',
    'RMSPropWrapper', 'AdamWWrapper', 'SAMWrapper', 'LookaheadWrapper',
    'AdaBoundWrapper', 'RAdamWrapper', 'LAMBWrapper',
    # Aliases for convenience
    'SAM', 'Lookahead', 'AdaBound', 'RAdam', 'LAMB'
]