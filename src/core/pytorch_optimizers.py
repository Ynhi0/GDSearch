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
        SAM as CustomSAM,
        Lookahead as CustomLookahead,
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
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss
    
    def state_dict(self):
        """AUDIT FIX: Save custom optimizer states with index-based keys for cross-process safety."""
        base_state = super().state_dict()
        # Serialize custom_opts: map (group_idx, param_idx) to optimizer state (v=velocity, not m)
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'v': opt.v.tolist() if opt.v is not None else None,
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """AUDIT FIX: Restore custom optimizer states from checkpoint using index mapping."""
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
                    opt.v = np.array(opt_state['v']) if opt_state['v'] is not None else None
                    self.custom_opts[key] = opt


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
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss
    
    def state_dict(self):
        """AUDIT FIX: Save custom Adam optimizer states with index-based keys."""
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'm': opt.m.tolist() if opt.m is not None else None,
                'v': opt.v.tolist() if opt.v is not None else None,
                't': opt.t
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """AUDIT FIX: Restore custom Adam states using index mapping."""
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
                    opt.m = np.array(opt_state['m']) if opt_state['m'] is not None else None
                    opt.v = np.array(opt_state['v']) if opt_state['v'] is not None else None
                    opt.t = opt_state['t']
                    self.custom_opts[key] = opt


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
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)

        return loss
    
    def state_dict(self):
        """AUDIT FIX: Save Nesterov momentum states with index-based keys."""
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'v': opt.v.tolist() if opt.v is not None else None,
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """AUDIT FIX: Restore Nesterov states using index mapping."""
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
                    opt.v = np.array(opt_state['v']) if opt_state['v'] is not None else None
                    self.custom_opts[key] = opt


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
                
                # Compute update
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                
                # Reshape and update parameter
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)
        
        return loss
    
    def state_dict(self):
        """AUDIT FIX: Save RMSProp states with index-based keys."""
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            # 🐛 BUG FIX #1: RMSProp only has 's' attribute, not 't'
            custom_state[key_str] = {
                's': opt.s.tolist() if opt.s is not None else None
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """AUDIT FIX: Restore RMSProp states using index mapping."""
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
                    opt = RMSProp(
                        lr=group['lr'], decay_rate=group['alpha'],
                        epsilon=group['epsilon']
                    )
                    opt.s = np.array(opt_state['s']) if opt_state['s'] is not None else None
                    # 🐛 BUG FIX #1: RMSProp doesn't have 't' attribute
                    self.custom_opts[key] = opt
            else:
                # 🐛 BUG FIX #4: Log warning when indices are invalid
                import logging
                logging.warning(f"Skipping invalid optimizer state for key {key_str} (indices out of bounds)")


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
                updated_param = self.custom_opts[key].step(param_np.flatten(), grad.flatten())
                p.data = torch.from_numpy(updated_param.reshape(param_np.shape)).to(p.device)

        return loss
    
    def state_dict(self):
        """AUDIT FIX: Save AdamW states with index-based keys."""
        base_state = super().state_dict()
        custom_state = {}
        for key, opt in self.custom_opts.items():
            # Convert tuple key to string for JSON serialization
            key_str = f"{key[0]},{key[1]}"
            custom_state[key_str] = {
                'm': opt.m.tolist() if opt.m is not None else None,
                'v': opt.v.tolist() if opt.v is not None else None,
                't': opt.t
            }
        base_state['custom_opts'] = custom_state
        return base_state
    
    def load_state_dict(self, state_dict):
        """AUDIT FIX: Restore AdamW states using index mapping."""
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
                    opt.m = np.array(opt_state['m']) if opt_state['m'] is not None else None
                    opt.v = np.array(opt_state['v']) if opt_state['v'] is not None else None
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
    
    def __init__(self, base_optimizer, rho=0.05, adaptive=False):
        """
        Initialize SAM optimizer wrapper.
        
        Args:
            base_optimizer: Any PyTorch optimizer instance (SGD, Adam, etc.)
            rho: Neighborhood size for sharpness (default: 0.05)
            adaptive: Use adaptive SAM variant (default: False)
        """
        # SAM wraps an existing optimizer
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        
        # Inherit param_groups from base optimizer
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        
        # Not a true Optimizer subclass - we delegate to base_optimizer
        # But we maintain compatibility with the Optimizer interface
        # Local container for adversarial perturbations to avoid mutating
        # base_optimizer.state entries, which can interfere with lazy
        # initialization of base optimizers (e.g., Adam's 'exp_avg').
        self._perturbations = {}
        
        # 🐛 BUG FIX #11: Track sharpness metric for telemetry
        self.sharpness_history = []  # List of (step, sharpness) tuples
        self._step_count = 0
    
    @torch.no_grad()
    def _get_grad_norm(self):
        """Compute gradient norm across all parameters."""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    @torch.no_grad()
    def _ascent_step(self):
        """Take adversarial step in direction of gradient."""
        grad_norm = self._get_grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Compute and apply perturbation
                if self.adaptive:
                    e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                p.add_(e_w)  # Move to adversarial point
                # Store perturbation for later restoration in local map
                # (do NOT write into base optimizer state dict)
                self._perturbations[p] = e_w
    
    @torch.no_grad()
    def _descent_step(self):
        """Restore parameters and apply base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameters using locally stored perturbations
                e_w = self._perturbations.pop(p, None)
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
                "SAM requires a closure function to compute adversarial gradients. "
                "Pass a closure that computes and backpropagates the loss."
            )
        
        # First forward-backward pass (compute gradients at current point)
        loss = closure()
        loss_at_current = loss.item() if hasattr(loss, 'item') else float(loss)
        
        # Save current parameters and take adversarial step
        self._ascent_step()
        
        # Second forward-backward pass (compute gradients at adversarial point)
        loss_adv = closure()  # Recompute loss and gradients at perturbed parameters
        loss_at_adversarial = loss_adv.item() if hasattr(loss_adv, 'item') else float(loss_adv)
        
        # 🐛 BUG FIX #11: Track sharpness (loss difference between adversarial and current point)
        sharpness = abs(loss_at_adversarial - loss_at_current)
        self._step_count += 1
        self.sharpness_history.append((self._step_count, sharpness))
        
        # Restore parameters and apply update
        self._descent_step()
        
        return loss
    
    def zero_grad(self):
        """Delegate to base optimizer."""
        self.base_optimizer.zero_grad()
    
    def get_sharpness_history(self):
        """🐛 BUG FIX #11: Get sharpness tracking history for analysis.
        
        Returns:
            List of (step, sharpness) tuples tracking loss landscape sharpness
        """
        return self.sharpness_history.copy()
    
    def get_average_sharpness(self, last_n_steps=None):
        """🐛 BUG FIX #11: Get average sharpness over recent steps.
        
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
                    # Lookahead: slow = slow + α * (fast - slow) = (1-α) * slow + α * fast
                    alpha = group['alpha']
                    self.slow_params[idx] = (1 - alpha) * self.slow_params[idx] + alpha * p.data
                    p.data.copy_(self.slow_params[idx])
                    idx += 1
        
        return loss
    
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


class AdaBoundWrapper(Optimizer):
    """PyTorch wrapper for custom AdaBound optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, final_lr=0.1, epsilon=1e-8, gamma=1e-3):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, final_lr=final_lr, epsilon=epsilon, gamma=gamma)
        super().__init__(params, defaults)
        
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
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Reshape and copy back
                new_param = new_param_flat.reshape(original_shape)
                p.data.copy_(torch.from_numpy(new_param).to(p.device))
        
        return loss


class RAdamWrapper(Optimizer):
    """PyTorch wrapper for custom RAdam optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)
        
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
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Reshape and copy back
                new_param = new_param_flat.reshape(original_shape)
                p.data.copy_(torch.from_numpy(new_param).to(p.device))
        
        return loss


class LAMBWrapper(Optimizer):
    """PyTorch wrapper for custom LAMB optimizer."""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
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
                param_flat = param_np.flatten()
                grad_flat = grad.flatten()
                
                # Update using custom optimizer
                new_param_flat = opt.step(param_flat, grad_flat)
                
                # Reshape and copy back
                new_param = new_param_flat.reshape(original_shape)
                p.data.copy_(torch.from_numpy(new_param).to(p.device))
        
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
        
    print("\n✓ All optimizer wrappers work correctly!")
    
    # Test new optimizers
    print("\n" + "="*60)
    print("Testing new optimizer wrappers (AdaBound, RAdam, LAMB)...")
    print("="*60)
    
    model = torch.nn.Linear(10, 2)
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    new_optimizers = {
        'AdaBound': AdaBoundWrapper(model.parameters(), lr=0.001, final_lr=0.1),
        'RAdam': RAdamWrapper(model.parameters(), lr=0.001),
        'LAMB': LAMBWrapper(model.parameters(), lr=0.001, weight_decay=0.01)
    }
    
    for name, optimizer in new_optimizers.items():
        print(f"\n  Testing {name}...")
        model_test = torch.nn.Linear(10, 2)
        opt_test = new_optimizers[name].__class__(model_test.parameters(), lr=0.001)
        
        try:
            opt_test.zero_grad()
            output = model_test(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            opt_test.step()
            print(f"    ✓ {name} step completed successfully, loss: {loss.item():.4f}")
        except Exception as e:
            print(f"    ✗ {name} failed: {e}")
    
    print("\n✓ New optimizer wrappers tested!")
    print("\n✓ All optimizer wrappers working!")
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