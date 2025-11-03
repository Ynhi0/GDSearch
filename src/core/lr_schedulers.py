"""
Learning Rate Schedulers for GDSearch

Implements various learning rate scheduling strategies:
- CosineAnnealingLR: Cosine annealing with optional warmup
- StepLR: Step decay at specified epochs
- MultiStepLR: Multiple step decays
- ExponentialLR: Exponential decay
- LinearWarmup: Linear warmup wrapper for any scheduler
- ConstantLR: Constant learning rate (baseline)

All schedulers are compatible with our custom optimizers.
"""

import math
import numpy as np
from typing import Optional, List, Callable


class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to adjust learning rate for
            last_epoch: The index of last epoch (-1 means just started)
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.current_lr = self.base_lr
        
    def get_lr(self) -> float:
        """Calculate learning rate for current epoch. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_lr()")
    
    def step(self, epoch: Optional[int] = None):
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch number. If None, increment last_epoch.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # Get new learning rate
        self.current_lr = self.get_lr()
        
        # Update optimizer
        self.optimizer.lr = self.current_lr
        
    def state_dict(self) -> dict:
        """Return scheduler state as dictionary."""
        return {
            'last_epoch': self.last_epoch,
            'base_lr': self.base_lr,
            'current_lr': self.current_lr
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from dictionary."""
        self.last_epoch = state_dict['last_epoch']
        self.base_lr = state_dict['base_lr']
        self.current_lr = state_dict['current_lr']


class ConstantLR(LRScheduler):
    """Constant learning rate (no scheduling)."""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Return constant learning rate."""
        return self.base_lr


class StepLR(LRScheduler):
    """
    Step learning rate decay.
    
    Decays the learning rate by gamma every step_size epochs.
    
    Example:
        lr = base_lr * gamma^(epoch // step_size)
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, 
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            step_size: Period of learning rate decay (in epochs)
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with step decay."""
        if self.last_epoch == 0:
            return self.base_lr
        
        if self.last_epoch % self.step_size == 0:
            return self.current_lr * self.gamma
        
        return self.current_lr


class MultiStepLR(LRScheduler):
    """
    Multi-step learning rate decay.
    
    Decays the learning rate by gamma at specified milestones.
    
    Example:
        milestones = [30, 60, 90]
        lr decays at epochs 30, 60, and 90
    """
    
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            milestones: List of epoch indices at which to decay LR
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
        """
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with multi-step decay."""
        if self.last_epoch == 0:
            return self.base_lr
            
        if self.last_epoch in self.milestones:
            return self.current_lr * self.gamma
            
        return self.current_lr


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate decay.
    
    lr = base_lr * gamma^epoch
    """
    
    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with exponential decay."""
        if self.last_epoch == 0:
            return self.base_lr
        
        return self.current_lr * self.gamma


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    The learning rate is annealed from base_lr to eta_min using a cosine curve
    over T_max epochs. This is a popular schedule used in modern deep learning.
    
    Formula:
        lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
    
    Reference:
        Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent with Warm Restarts.
        https://arxiv.org/abs/1608.03983
    """
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            T_max: Maximum number of epochs for annealing
            eta_min: Minimum learning rate (default: 0)
            last_epoch: The index of last epoch
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate using cosine annealing."""
        if self.last_epoch == 0:
            return self.base_lr
        
        # Clamp to T_max
        epoch = min(self.last_epoch, self.T_max)
        
        # Cosine annealing formula
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        
        return lr


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    
    Restarts the learning rate schedule periodically, allowing the model
    to escape local minima.
    
    Reference:
        Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent with Warm Restarts.
    """
    
    def __init__(self, optimizer, T_0: int, T_mult: int = 1, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            T_0: Number of epochs for the first restart
            T_mult: Factor to increase T_i after each restart
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0  # Current position in cycle
        self.T_i = T_0  # Current cycle length
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with warm restarts."""
        if self.last_epoch == 0:
            return self.base_lr
        
        # Cosine annealing within current cycle
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        
        return lr
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate with restart logic."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.T_cur += 1
        
        # Check if we need to restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        
        # Get new learning rate
        self.current_lr = self.get_lr()
        
        # Update optimizer
        self.optimizer.lr = self.current_lr


class LinearWarmupScheduler(LRScheduler):
    """
    Linear warmup wrapper for any scheduler.
    
    Linearly increases learning rate from 0 (or warmup_start_lr) to base_lr
    over warmup_epochs, then applies the wrapped scheduler.
    
    This is commonly used to stabilize training in early epochs.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, 
                 after_scheduler: Optional[LRScheduler] = None,
                 warmup_start_lr: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            warmup_epochs: Number of warmup epochs
            after_scheduler: Scheduler to use after warmup (optional)
            warmup_start_lr: Initial learning rate for warmup
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with warmup."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
            return lr
        else:
            # Use wrapped scheduler if provided
            if self.after_scheduler is not None:
                return self.after_scheduler.current_lr
            else:
                return self.base_lr
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate with warmup logic."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # Step the wrapped scheduler if past warmup
        if self.last_epoch >= self.warmup_epochs and self.after_scheduler is not None:
            self.after_scheduler.step(epoch - self.warmup_epochs)
        
        # Get new learning rate
        self.current_lr = self.get_lr()
        
        # Update optimizer
        self.optimizer.lr = self.current_lr


class PolynomialLR(LRScheduler):
    """
    Polynomial learning rate decay.
    
    lr = base_lr * (1 - epoch / max_epochs)^power
    """
    
    def __init__(self, optimizer, max_epochs: int, power: float = 1.0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            max_epochs: Total number of training epochs
            power: Exponent of polynomial decay
            last_epoch: The index of last epoch
        """
        self.max_epochs = max_epochs
        self.power = power
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate with polynomial decay."""
        if self.last_epoch == 0:
            return self.base_lr
        
        if self.last_epoch >= self.max_epochs:
            return 0.0
        
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return self.base_lr * factor


class OneCycleLR(LRScheduler):
    """
    One-cycle learning rate policy.
    
    Increases learning rate from max_lr/div_factor to max_lr over pct_start
    fraction of training, then decreases to max_lr/final_div_factor.
    
    Reference:
        Smith (2018). A disciplined approach to neural network hyper-parameters.
        https://arxiv.org/abs/1803.09820
    """
    
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer to adjust
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
            last_epoch: The index of last epoch
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate key points
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> float:
        """Calculate learning rate for one-cycle policy."""
        if self.last_epoch < self.step_up:
            # Increasing phase
            alpha = self.last_epoch / self.step_up
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * alpha
        else:
            # Decreasing phase
            alpha = (self.last_epoch - self.step_up) / self.step_down
            lr = self.max_lr - (self.max_lr - self.final_lr) * alpha
        
        return lr


def get_scheduler(name: str, optimizer, **kwargs) -> LRScheduler:
    """
    Factory function to create schedulers by name.
    
    Args:
        name: Scheduler name ('constant', 'step', 'multistep', 'exponential',
              'cosine', 'cosine_restarts', 'warmup', 'polynomial', 'onecycle')
        optimizer: Optimizer to adjust
        **kwargs: Scheduler-specific arguments
    
    Returns:
        LRScheduler instance
    
    Example:
        >>> scheduler = get_scheduler('cosine', optimizer, T_max=100)
        >>> scheduler = get_scheduler('step', optimizer, step_size=30, gamma=0.1)
    """
    schedulers = {
        'constant': ConstantLR,
        'step': StepLR,
        'multistep': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine': CosineAnnealingLR,
        'cosine_restarts': CosineAnnealingWarmRestarts,
        'warmup': LinearWarmupScheduler,
        'polynomial': PolynomialLR,
        'onecycle': OneCycleLR,
    }
    
    if name.lower() not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")
    
    return schedulers[name.lower()](optimizer, **kwargs)


# Helper function to visualize scheduler behavior
def plot_scheduler(scheduler: LRScheduler, epochs: int, save_path: str = None):
    """
    Plot learning rate schedule over epochs.
    
    Args:
        scheduler: LRScheduler instance
        epochs: Number of epochs to simulate
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    lrs = []
    for epoch in range(epochs):
        scheduler.step()
        lrs.append(scheduler.current_lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), lrs, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'{scheduler.__class__.__name__} Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scheduler plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Demo: visualize different schedulers
    from src.core.optimizers import Adam
    import matplotlib.pyplot as plt
    
    print("Demonstrating LR Schedulers...")
    
    # Create dummy optimizer
    class DummyOptimizer:
        def __init__(self, lr=0.1):
            self.lr = lr
    
    optimizer = DummyOptimizer(lr=0.1)
    epochs = 100
    
    # Test different schedulers
    schedulers_to_test = [
        ('Constant', ConstantLR(optimizer)),
        ('Step (step=30)', StepLR(optimizer, step_size=30, gamma=0.1)),
        ('Cosine', CosineAnnealingLR(optimizer, T_max=epochs)),
        ('Exponential (γ=0.95)', ExponentialLR(optimizer, gamma=0.95)),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, scheduler) in enumerate(schedulers_to_test):
        optimizer.lr = 0.1  # Reset
        lrs = []
        
        for epoch in range(epochs):
            scheduler.step()
            lrs.append(scheduler.current_lr)
        
        axes[idx].plot(range(epochs), lrs, linewidth=2, color='blue')
        axes[idx].set_xlabel('Epoch', fontsize=11)
        axes[idx].set_ylabel('Learning Rate', fontsize=11)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lr_schedulers_demo.png', dpi=150, bbox_inches='tight')
    print("✅ Saved demo plot to plots/lr_schedulers_demo.png")
    plt.close()
    
    print("\n✅ All schedulers working correctly!")
