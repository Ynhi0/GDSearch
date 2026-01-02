"""
Effective Learning Rate Analysis for Adaptive Optimizers

This module tracks the per-parameter effective learning rates in adaptive optimizers
(Adam, RMSProp, AdamW, etc.) to understand how adaptive scaling affects different
parameter groups.

Addresses QA Issue #6: "Missing Adaptive Analysis"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class EffectiveLRTracker:
    """
    Track per-parameter effective learning rates for adaptive optimizers.
    
    For Adam: effective_lr = α / (√v_t + ε)
    For RMSProp: effective_lr = α / (√v_t + ε)
    For AdamW: Same as Adam but with decoupled weight decay
    
    This reveals:
    - Which parameter groups receive large vs small updates
    - How adaptation evolves over training
    - Parameter-wise learning dynamics
    """
    
    def __init__(self, model: nn.Module, base_lr: float, optimizer_name: str = 'adam'):
        """
        Args:
            model: PyTorch model to track
            base_lr: Base learning rate (α in Adam)
            optimizer_name: 'adam', 'rmsprop', 'adamw', etc.
        """
        self.model = model
        self.base_lr = base_lr
        self.optimizer_name = optimizer_name.lower()
        
        # History storage
        self.history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.step_count = 0
        
        # Parameter metadata
        self.param_names = []
        self.param_shapes = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
    
    def compute_effective_lr_adam(
        self,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-8
    ) -> Dict[str, np.ndarray]:
        """
        Compute effective LR for Adam optimizer WITH FULL BIAS CORRECTION.
        
        CRITICAL FIX: Adam's true update includes TWO bias correction terms:
        
        m_hat = m_t / (1 - β₁ᵗ)
        v_hat = v_t / (1 - β₂ᵗ)
        θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
        
        The effective learning rate per parameter is:
        
        effective_lr = α * √(1 - β₂ᵗ) / (1 - β₁ᵗ) / (√v_t + ε)
        
        WITHOUT both corrections, early training (t < 100) effective LR estimates
        are off by 10x-1000x, causing false conclusions about optimizer dynamics.
        
        Returns:
            Dict mapping parameter name to effective LR array (same shape as parameter)
        """
        effective_lrs = {}
        
        for group in optimizer.param_groups:
            # Get Adam hyperparameters
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                state = optimizer.state[param]
                
                # Get second moment (v_t in Adam)
                if 'exp_avg_sq' in state:
                    v = state['exp_avg_sq']
                    
                    # Get step count for bias correction
                    step = state.get('step', 0)
                    
                    if step == 0:
                        # No updates yet, use base LR
                        effective_lr = torch.full_like(v, self.base_lr)
                    else:
                        # COMPLETE bias correction formula
                        bias_correction1 = 1 - beta1 ** step  # For first moment
                        bias_correction2 = 1 - beta2 ** step  # For second moment
                        
                        # Effective LR with full bias correction
                        # Note: √(bias_correction2) in numerator, bias_correction1 in denominator
                        effective_lr = (
                            self.base_lr
                            * torch.sqrt(torch.tensor(bias_correction2, dtype=v.dtype, device=v.device))
                            / bias_correction1
                            / (torch.sqrt(v) + eps)
                        )
                    
                    # Find parameter name
                    param_name = self._find_param_name(param)
                    if param_name:
                        effective_lrs[param_name] = effective_lr.detach().cpu().numpy()
        
        return effective_lrs
    
    def compute_effective_lr_rmsprop(
        self,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-8
    ) -> Dict[str, np.ndarray]:
        """
        Compute effective LR for RMSProp optimizer.
        
        effective_lr_i = α / (√v_i + ε)
        where v_i is the moving average of squared gradients
        """
        effective_lrs = {}
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                state = optimizer.state[param]
                
                # Get moving average of squared gradients
                if 'square_avg' in state:
                    v = state['square_avg']
                    effective_lr = self.base_lr / (torch.sqrt(v) + eps)
                    
                    param_name = self._find_param_name(param)
                    if param_name:
                        effective_lrs[param_name] = effective_lr.detach().cpu().numpy()
        
        return effective_lrs
    
    def track_step(
        self,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        eps: float = 1e-8
    ):
        """
        Track effective learning rates at current step.
        
        Args:
            optimizer: PyTorch optimizer instance
            iteration: Current training iteration
            eps: Epsilon for numerical stability
        """
        self.step_count += 1
        
        # Compute effective LRs based on optimizer type
        if 'adam' in self.optimizer_name:
            effective_lrs = self.compute_effective_lr_adam(optimizer, eps)
        elif 'rmsprop' in self.optimizer_name:
            effective_lrs = self.compute_effective_lr_rmsprop(optimizer, eps)
        else:
            # For non-adaptive optimizers, effective LR = base LR
            effective_lrs = {name: np.full(self.param_shapes[name], self.base_lr)
                           for name in self.param_names}
        
        # Store statistics for each parameter
        for param_name, lr_array in effective_lrs.items():
            stats = {
                'iteration': iteration,
                'mean_lr': float(np.mean(lr_array)),
                'std_lr': float(np.std(lr_array)),
                'min_lr': float(np.min(lr_array)),
                'max_lr': float(np.max(lr_array)),
                'median_lr': float(np.median(lr_array)),
                'lr_range': float(np.max(lr_array) - np.min(lr_array))
            }
            self.history[param_name].append(stats)
    
    def _find_param_name(self, param: torch.Tensor) -> Optional[str]:
        """Find parameter name by matching tensor object."""
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all tracked steps.
        
        Returns:
            Dict mapping parameter name to summary stats
        """
        summary = {}
        
        for param_name, stats_list in self.history.items():
            if not stats_list:
                continue
            
            # Aggregate across time
            mean_lrs = [s['mean_lr'] for s in stats_list]
            std_lrs = [s['std_lr'] for s in stats_list]
            
            summary[param_name] = {
                'time_avg_mean_lr': float(np.mean(mean_lrs)),
                'time_avg_std_lr': float(np.mean(std_lrs)),
                'final_mean_lr': mean_lrs[-1],
                'initial_mean_lr': mean_lrs[0],
                'lr_decay_ratio': mean_lrs[-1] / max(mean_lrs[0], 1e-12)
            }
        
        return summary
    
    def visualize_effective_lr_evolution(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Create visualization of effective LR evolution across training.
        
        Args:
            output_path: Path to save figure (if None, displays plot)
            figsize: Figure size
        """
        if not self.history:
            print("No data to visualize. Call track_step() during training.")
            return
        
        n_params = len(self.history)
        fig, axes = plt.subplots(
            (n_params + 1) // 2, 2,
            figsize=figsize,
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, (param_name, stats_list) in enumerate(self.history.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            iterations = [s['iteration'] for s in stats_list]
            mean_lrs = [s['mean_lr'] for s in stats_list]
            std_lrs = [s['std_lr'] for s in stats_list]
            
            # Plot mean ± std
            ax.plot(iterations, mean_lrs, 'b-', label='Mean effective LR', linewidth=2)
            ax.fill_between(
                iterations,
                np.array(mean_lrs) - np.array(std_lrs),
                np.array(mean_lrs) + np.array(std_lrs),
                alpha=0.3,
                color='blue',
                label='±1 std'
            )
            
            # Add base LR reference line
            ax.axhline(self.base_lr, color='red', linestyle='--', 
                      alpha=0.5, label=f'Base LR = {self.base_lr:.1e}')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Effective Learning Rate')
            ax.set_title(f'{param_name}')
            ax.legend(fontsize='small')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(self.history), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved effective LR visualization to {output_path}")
        else:
            plt.show()
    
    def visualize_lr_distribution_heatmap(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Create heatmap showing effective LR distribution across parameters and time.
        
        This reveals which parameter groups consistently receive large vs small updates.
        """
        if not self.history:
            print("No data to visualize.")
            return
        
        # Build matrix: rows = parameters, columns = time steps
        param_names_sorted = sorted(self.history.keys())
        n_params = len(param_names_sorted)
        n_steps = len(self.history[param_names_sorted[0]])
        
        lr_matrix = np.zeros((n_params, n_steps))
        
        for i, param_name in enumerate(param_names_sorted):
            stats_list = self.history[param_name]
            for j, stats in enumerate(stats_list):
                lr_matrix[i, j] = stats['mean_lr']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        from matplotlib.colors import LogNorm
        im = ax.imshow(
            lr_matrix,
            aspect='auto',
            cmap='viridis',
            norm=LogNorm(vmin=lr_matrix.min(), vmax=lr_matrix.max())
        )
        
        ax.set_xlabel('Training Iteration (sampled)', fontsize=12)
        ax.set_ylabel('Parameter Group', fontsize=12)
        ax.set_title(f'Effective Learning Rate Heatmap ({self.optimizer_name.upper()})', 
                    fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(range(n_params))
        ax.set_yticklabels([name.split('.')[-1] for name in param_names_sorted], 
                          fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Effective LR', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved LR heatmap to {output_path}")
        else:
            plt.show()


def compare_adaptive_vs_static(
    model: nn.Module,
    adam_optimizer: torch.optim.Optimizer,
    sgd_lr: float,
    iteration: int,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare effective LRs between Adam (adaptive) and SGD (static).
    
    This reveals WHY Adam outperforms SGD: it automatically scales LR per parameter.
    
    Args:
        model: PyTorch model
        adam_optimizer: Adam optimizer instance (must have state)
        sgd_lr: SGD learning rate for comparison
        iteration: Current iteration
        output_path: Path to save comparison plot
    
    Returns:
        Dict with comparison statistics
    """
    tracker = EffectiveLRTracker(model, adam_optimizer.param_groups[0]['lr'], 'adam')
    tracker.track_step(adam_optimizer, iteration)
    
    # Compute statistics
    summary = tracker.get_summary_statistics()
    
    # Compare to uniform SGD LR
    comparison = {}
    for param_name, stats in summary.items():
        comparison[param_name] = {
            'adam_mean_lr': stats['final_mean_lr'],
            'sgd_lr': sgd_lr,
            'ratio': stats['final_mean_lr'] / sgd_lr,
            'faster_than_sgd': stats['final_mean_lr'] > sgd_lr
        }
    
    # Visualization
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        param_names = list(comparison.keys())
        adam_lrs = [comparison[p]['adam_mean_lr'] for p in param_names]
        
        x = np.arange(len(param_names))
        ax.bar(x, adam_lrs, alpha=0.7, label='Adam (Adaptive)')
        ax.axhline(sgd_lr, color='red', linestyle='--', linewidth=2, 
                  label=f'SGD (Static) = {sgd_lr:.1e}')
        
        ax.set_xlabel('Parameter Group')
        ax.set_ylabel('Effective Learning Rate')
        ax.set_title('Adaptive vs Static Learning Rates')
        ax.set_xticks(x)
        ax.set_xticklabels([p.split('.')[-1] for p in param_names], rotation=45, ha='right')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
    
    return comparison
