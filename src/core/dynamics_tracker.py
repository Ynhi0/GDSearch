"""
Training Dynamics Tracker
Real-time tracking of optimization dynamics during neural network training.

This module provides a lightweight tracker that monitors per-iteration dynamics
including gradient norms, update magnitudes, parameter distances, and loss oscillations.

Required by research proposal for "phân tích động học chi tiết" (detailed dynamics analysis).
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class TrainingDynamicsTracker:
    """
    Track per-iteration dynamics during neural network training.
    
    Captures metrics required for dynamics analysis:
    - Loss values
    - Gradient norms
    - Parameter update magnitudes
    - Distance from initialization
    - Learning rate (if adaptive)
    
    Example:
        >>> tracker = TrainingDynamicsTracker()
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         loss = model(batch)
        >>>         loss.backward()
        >>>         tracker.track_step(iteration, loss.item(), model, optimizer)
        >>>         optimizer.step()
        >>> tracker.save_dynamics('dynamics.csv')
        >>> tracker.plot_dynamics('plots/')
    """
    
    def __init__(self, track_params: bool = False, param_sample_freq: int = 10):
        """
        Initialize dynamics tracker.
        
        Args:
            track_params: If True, store full parameter snapshots (memory intensive)
            param_sample_freq: Only track params every N iterations (to save memory)
        """
        self.iterations = []
        self.losses = []
        self.grad_norms = []
        self.update_magnitudes = []
        self.param_distances = []  # Distance from initialization
        self.learning_rates = []
        
        # Optional: track parameter snapshots (memory intensive)
        self.track_params = track_params
        self.param_sample_freq = param_sample_freq
        self.param_snapshots = [] if track_params else None
        
        # Store initial parameters for distance calculation
        self.initial_params = None
        
        # Computed metrics (filled during analysis)
        self.instantaneous_speeds = None
        self.loss_oscillations = None
        self.smoothness_index = None
        
    def set_initial_params(self, model: torch.nn.Module):
        """
        Store initial parameter values for distance tracking.
        
        Args:
            model: PyTorch model
        """
        self.initial_params = torch.nn.utils.parameters_to_vector(
            [p for p in model.parameters() if p.requires_grad]
        ).detach().cpu().clone()
        
    def track_step(self, iteration: int, loss: float, 
                   model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer):
        """
        Track metrics for a single optimization step (AFTER backward, BEFORE optimizer.step).
        
        Args:
            iteration: Current iteration number
            loss: Loss value for this iteration
            model: PyTorch model (with computed gradients)
            optimizer: PyTorch optimizer
        """
        # Store iteration and loss
        self.iterations.append(iteration)
        self.losses.append(loss)
        
        # Compute total gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        self.grad_norms.append(grad_norm)
        
        # Get current learning rate
        lr = optimizer.param_groups[0]['lr']
        self.learning_rates.append(lr)
        
        # Estimate update magnitude (before optimizer step)
        # For SGD: update = lr * grad
        # For Adam: update ≈ lr * grad (approximation)
        update_mag = lr * grad_norm
        self.update_magnitudes.append(update_mag)
        
        # Compute distance from initialization
        if self.initial_params is not None:
            current_params = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).detach().cpu()
            
            distance = torch.norm(current_params - self.initial_params).item()
            self.param_distances.append(distance)
        else:
            self.param_distances.append(0.0)
        
        # Optionally store parameter snapshot
        if self.track_params and iteration % self.param_sample_freq == 0:
            param_snapshot = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).detach().cpu().numpy()
            self.param_snapshots.append(param_snapshot)
    
    def compute_derived_metrics(self):
        """Compute derived dynamics metrics from tracked data."""
        try:
            from src.analysis.dynamics_metrics import (
                compute_instantaneous_speed,
                compute_oscillation_magnitude,
                compute_smoothness_index
            )
        except ImportError:
            # Fallback if module structure different
            from ..analysis.dynamics_metrics import (
                compute_instantaneous_speed,
                compute_oscillation_magnitude,
                compute_smoothness_index
            )
        
        # Instantaneous speeds (if param snapshots available)
        if self.param_snapshots is not None and len(self.param_snapshots) > 1:
            trajectory = np.array(self.param_snapshots)
            self.instantaneous_speeds = compute_instantaneous_speed(trajectory)
        
        # Loss oscillations
        if len(self.losses) > 1:
            losses_arr = np.array(self.losses)
            self.loss_oscillations = compute_oscillation_magnitude(losses_arr, ema_alpha=0.1)
        
        # Trajectory smoothness (if param snapshots available)
        if self.param_snapshots is not None and len(self.param_snapshots) > 2:
            trajectory = np.array(self.param_snapshots)
            self.smoothness_index = compute_smoothness_index(trajectory)
    
    def save_dynamics(self, output_path: str):
        """
        Save tracked dynamics to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        df = pd.DataFrame({
            'iteration': self.iterations,
            'loss': self.losses,
            'grad_norm': self.grad_norms,
            'update_magnitude': self.update_magnitudes,
            'param_distance': self.param_distances,
            'learning_rate': self.learning_rates
        })
        
        # Add oscillations if computed
        if self.loss_oscillations is not None:
            df['loss_oscillation'] = self.loss_oscillations
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dynamics saved to {output_path}")
        
        return df
    
    def plot_dynamics(self, output_dir: str, optimizer_name: str = "Optimizer"):
        """
        Create comprehensive dynamics visualization plots.
        
        Args:
            output_dir: Directory to save plots
            optimizer_name: Name for plot titles
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Compute derived metrics if not done
        if self.loss_oscillations is None:
            self.compute_derived_metrics()
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{optimizer_name} - Training Dynamics Analysis', fontsize=14, fontweight='bold')
        
        iterations = np.array(self.iterations)
        
        # Panel 1: Loss trajectory with oscillations
        ax = axes[0, 0]
        ax.plot(iterations, self.losses, 'b-', alpha=0.6, label='Loss')
        if self.loss_oscillations is not None:
            # Plot EMA as smooth line
            ema = np.zeros_like(self.losses)
            ema[0] = self.losses[0]
            for t in range(1, len(self.losses)):
                ema[t] = 0.1 * self.losses[t] + 0.9 * ema[t-1]
            ax.plot(iterations, ema, 'r-', linewidth=2, label='EMA (α=0.1)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Trajectory with Trend')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Panel 2: Gradient norm
        ax = axes[0, 1]
        ax.plot(iterations, self.grad_norms, 'g-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Magnitude over Time')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Panel 3: Update magnitude
        ax = axes[1, 0]
        ax.plot(iterations, self.update_magnitudes, 'm-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Update Magnitude (lr × grad_norm)')
        ax.set_title('Parameter Update Size')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Panel 4: Distance from initialization
        ax = axes[1, 1]
        ax.plot(iterations, self.param_distances, 'c-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L2 Distance from Init')
        ax.set_title('Parameter Space Exploration')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / f'{optimizer_name}_dynamics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dynamics plots saved to {plot_path}")
        
        # Create oscillation-specific plot
        if self.loss_oscillations is not None:
            self._plot_oscillations(output_dir, optimizer_name)
    
    def _plot_oscillations(self, output_dir: str, optimizer_name: str):
        """Create detailed oscillation analysis plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'{optimizer_name} - Oscillation Analysis', fontsize=14, fontweight='bold')
        
        iterations = np.array(self.iterations)
        
        # Loss oscillation magnitude
        ax1.plot(iterations, self.loss_oscillations, 'r-', alpha=0.7)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss Oscillation Magnitude')
        ax1.set_title('Deviation from EMA Trend')
        ax1.grid(alpha=0.3)
        
        # Oscillation histogram
        ax2.hist(self.loss_oscillations, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Oscillation Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Oscillation Distribution')
        ax2.grid(alpha=0.3)
        
        # Add statistics
        mean_osc = np.mean(self.loss_oscillations)
        std_osc = np.std(self.loss_oscillations)
        ax2.axvline(mean_osc, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_osc:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = Path(output_dir) / f'{optimizer_name}_oscillations.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Oscillation plot saved to {plot_path}")
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics of dynamics.
        
        Returns:
            dict: Summary statistics
        """
        if len(self.losses) == 0:
            return {}
        
        stats = {
            'final_loss': self.losses[-1],
            'mean_grad_norm': float(np.mean(self.grad_norms)),
            'std_grad_norm': float(np.std(self.grad_norms)),
            'mean_update_mag': float(np.mean(self.update_magnitudes)),
            'std_update_mag': float(np.std(self.update_magnitudes)),
            'final_param_distance': self.param_distances[-1] if self.param_distances else 0.0,
        }
        
        # Add oscillation stats if available
        if self.loss_oscillations is not None:
            stats['mean_loss_oscillation'] = float(np.mean(self.loss_oscillations))
            stats['std_loss_oscillation'] = float(np.std(self.loss_oscillations))
        
        # Add smoothness if available
        if self.smoothness_index is not None:
            stats['smoothness_index'] = float(self.smoothness_index)
        
        return stats


def compare_multiple_dynamics(dynamics_dict: Dict[str, TrainingDynamicsTracker],
                              output_dir: str):
    """
    Create comparative plots for multiple optimizers.
    
    Args:
        dynamics_dict: Dictionary mapping optimizer names to their trackers
        output_dir: Directory to save comparison plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimizer Dynamics Comparison', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dynamics_dict)))
    
    for (opt_name, tracker), color in zip(dynamics_dict.items(), colors):
        iterations = np.array(tracker.iterations)
        
        # Loss comparison
        axes[0, 0].plot(iterations, tracker.losses, label=opt_name, color=color, alpha=0.7)
        
        # Gradient norm comparison
        axes[0, 1].plot(iterations, tracker.grad_norms, label=opt_name, color=color, alpha=0.7)
        
        # Update magnitude comparison
        axes[1, 0].plot(iterations, tracker.update_magnitudes, label=opt_name, color=color, alpha=0.7)
        
        # Distance comparison
        axes[1, 1].plot(iterations, tracker.param_distances, label=opt_name, color=color, alpha=0.7)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Magnitude')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Update Magnitude')
    axes[1, 0].set_title('Parameter Update Size')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('L2 Distance from Init')
    axes[1, 1].set_title('Parameter Space Exploration')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'dynamics_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to {plot_path}")
    
    # Create summary table
    summary_data = {}
    for opt_name, tracker in dynamics_dict.items():
        summary_data[opt_name] = tracker.get_summary_stats()
    
    summary_df = pd.DataFrame(summary_data).T
    summary_path = Path(output_dir) / 'dynamics_summary.csv'
    summary_df.to_csv(summary_path)
    
    print(f"✓ Summary table saved to {summary_path}")
    
    return summary_df


if __name__ == "__main__":
    print("Training Dynamics Tracker - Demo")
    print("=" * 60)
    print("This module tracks per-iteration dynamics during training.")
    print("See examples in experiments for integration with training loops.")
