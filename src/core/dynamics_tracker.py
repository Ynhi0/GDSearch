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
    
    def __init__(self, track_params: bool = False, param_sample_freq: int = 10, param_snapshot_dir: Optional[str] = None):
        """
        Initialize dynamics tracker.
        
        Args:
            track_params: If True, log parameter snapshots to DISK (not RAM) to avoid OOM
            param_sample_freq: Only track params every N iterations (to save memory)
            param_snapshot_dir: Directory to save parameter snapshots (default: None = disabled)
        
        Removed in-memory param_snapshots list (343GB RAM risk).
        Now writes snapshots to disk incrementally using np.save() to prevent OOM.
        """
        self.iterations: List[int] = []
        self.losses: List[float] = []
        self.grad_norms: List[float] = []
        self.update_magnitudes: List[float] = []
        self.param_distances: List[float] = []  # Distance from initialization
        self.learning_rates: List[float] = []
        
        # Add normalized speed metric
        self.normalized_speeds: List[float] = []  # Speed normalized by LR (removes scheduler confounding)

        # Pseudo-convergence detection (saddle-like regions)
        self.pseudo_convergence_flags: List[bool] = []
        self.pseudo_escape_times: List[float] = []
        
        # Disk-based parameter tracking
        self.track_params = track_params
        self.param_sample_freq = param_sample_freq
        self.param_snapshot_dir = Path(param_snapshot_dir) if param_snapshot_dir else None
        if self.param_snapshot_dir:
            self.param_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_counter = 0
        
        # Store initial parameters for distance calculation
        self.initial_params = None
        self.prev_params = None  # For speed calculation
        
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
            
            # Compute normalized speed (removes LR scheduler confounding)
            if self.prev_params is not None and lr > 0:
                step_distance = torch.norm(current_params - self.prev_params).item()
                normalized_speed = step_distance / lr  # Distance per unit LR
                self.normalized_speeds.append(normalized_speed)
            else:
                self.normalized_speeds.append(0.0)
            
            self.prev_params = current_params.clone()
        else:
            self.param_distances.append(0.0)
            self.normalized_speeds.append(0.0)
        
        # Write parameter snapshots to DISK (not RAM)
        if self.track_params and self.param_snapshot_dir and iteration % self.param_sample_freq == 0:
            param_snapshot = torch.nn.utils.parameters_to_vector(
                [p for p in model.parameters() if p.requires_grad]
            ).detach().cpu().numpy()
            
            snapshot_path = self.param_snapshot_dir / f"snapshot_iter_{iteration:06d}.npy"
            np.save(snapshot_path, param_snapshot)
            self.snapshot_counter += 1
    
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
        
        # Load snapshots from disk only when needed (avoid OOM)
        if self.param_snapshot_dir and self.param_snapshot_dir.exists():
            snapshot_files = sorted(self.param_snapshot_dir.glob("snapshot_iter_*.npy"))
            if len(snapshot_files) > 1:
                # Load snapshots in batches to avoid OOM
                trajectory_samples = []
                for f in snapshot_files[:min(100, len(snapshot_files))]:  # Limit to 100 snapshots
                    trajectory_samples.append(np.load(f))
                if len(trajectory_samples) > 1:
                    trajectory = np.array(trajectory_samples)
                    self.instantaneous_speeds = compute_instantaneous_speed(trajectory)
                if len(trajectory_samples) > 2:
                    self.smoothness_index = compute_smoothness_index(trajectory)
        
        # Loss oscillations
        if len(self.losses) > 1:
            losses_arr = np.array(self.losses)
            self.loss_oscillations = compute_oscillation_magnitude(losses_arr, ema_alpha=0.1)

        # Detect pseudo-convergence (near-zero gradients at high loss) and estimate escape time
        self._compute_pseudo_convergence()

    def _compute_pseudo_convergence(
        self,
        grad_tol: float = 1e-4,
        loss_margin: float = 0.1,
        escape_drop: float = 0.05,
        max_window: int = 500
    ):
        """
        Identify iterations where gradients are near zero yet loss remains high (saddle-like pseudo-convergence)
        and estimate iterations needed to escape via a meaningful loss drop.
        """
        if not self.losses or not self.grad_norms:
            self.pseudo_convergence_flags = []
            self.pseudo_escape_times = []
            return

        losses = np.asarray(self.losses, dtype=float)
        grads = np.asarray(self.grad_norms, dtype=float)

        flags: List[bool] = [False] * len(losses)
        escape_steps: List[float] = [np.nan] * len(losses)

        best_loss = float('inf')
        for i, (loss, grad) in enumerate(zip(losses, grads)):
            best_loss = min(best_loss, loss)

            is_flat = grad <= grad_tol
            significantly_above_best = loss > best_loss * (1.0 + loss_margin)

            if is_flat and significantly_above_best:
                flags[i] = True
                target_loss = min(loss * (1.0 - escape_drop), best_loss * (1.0 + loss_margin / 2.0))

                # Look ahead up to max_window steps to see when loss meaningfully decreases
                upper = min(len(losses), i + max_window)
                for j in range(i + 1, upper):
                    if losses[j] <= target_loss:
                        escape_steps[i] = float(j - i)
                        break

        self.pseudo_convergence_flags = flags
        self.pseudo_escape_times = escape_steps
    
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
        
        # Add normalized speed metric (removes LR scheduler confounding)
        if len(self.normalized_speeds) == len(self.iterations):
            df['normalized_speed'] = self.normalized_speeds
        
        # Add oscillations if computed
        if self.loss_oscillations is not None:
            df['loss_oscillation'] = self.loss_oscillations

        # Add pseudo-convergence markers and escape times if available
        if self.pseudo_convergence_flags and len(self.pseudo_convergence_flags) == len(self.iterations):
            df['pseudo_convergence'] = self.pseudo_convergence_flags
        if self.pseudo_escape_times and len(self.pseudo_escape_times) == len(self.iterations):
            df['time_to_escape'] = self.pseudo_escape_times
        
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
        assert self.loss_oscillations is not None, "loss_oscillations unexpectedly None"
        loss_arr = np.array(self.loss_oscillations)
        mean_osc = float(np.mean(loss_arr)) if loss_arr.size else 0.0
        std_osc = float(np.std(loss_arr)) if loss_arr.size else 0.0
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
        
        # Safely compute statistics, guarding against empty or None lists
        grad_arr = np.array(self.grad_norms) if self.grad_norms is not None and len(self.grad_norms) > 0 else np.array([])
        update_arr = np.array(self.update_magnitudes) if self.update_magnitudes is not None and len(self.update_magnitudes) > 0 else np.array([])

        stats = {
            'final_loss': self.losses[-1],
            'mean_grad_norm': float(np.mean(grad_arr)) if grad_arr.size else 0.0,
            'std_grad_norm': float(np.std(grad_arr)) if grad_arr.size else 0.0,
            'mean_update_mag': float(np.mean(update_arr)) if update_arr.size else 0.0,
            'std_update_mag': float(np.std(update_arr)) if update_arr.size else 0.0,
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
    
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(dynamics_dict)))
    
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
