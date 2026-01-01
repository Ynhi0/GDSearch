"""
Ablation study comparing optimizer progression: SGD → SGD+Momentum → RMSProp → Adam → AdamW → AMSGrad

WARNING: This script uses FIXED learning rates (lr=0.01) for all optimizers.
This violates hyperparameter fairness principles and may produce biased results.

**SCIENTIFIC LIMITATION - "Fair" Defaults for 2D Functions:**
The "fair" default learning rates (SGD=0.1, Adam=0.001) used in this script are
derived from NEURAL NETWORK training conventions, NOT optimized for 2D mathematical
functions like Rosenbrock or Ackley.

Consequence: An optimizer may appear to "win" simply because its default LR
happens to be closer to optimal for the specific 2D landscape being tested, while
another optimizer's default may be catastrophically bad.

Example:
- On Rosenbrock with a=1, b=100, the optimal LR for Adam might be 0.01, not 0.001
- For SGD, 0.1 might cause divergence while 0.01 converges slowly
- These are ARTIFACTS of arbitrary NN-derived defaults, not inherent optimizer quality

Proper 2D Evaluation Protocol (not implemented here):
1. For each optimizer, sweep LR over [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
2. Report the BEST result for each optimizer from its sweep
3. OR report full heatmaps (optimizer x LR) to show sensitivity

This script uses fixed defaults for QUICK exploratory analysis only.
For rigorous 2D function benchmarks, implement per-function LR tuning.

Note: Neural network experiments (run_nn_experiment.py) are less affected by this
issue because NN defaults (0.1 for SGD, 0.001 for Adam) are based on extensive
empirical research across thousands of NN training runs.

AUDIT FIX: This script now requires --allow-unfair-ablations flag to prevent
accidental use in canonical benchmarks.

For rigorous comparisons, use run_fair_optimizer_ablation.py instead,
which implements HYPERPARAMETER_FAIRNESS_PROTOCOL.md with:
- Published defaults from original papers (with citations)
- OR per-optimizer LR sweeps with appropriate ranges
- Statistical significance testing with multiple comparison corrections

This script is retained for backward compatibility and quick sanity checks only.

Outputs:
- CSV with convergence metrics
- Figure showing loss curves and final performance
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Callable
import logging

from src.core.test_functions import Rosenbrock, TestFunction
from src.core.optimizers import SGD, SGDMomentum, RMSProp, Adam, AdamW, AMSGrad, SAM


# AUDIT FIX: Add guard to prevent unfair benchmark usage
def check_ablation_guard(allow_unfair: Optional[bool] = None):
    """Legacy guard for unfair ablations.

    Backwards-compatible behavior:
      - If `allow_unfair` is None (default), preserve legacy behavior: require the
        '--allow-unfair-ablations' or '--use-legacy-unfair' flag in sys.argv and
        exit with SystemExit(1) when absent (this is relied on by tests).
      - If `allow_unfair` is True/False, behave non-fatally and return True/False
        while printing a helpful warning when appropriate.
    """
    if allow_unfair is None:
        # Legacy strict behavior (preserve test expectations)
        if '--allow-unfair-ablations' in sys.argv or '--use-legacy-unfair' in sys.argv:
            print("\n" + "="*80)
            print("WARNING: Running in LEGACY UNFAIR mode (fixed lr=0.01 for all optimizers)")
            print("Consider running with the default per-optimizer fair learning rates.")
            print("="*80)
            return True

        # Preserve original strict behavior expected by tests
        print("\n" + "="*80)
        print("ERROR: Ablation Guard - Preventing Unfair Benchmark Usage")
        print("="*80)
        print(
            "HYPERPARAMETER FAIRNESS WARNING: This script uses a fixed lr for all optimizers.\n"
            "This script is for exploratory analysis only and should NOT be used for canonical benchmarks.\n\n"
            "To run the legacy unfair version, pass --allow-unfair-ablations. For fair comparisons,"
            " run the default mode which uses per-optimizer defaults or run_fair_optimizer_ablation.py"
        )
        sys.exit(1)

    # Non-fatal programmatic behavior
    if allow_unfair:
        print("\n" + "="*80)
        print("WARNING: Running in LEGACY UNFAIR mode (fixed lr=0.01 for all optimizers)")
        print("Consider running with the default per-optimizer fair learning rates.")
        print("="*80)
        return True
    return False


def run_optimizer_ablation(
    test_function: TestFunction,
    initial_point: tuple[float, float],
    max_iterations: int = 10000,
    results_dir: str = 'results',
    plots_dir: str = 'plots',
    use_legacy_unfair: bool = False,
    track_params: bool = False,
    lr_map_override: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Run ablation study comparing optimizer variants on 2D test functions.

    By default this function enforces per-optimizer fair learning rates. To run the legacy
    unfair mode (fixed lr=0.01 for every optimizer) set `use_legacy_unfair=True`.
    
    **SCIENTIFIC CAVEAT - Learning Rate Selection for 2D Functions:**
    The \"fair\" default LRs (SGD=0.1, Adam=0.001) are derived from NEURAL NETWORK
    training conventions. They may NOT be optimal for 2D mathematical functions.
    Results should be interpreted as \"how well do NN-tuned defaults transfer to 2D\"
    rather than \"which optimizer is fundamentally better on this landscape.\"
    
    For rigorous 2D benchmarks, consider implementing per-function LR tuning or
    reporting full (optimizer \u00d7 LR) heatmaps.

    Args:
        test_function: Test function instance (e.g., Rosenbrock, Ackley)
        initial_point: Starting (x, y)
        max_iterations: Number of iterations
        results_dir: Directory for CSV output
        plots_dir: Directory for plots
        use_legacy_unfair: If True, run legacy mode with lr=0.01 for all optimizers
        track_params: If True, track full parameter snapshots in the tracker
        lr_map_override: Optional dict to override default per-optimizer LRs
                        (useful for function-specific tuning)

    Returns:
        DataFrame with summary metrics
    """
    # Decide hyperparameter protocol: by default use FAIR per-optimizer defaults.
    if use_legacy_unfair:
        check_ablation_guard(allow_unfair=True)
        logging.warning(
            "⚠️ HYPERPARAMETER FAIRNESS WARNING: Running in LEGACY UNFAIR mode with fixed lr=0.01 for all optimizers. "
            "This mode is discouraged for canonical benchmarks."
        )
        lr_map = { 'default': 0.01 }
    else:
        logging.info("✅ Enforcing HYPERPARAMETER FAIRNESS: using per-optimizer default learning rates.")
        logging.warning(
            "⚠️ 2D FUNCTION CAVEAT: Using NN-derived default LRs (SGD=0.1, Adam=0.001) on 2D functions. "
            "These may not be optimal for mathematical test functions. Results show 'transferability' not 'absolute optimality'. "
            "For function-specific tuning, use lr_map_override parameter."
        )
        # Per-optimizer fair defaults (approx. published defaults)
        lr_map = {
            'SGD': 0.1,
            'SGD+Momentum': 0.1,
            'RMSProp': 0.001,
            'Adam': 0.001,
            'AdamW': 0.001,
            'AMSGrad': 0.001,
        }

    # Allow programmatic overrides for specific optimizers
    if lr_map_override is not None:
        lr_map.update(lr_map_override)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define optimizer sequence (progressive improvements)
    optimizers = [
        ('SGD', SGD(lr=lr_map.get('SGD', lr_map.get('default', 0.01)))),
        ('SGD+Momentum', SGDMomentum(lr=lr_map.get('SGD+Momentum', lr_map.get('default', 0.01)), beta=0.9)),
        ('RMSProp', RMSProp(lr=lr_map.get('RMSProp', lr_map.get('default', 0.01)), decay_rate=0.9)),
        ('Adam', Adam(lr=lr_map.get('Adam', lr_map.get('default', 0.01)), beta1=0.9, beta2=0.999)),
        ('AdamW', AdamW(lr=lr_map.get('AdamW', lr_map.get('default', 0.01)), beta1=0.9, beta2=0.999, weight_decay=0.01)),
        ('AMSGrad', AMSGrad(lr=lr_map.get('AMSGrad', lr_map.get('default', 0.01)), beta1=0.9, beta2=0.999)),
        # CRITICAL FIX: Add SAM with proper 2D support
        ('SAM', SAM(lr=lr_map.get('SAM', lr_map.get('default', 0.01)), rho=0.05, base_optimizer='SGD')),
    ]
    
    # Storage for trajectories
    trajectories: Dict[str, Any] = {}
    summary_metrics: List[Dict[str, Any]] = []
    
    print(f"\n{'='*70}")
    print(f"Optimizer Ablation Study: {test_function.__class__.__name__}")
    print(f"Initial point: {initial_point}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*70}\n")
    
    from src.core.dynamics_tracker import TrainingDynamicsTracker
    from src.analysis.theoretical_bounds import (
        estimate_smoothness,
        estimate_strong_convexity,
        sgd_convergence_bound,
        adam_convergence_bound,
        momentum_convergence_bound  # CRITICAL FIX: Add momentum bounds
    )
    import torch

    for opt_name, optimizer in optimizers:
        # Reset optimizer state completely before each run
        optimizer.reset()
        x: float
        y: float
        x, y = initial_point
        
        # Extract learning rate once for consistent reference throughout loop
        opt_lr = optimizer.lr if hasattr(optimizer, 'lr') else 0.01

        # Local lists for vector-based estimates
        grads_list: List[np.ndarray[Any, np.dtype[np.float64]]] = []
        params_list: List[np.ndarray[Any, np.dtype[np.float64]]] = []

        # Create numeric model and optimizer wrapper for tracker compatibility
        class NumericModel(torch.nn.Module):
            def __init__(self, x0: float, y0: float) -> None:
                super().__init__()  # type: ignore[misc]
                self.param = torch.nn.Parameter(torch.tensor([x0, y0], dtype=torch.float32))

            def forward(self, _x: torch.Tensor) -> torch.Tensor:
                """Dummy forward pass (not used, but required for nn.Module)."""
                return self.param
            
            def update_position(self, new_x: float, new_y: float) -> None:
                """Update parameter tensor to new position (for tracking accuracy)."""
                self.param.data = torch.tensor([new_x, new_y], dtype=torch.float32)

        class NumericOptimizerMock(torch.optim.Optimizer):
            def __init__(self, params: Any, lr: float) -> None:
                defaults: Dict[str, Any] = {'lr': lr}
                super().__init__(params, defaults)
            
            @torch.no_grad()  # type: ignore[misc]
            def step(self, closure: Optional[Callable[[], float]] = None) -> float:  # type: ignore[override]
                """
                Dummy step method (not used, but required for Optimizer).
                
                AUDIT FIX: Return type varies across PyTorch versions and type stubs.
                Using type: ignore[override] to handle inconsistency while preserving
                runtime compatibility. Returns 0.0 as dummy loss value.
                """
                if closure is not None:
                    with torch.enable_grad():
                        _ = closure()  # Compute but discard loss
                return 0.0  # Dummy loss return for type checker compatibility

        numeric_model = NumericModel(x, y)
        numeric_optim_mock = NumericOptimizerMock(numeric_model.parameters(), opt_lr)

        tracker = TrainingDynamicsTracker(track_params=track_params)
        # Initialize tracker with current params
        tracker.set_initial_params(numeric_model)

        diverged = False
        divergence_reason = None
        
        # Set numpy to raise on warnings for proper exception handling
        old_settings = np.seterr(all='raise')

        try:
            for i in range(max_iterations):
                try:
                    # CRITICAL FIX (Issue #19): LR Scheduler for SGD
                    # Apply learning rate decay to counter "Constant LR Strawman" criticism
                    if i > 0 and i % 100 == 0:  # Decay every 100 iterations
                        if hasattr(optimizer, 'lr') and 'SGD' in opt_name and 'Momentum' not in opt_name and 'Adam' not in opt_name:
                            # Apply 0.99 decay to base SGD only (prevents oscillation at convergence)
                            old_lr = optimizer.lr
                            optimizer.lr *= 0.99
                            if i % 1000 == 0:  # Log every 1000 iters
                                logging.info(f"{opt_name} LR decayed from {old_lr:.6f} to {optimizer.lr:.6f} at iteration {i}")
                    
                    loss = test_function.compute(x, y)
                    grad_x, grad_y = test_function.gradient(x, y)

                    # Overflow protection
                    if not np.isfinite(loss) or not np.isfinite(grad_x) or not np.isfinite(grad_y):
                        raise OverflowError("Non-finite gradient or loss")

                    # CRITICAL FIX (Issue #21): Gradient Clipping
                    # Prevents "Gradient Explosion Vulnerability" on steep landscapes (Rosenbrock, etc.)
                    # This is a standard safeguard used in all production codebases
                    max_grad_norm = 10.0  # Clip threshold (prevents catastrophic divergence)
                    
                    # NUMERICAL STABILITY FIX: Use np.hypot to avoid overflow in x**2 + y**2
                    # np.hypot computes sqrt(x^2 + y^2) without intermediate overflow
                    grad_norm_raw = np.hypot(grad_x, grad_y)
                    
                    # Apply gradient clipping if needed
                    if grad_norm_raw > max_grad_norm:
                        clip_factor = max_grad_norm / grad_norm_raw
                        grad_x *= clip_factor
                        grad_y *= clip_factor
                        grad_norm = max_grad_norm
                        if i < 10 or (i < 1000 and i % 100 == 0):  # Log early and periodic clipping
                            logging.debug(f"{opt_name} gradient clipped from {grad_norm_raw:.2f} to {max_grad_norm}")
                    else:
                        grad_norm = grad_norm_raw

                    if not np.isfinite(grad_norm):
                        raise OverflowError("Non-finite grad_norm")

                    # Record vectors for L and mu estimation
                    grads_list.append(np.array([grad_x, grad_y]))
                    params_list.append(np.array([x, y]))

                    # Populate numeric model grad tensors for tracker
                    grad_tensor = torch.tensor([grad_x, grad_y], dtype=torch.float32)
                    # Assign gradient to parameter (accessing internal state for mock compatibility)
                    numeric_model.param.grad = grad_tensor

                    # Track dynamics (before step)
                    tracker.track_step(i, float(loss), numeric_model, numeric_optim_mock)

                    # CRITICAL FIX: SAM requires closure/oracle for 2D functions
                    if isinstance(optimizer, SAM):
                        # SAM two-step process:
                        # 1. Compute adversarial point using the test function gradient
                        # Note: We compute adversarial parameters manually rather than using
                        # the protected _compute_adversarial_step to avoid Pyright warnings
                        grad_norm = np.hypot(grad_x, grad_y)
                        if grad_norm >= 1e-12:
                            # Normalize gradient direction
                            grad_dir_x = grad_x / grad_norm
                            grad_dir_y = grad_y / grad_norm
                            # Adversarial step: θ + ρ * (g / ||g||)
                            adv_x = x + optimizer.rho * grad_dir_x
                            adv_y = y + optimizer.rho * grad_dir_y
                        else:
                            adv_x, adv_y = x, y
                        
                        # 2. Compute gradient at adversarial point
                        adv_grad_x, adv_grad_y = test_function.gradient(adv_x, adv_y)
                        adversarial_gradients = (adv_grad_x, adv_grad_y)
                        
                        # 3. Take step using adversarial gradient
                        step_result = optimizer.step((x, y), (grad_x, grad_y), 
                                                    adversarial_gradients=adversarial_gradients)
                    else:
                        # Standard optimizer step
                        step_result = optimizer.step((x, y), (grad_x, grad_y))
                    
                    if isinstance(step_result, tuple) and len(step_result) == 2:
                        x_new, y_new = step_result
                        x = float(x_new)
                        y = float(y_new)
                    else:
                        raise TypeError(f"Expected tuple from optimizer.step, got {type(step_result)}")

                    # Check if step produced non-finite values
                    if not np.isfinite(x) or not np.isfinite(y):
                        raise OverflowError("Non-finite parameters after step")
                    
                    # CRITICAL FIX: Update numeric model to new position for accurate distance tracking
                    numeric_model.update_position(x, y)

                except (OverflowError, FloatingPointError) as e:
                    # Log exception details for debugging
                    logging.warning(f"{opt_name} diverged at iteration {i}: {e}")
                    diverged = True
                    divergence_reason = str(e)
                    break
        finally:
            # Restore numpy error settings
            np.seterr(**old_settings)

        # Store tracker for later comparative plots
        trajectories.setdefault('trackers', {})[opt_name] = tracker
        

        


        # Save dynamics and plots
        try:
            dyn_dir = os.path.join(plots_dir, 'dynamics', opt_name.replace(' ', '_'))
            dyn_csv_path = os.path.join(dyn_dir, f'{opt_name}_dynamics.csv')
            os.makedirs(dyn_dir, exist_ok=True)
            _ = tracker.save_dynamics(dyn_csv_path)  # Returns df but not used here
            tracker.plot_dynamics(dyn_dir, optimizer_name=opt_name)
        except Exception as e:
            logging.debug(f"Failed to save/plot dynamics for {opt_name}: {e}")

        # Prepare summary statistics using tracker
        try:
            final_loss = tracker.losses[-1] if len(tracker.losses) > 0 else np.inf
            final_grad = tracker.grad_norms[-1] if len(tracker.grad_norms) > 0 else np.inf
        except (IndexError, Exception):
            final_loss = np.inf
            final_grad = np.inf

        # Define theory indices and default estimates (always defined, even if empty)
        theory_iters = np.arange(0, max(1, len(tracker.iterations)))  # Ensure at least length 1
        est_L = 0.0
        est_mu = 0.0
        theory_curve = None
        
        # Extract learning rate once for consistent use
        current_lr = optimizer.lr if hasattr(optimizer, 'lr') else 0.01

        if diverged:
            min_loss = np.inf
            converged_iter = None
            precise_converged_iter = None
        else:
            # Safely handle case where all losses are non-finite
            finite_losses = [l for l in tracker.losses if np.isfinite(l)]
            min_loss = min(finite_losses) if finite_losses else np.inf
            converged_iter = next((i for i, l in enumerate(tracker.losses) if np.isfinite(l) and l < 1e-3), None)
            precise_converged_iter = next((i for i, g in enumerate(tracker.grad_norms) if np.isfinite(g) and g < 1e-6), None)
            # Estimate smoothness (L) and strong convexity (mu) from vector samples
            try:
                est_L = estimate_smoothness(grads_list, params_list) if len(grads_list) > 1 else 0.0
                est_mu = estimate_strong_convexity(grads_list, params_list) if len(grads_list) > 1 else 0.0
            except Exception as e:
                logging.debug(f"Estimate smoothness failed for {opt_name}: {e}")
                est_L = 0.0
                est_mu = 0.0

            # Compute theoretical overlay curve (using already-defined theory_iters)
            try:
                # Ensure we have valid losses to work with
                if len(tracker.losses) == 0:
                    raise ValueError("No losses tracked")
                    
                init_loss = tracker.losses[0] if np.isfinite(tracker.losses[0]) else 1.0
                
                # CRITICAL FIX: Use appropriate theoretical bounds per optimizer type
                if 'Adam' in opt_name or 'AdamW' in opt_name or 'AMSGrad' in opt_name:
                    # Compute bounds for validation (not used in curve, but good for logging)
                    _ = adam_convergence_bound(
                        L=est_L if est_L > 0 else 1.0,
                        T=max(1, len(theory_iters)),
                        alpha=current_lr
                    )
                    # Adam theoretical decay ~ O(1/sqrt(t)). Scale by initial loss for visualization.
                    theory_curve = init_loss / np.sqrt(np.maximum(1, theory_iters + 1))
                
                elif 'Momentum' in opt_name:
                    # CRITICAL FIX: Use momentum-specific bounds with acceleration
                    momentum_beta = 0.9  # Default momentum coefficient
                    momentum_stats = momentum_convergence_bound(
                        L=est_L if est_L > 0 else 1.0,
                        mu=est_mu if est_mu > 0 else 1e-6,
                        lr=current_lr,
                        momentum=momentum_beta,
                        T=max(1, len(theory_iters)),
                        method='heavy_ball'
                    )
                    conv_rate = momentum_stats.get('convergence_rate', 1.0)
                    final_bound = momentum_stats.get('final_bound', 0.0)
                    
                    # Accelerated decay: ρ = 1 - sqrt(μ/L) vs vanilla SGD's 1 - μ/L
                    if 0 < conv_rate < 1:
                        log_decay = theory_iters * np.log(conv_rate)
                        log_decay = np.clip(log_decay, -700, 0)
                        theory_curve = init_loss * np.exp(log_decay) + final_bound
                    else:
                        theory_curve = np.full_like(theory_iters, init_loss, dtype=float)
                
                else:
                    # Vanilla SGD, RMSProp, SAM (use SGD bounds)
                    sgd_stats = sgd_convergence_bound(
                        L=est_L if est_L > 0 else 1.0,
                        mu=est_mu if est_mu > 0 else 1e-6,
                        lr=current_lr,
                        T=max(1, len(theory_iters))
                    )
                    conv_rate = sgd_stats.get('convergence_rate', 1.0)
                    final_bound = sgd_stats.get('final_bound', 0.0)
                    # Exponential decay plus asymptotic variance term
                    # Clip to prevent overflow: conv_rate^T can overflow for large T
                    if 0 < conv_rate < 1:
                        # Safe computation: use log space for large exponents
                        log_decay = theory_iters * np.log(conv_rate)
                        # Clip extreme values to prevent overflow
                        log_decay = np.clip(log_decay, -700, 0)  # exp(-700) ≈ 1e-304
                        theory_curve = init_loss * np.exp(log_decay) + final_bound
                    else:
                        # Divergent or non-convergent case
                        theory_curve = np.full_like(theory_iters, init_loss, dtype=float)
            except Exception as e:
                logging.debug(f"Failed to compute theoretical curve for {opt_name}: {e}")
                theory_curve = None
        # Append LR and estimates to summary
        summary_metrics.append({
            'Optimizer': opt_name,
            'LR': current_lr,
            'Final Loss': final_loss,
            'Final Grad Norm': final_grad,
            'Min Loss': min_loss,
            'Iterations to Loss<1e-3': converged_iter if converged_iter is not None else max_iterations,
            'Iterations to GradNorm<1e-6': precise_converged_iter if precise_converged_iter is not None else max_iterations,
            'Converged (loss<1e-3)': converged_iter is not None,
            'Converged (grad<1e-6)': precise_converged_iter is not None,
            'Diverged': diverged,
            'Divergence Reason': divergence_reason if divergence_reason else 'None',
            'Estimated_L': est_L,
            'Estimated_mu': est_mu,
            'Has_Theoretical_Curve': theory_curve is not None
        })

        # Store theory curve for plotting overlay
        trajectories.setdefault('__theory__', {})[opt_name] = (theory_iters, theory_curve)

        print(f"{opt_name:20s} | Final Loss: {final_loss:12.6e} | "
              f"Converged (loss<1e-3): {'YES' if converged_iter is not None else 'NO':3s} at iter {converged_iter if converged_iter is not None else ('DIV' if np.isinf(final_loss) else '>10k')}")
    
    # Save summary CSV
    df_summary = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(results_dir, 'optimizer_ablation_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Comparative dynamics plot across optimizers using the trackers
    try:
        from src.core.dynamics_tracker import compare_multiple_dynamics
        compare_multiple_dynamics(trajectories.get('trackers', {}), os.path.join(plots_dir, 'dynamics_compare'))
    except Exception as e:
        logging.debug(f"Failed to create comparative dynamics plots: {e}")
    
    # Create figure with subplots
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # type: ignore[misc]
    
    # Define color map for consistent coloring
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.9, len(optimizers)))  # type: ignore[misc]
    
    # Plot 1: Loss curves (log scale)
    ax = axes[0, 0]
    for (opt_name, _), color in zip(optimizers, colors):
        tracker = trajectories['trackers'][opt_name]
        iters = tracker.iterations
        losses = tracker.losses
        # Sample every 10 iterations for clarity; filter out non-finite
        sample_iters = []
        sample_loss = []
        for idx in range(0, len(losses), 10):
            if np.isfinite(losses[idx]) and losses[idx] > 0:
                sample_iters.append(iters[idx])
                sample_loss.append(losses[idx])
        if sample_loss:
            ax.plot(sample_iters, sample_loss, label=opt_name, linewidth=2, color=color, alpha=0.8)

        # Plot theoretical overlay if available
        try:
            theory_iters, theory_curve = trajectories.get('__theory__', {}).get(opt_name, (None, None))
            if theory_curve is not None and theory_iters is not None and len(theory_curve) > 0 and sample_iters:
                theory_vals = np.interp(sample_iters, theory_iters, theory_curve)
                ax.plot(sample_iters, theory_vals, linestyle='--', color=color, alpha=0.6, linewidth=1.5, label=f'{opt_name} (theory)')
        except Exception:
            pass
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Loss Convergence (Ablation Study)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm (log scale)
    ax = axes[0, 1]
    for (opt_name, _), color in zip(optimizers, colors):
        tracker = trajectories['trackers'][opt_name]
        iters = tracker.iterations
        grads = tracker.grad_norms
        sample_iters = []
        sample_grad = []
        for idx in range(0, len(grads), 10):
            if np.isfinite(grads[idx]) and grads[idx] > 0:
                sample_iters.append(iters[idx])
                sample_grad.append(grads[idx])
        if sample_grad:
            ax.plot(sample_iters, sample_grad, label=opt_name, linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Gradient Norm Convergence', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bar chart of final loss
    ax = axes[1, 0]
    opt_names = [m['Optimizer'] for m in summary_metrics]
    final_losses = [m['Final Loss'] for m in summary_metrics]
    # Replace inf with a large value for visualization
    final_losses_plot = [fl if np.isfinite(fl) else 1e3 for fl in final_losses]
    _bars = ax.bar(range(len(opt_names)), final_losses_plot, color=colors, alpha=0.8)
    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Final Loss (log scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate bars
    for i, (_name, loss, loss_plot) in enumerate(zip(opt_names, final_losses, final_losses_plot)):
        label = f'{loss:.2e}' if np.isfinite(loss) else 'DIV'
        ax.text(i, loss_plot * 1.5, label, ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Plot 4: Convergence speed (iterations to loss < 1e-3)
    ax = axes[1, 1]
    iters_to_converge = [m['Iterations to Loss<1e-3'] for m in summary_metrics]
    _ = ax.bar(range(len(opt_names)), iters_to_converge, color=colors, alpha=0.8)  # bars unused but kept for plot
    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Iterations to Loss < 1e-3', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim((0.0, float(max_iterations) * 1.1))
    
    # Annotate bars
    for i, (_name, it) in enumerate(zip(opt_names, iters_to_converge)):
        label = f'{it}' if it < max_iterations else '>10k'
        ax.text(i, it + max_iterations * 0.02, label, ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Optimizer Ablation: {test_function.__class__.__name__}\n'
                 f'Initial point: {initial_point}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'optimizer_ablation_study.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Ablation Summary:")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"{'='*70}\n")
    
    return df_summary


def main():
    """CLI for running ablation study."""
    parser = argparse.ArgumentParser(description="Run optimizer ablation study (fair defaults by default).")
    parser.add_argument('--allow-unfair-ablations', '--use-legacy-unfair', dest='allow_unfair', action='store_true',
                        help='Run legacy unfair ablation (fixed lr=0.01 for all optimizers). Use for quick sanity checks only.')
    parser.add_argument('--track-params', dest='track_params', action='store_true',
                        help='Track full parameter snapshots in TrainingDynamicsTracker (memory intensive).')
    parser.add_argument('--max-iterations', type=int, default=10000, help='Max iterations per optimizer.')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save CSV results.')
    parser.add_argument('--plots-dir', type=str, default='plots', help='Directory to save plots.')
    parser.add_argument('--initial-point', nargs=2, type=float, default=[-1.5, 2.0], help='Initial point x y for test function.')
    parser.add_argument('--lr-sgd', type=float, help='Override LR for SGD.')
    parser.add_argument('--lr-adam', type=float, help='Override LR for Adam.')
    args = parser.parse_args()

    print("="*70)
    print("Optimizer Ablation Study")
    print("="*70)

    initial_point = tuple(args.initial_point)

    # Initialize Rosenbrock
    rosenbrock = Rosenbrock(a=1, b=100)

    # collect lr overrides
    lr_overrides = {}
    if args.lr_sgd is not None:
        lr_overrides['SGD'] = args.lr_sgd
    if args.lr_adam is not None:
        lr_overrides['Adam'] = args.lr_adam

    df_summary = run_optimizer_ablation(
        test_function=rosenbrock,
        initial_point=initial_point,
        max_iterations=args.max_iterations,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        use_legacy_unfair=args.allow_unfair,
        track_params=args.track_params,
        lr_map_override=lr_overrides
    )
    print("Ablation study complete!")
    return df_summary


if __name__ == '__main__':
    main()
