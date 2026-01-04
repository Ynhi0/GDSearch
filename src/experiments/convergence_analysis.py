"""
Convergence Analysis for Non-Convex Optimization Problems

Provides comprehensive empirical convergence analysis including:
- Convergence rate estimation
- Sublinear/linear/superlinear convergence detection
- Convergence diagnostics (stagnation, oscillation, divergence)
- Statistical convergence metrics across multiple seeds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Sequence
from scipy import stats
import warnings
import logging
from numpy.typing import ArrayLike


class ConvergenceAnalyzer:
    """Analyze convergence properties of optimization trajectories."""
    
    def __init__(self, tolerance: float = 1e-6, window_size: int = 50):
        """
        Initialize convergence analyzer.
        
        Args:
            tolerance: Convergence threshold for loss/gradient norm
            window_size: Window size for convergence rate estimation
        """
        self.tolerance = tolerance
        self.window_size = window_size
    
    def analyze_trajectory(
        self,
        losses: ArrayLike,
        grad_norms: Optional[ArrayLike] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single optimization trajectory.
        
        Args:
            losses: Array of loss values over iterations
            grad_norms: Optional array of gradient norms
            
        Returns:
            Dictionary with convergence metrics
        """
        losses = np.array(losses)
        n = len(losses)
        
        # Handle empty or invalid trajectories
        if n == 0:
            return self._empty_metrics()
        
        # Filter out non-finite values
        finite_mask = np.isfinite(losses)
        if not np.any(finite_mask):
            return self._diverged_metrics()
        
        losses_clean = losses[finite_mask]
        iterations_clean = np.arange(n)[finite_mask]
        
        # Basic metrics
        final_loss = losses_clean[-1]
        min_loss = np.min(losses_clean)
        initial_loss = losses_clean[0]
        
        # Convergence detection
        converged = final_loss < self.tolerance
        convergence_iter = self._find_convergence_iteration(losses_clean)
        
        # Convergence rate estimation
        conv_rate_type, conv_rate_value = self._estimate_convergence_rate(
            iterations_clean, losses_clean
        )
        
        # Stagnation detection
        stagnation_detected, stagnation_iter = self._detect_stagnation(losses_clean)
        
        # Oscillation analysis
        oscillation_metric = self._compute_oscillation(losses_clean)
        
        # Progress metrics
        total_reduction = initial_loss - final_loss
        reduction_ratio = total_reduction / initial_loss if initial_loss > 0 else 0
        
        # Gradient-based metrics (if available)
        grad_metrics = {}
        if grad_norms is not None and np.asarray(grad_norms).size > 0:
            grad_norms = np.array(grad_norms)[finite_mask]
            grad_metrics = {
                'final_grad_norm': grad_norms[-1],
                'min_grad_norm': np.min(grad_norms),
                'grad_converged': grad_norms[-1] < self.tolerance,
                'grad_convergence_iter': self._find_convergence_iteration(grad_norms)
            }
        
        return {
            'converged': converged,
            'convergence_iter': convergence_iter,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'initial_loss': initial_loss,
            'total_reduction': total_reduction,
            'reduction_ratio': reduction_ratio,
            'convergence_rate_type': conv_rate_type,
            'convergence_rate_value': conv_rate_value,
            'stagnation_detected': stagnation_detected,
            'stagnation_iter': stagnation_iter,
            'oscillation_metric': oscillation_metric,
            'total_iterations': n,
            'finite_iterations': len(losses_clean),
            **grad_metrics
        }
    
    def compare_optimizers(
        self,
        trajectories: Dict[str, List[Dict[str, np.ndarray]]]
    ) -> pd.DataFrame:
        """
        Compare convergence across multiple optimizers and seeds.
        
        Args:
            trajectories: Dict mapping optimizer names to lists of trajectories
                         Each trajectory is a dict with 'losses' and optionally 'grad_norms'
                         
        Returns:
            DataFrame with statistical comparison
        """
        results = []
        
        for opt_name, traj_list in trajectories.items():
            metrics_list = []
            
            for traj in traj_list:
                losses = traj.get('losses', [])
                grad_norms = traj.get('grad_norms', None)
                metrics = self.analyze_trajectory(losses, grad_norms)
                metrics_list.append(metrics)
            
            # Aggregate across seeds
            if metrics_list:
                agg_metrics = self._aggregate_metrics(metrics_list)
                agg_metrics['optimizer'] = opt_name
                agg_metrics['n_seeds'] = len(metrics_list)
                results.append(agg_metrics)
        
        df = pd.DataFrame(results)
        
        # Sort by convergence rate
        if 'mean_convergence_rate_value' in df.columns:
            from typing import cast
            df = cast(pd.DataFrame, df).sort_values(by=['mean_convergence_rate_value'], ascending=False)
        
        return df
    
    def _find_convergence_iteration(self, values: ArrayLike) -> Optional[int]:
        """Find first iteration where value < tolerance."""
        vals = np.asarray(values)
        converged_mask = vals < self.tolerance
        if np.any(converged_mask):
            return int(np.argmax(converged_mask))
        return None
    
    def _estimate_convergence_rate(
        self,
        iterations: np.ndarray,
        losses: np.ndarray
    ) -> Tuple[str, float]:
        """
        Estimate convergence rate type and value.
        
        Returns:
            (rate_type, rate_value) where rate_type is one of:
            - 'sublinear': O(1/k) or slower
            - 'linear': O(ρ^k) with ρ < 1
            - 'superlinear': faster than linear
            - 'unknown': cannot determine
        """
        if len(losses) < self.window_size:
            return 'unknown', np.nan
        
        # Use last window for rate estimation
        window_losses = losses[-self.window_size:]
        window_iters = iterations[-self.window_size:]
        
        # Filter out non-positive losses for log analysis
        positive_mask = window_losses > 1e-16
        if not np.any(positive_mask):
            return 'converged_exactly', 0.0
        
        window_losses = window_losses[positive_mask]
        window_iters = window_iters[positive_mask]
        
        if len(window_losses) < 10:
            return 'unknown', np.nan
        
        # Try linear fit in log space: log(loss) ~ c1 * k + c0
        # If slope is negative and significant, we have geometric convergence
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_losses = np.log(window_losses)
                
                # Remove non-finite log values
                finite_mask = np.isfinite(log_losses)
                if np.sum(finite_mask) < 5:
                    return 'unknown', np.nan
                
                log_losses = log_losses[finite_mask]
                iters = window_iters[finite_mask]
                
                # Linear regression in log space
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    iters, log_losses
                )

                # Cast regression outputs to scalars to satisfy static typing
                slope = self._to_scalar(slope)
                p_value = self._to_scalar(p_value)
                r_value = self._to_scalar(r_value)

                # Check if slope is significantly negative
                if p_value < 0.05 and slope < -1e-6:
                    # Linear convergence rate: ρ = exp(slope)
                    rho = float(np.exp(slope))
                    if 0 < rho < 1:
                        return 'linear', float(abs(slope))
                    else:
                        return 'superlinear', float(abs(slope))

                # GAP 32 FIX: Test for O(1/√k) convergence (standard non-convex SGD rate)
                # Standard non-convex SGD theory predicts convergence at rate O(1/√k).
                # Previously only testing 1/k and linear - missing the theoretically
                # predicted rate for Neural Networks!
                inv_iters_sqrt = 1.0 / np.sqrt(iters + 1)
                slope_sqrt, _, r_value_sqrt, p_value_sqrt, _ = stats.linregress(
                    inv_iters_sqrt, window_losses
                )
                
                slope_sqrt = self._to_scalar(slope_sqrt)
                p_value_sqrt = self._to_scalar(p_value_sqrt)
                r_value_sqrt = self._to_scalar(r_value_sqrt)
                
                # Try standard sublinear fit: loss ~ c / k (convex rate)
                inv_iters = 1.0 / (iters + 1)
                slope_sub, _, r_value_sub, p_value_sub, _ = stats.linregress(
                    inv_iters, window_losses
                )

                slope_sub = self._to_scalar(slope_sub)
                p_value_sub = self._to_scalar(p_value_sub)
                r_value_sub = self._to_scalar(r_value_sub)

                # GAP 32 FIX: Choose best fitting rate model
                # Compare R² values to determine which rate fits best
                r2_sqrt = r_value_sqrt ** 2 if p_value_sqrt < 0.05 else 0.0
                r2_sub = r_value_sub ** 2 if p_value_sub < 0.05 else 0.0
                
                if r2_sqrt > r2_sub and r2_sqrt > 0.7:
                    # O(1/√k) fits best - standard non-convex SGD rate
                    return 'root_sublinear', float(abs(slope_sqrt))
                elif r2_sub > 0.7:
                    # O(1/k) fits best - convex rate
                    return 'sublinear', float(abs(slope_sub))
                
        except Exception as e:
            logging.debug("Convergence detection failed: %s", e, exc_info=True)
            pass
        
        return 'unknown', np.nan
    
    def _to_scalar(self, x) -> float:
        """Safely convert a variety of numeric-like inputs to a Python float.

        Handles numpy scalars, 0-d arrays, 1-element iterables and objects with
        __float__ defined. Raises TypeError if conversion isn't possible.
        """
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        # Prefer __float__ if available
        if hasattr(x, "__float__"):
            try:
                return float(x)
            except Exception:
                pass
        # Try numpy conversion
        try:
            arr = np.asarray(x)
            if arr.size == 1:
                return float(arr.item())
        except Exception:
            pass
        # Fallback: treat as iterable and take first element
        try:
            it = iter(x)
            first = next(it)
            return float(first)
        except Exception as e:
            raise TypeError(f"Cannot convert object of type {type(x)} to float") from e

    def _detect_stagnation(
        self,
        losses: np.ndarray,
        threshold: float = 1e-8
    ) -> Tuple[bool, Optional[int]]:
        """
        Detect if optimization has stagnated.
        
        Returns:
            (stagnated, stagnation_iter)
        """
        if len(losses) < self.window_size * 2:
            return False, None
        
        # Check if loss change in recent window is very small
        recent_window = losses[-self.window_size:]
        loss_std = np.std(recent_window)
        loss_range = np.max(recent_window) - np.min(recent_window)
        
        # Stagnation if very small variation
        if loss_std < threshold and loss_range < threshold:
            # Find when stagnation started
            for i in range(len(losses) - self.window_size, 0, -1):
                window = losses[i:i + self.window_size]
                if np.std(window) > threshold:
                    return True, i + self.window_size
            return True, self.window_size
        
        return False, None
    
    def _compute_oscillation(self, losses: np.ndarray) -> float:
        """
        Compute oscillation metric (normalized standard deviation of differences).
        
        Returns:
            Oscillation metric (higher = more oscillatory)
        """
        if len(losses) < 2:
            return 0.0
        
        diffs = np.diff(losses)
        if len(diffs) == 0:
            return 0.0
        
        # Normalize by mean absolute loss
        mean_loss = np.mean(np.abs(losses))
        if mean_loss < 1e-16:
            return 0.0
        
        oscillation = np.std(diffs) / (mean_loss + 1e-16)
        return float(oscillation)
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across multiple seeds."""
        if not metrics_list:
            return {}
        
        agg = {}
        
        # Boolean metrics: success rate
        for key in ['converged', 'stagnation_detected']:
            if key in metrics_list[0]:
                values = [m[key] for m in metrics_list]
                agg[f'{key}_rate'] = np.mean(values)
        
        # Numeric metrics: mean ± std
        numeric_keys = [
            'convergence_iter', 'final_loss', 'min_loss',
            'total_reduction', 'reduction_ratio',
            'convergence_rate_value', 'oscillation_metric',
            'total_iterations', 'finite_iterations',
            'final_grad_norm', 'min_grad_norm'
        ]
        
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if values:
                finite_values = [v for v in values if np.isfinite(v)]
                if finite_values:
                    agg[f'mean_{key}'] = np.mean(finite_values)
                    agg[f'std_{key}'] = np.std(finite_values)
                    agg[f'min_{key}'] = np.min(finite_values)
                    agg[f'max_{key}'] = np.max(finite_values)
        
        # Categorical metrics: mode
        for key in ['convergence_rate_type']:
            if key in metrics_list[0]:
                values = [m[key] for m in metrics_list if m[key] != 'unknown']
                if values:
                    agg[f'primary_{key}'] = max(set(values), key=values.count)
        
        return agg
    
    def _empty_metrics(self) -> Dict:
        """Return metrics for empty trajectory."""
        return {
            'converged': False,
            'convergence_iter': None,
            'final_loss': np.nan,
            'min_loss': np.nan,
            'initial_loss': np.nan,
            'total_reduction': 0,
            'reduction_ratio': 0,
            'convergence_rate_type': 'unknown',
            'convergence_rate_value': np.nan,
            'stagnation_detected': False,
            'stagnation_iter': None,
            'oscillation_metric': 0,
            'total_iterations': 0,
            'finite_iterations': 0
        }
    
    def _diverged_metrics(self) -> Dict:
        """Return metrics for diverged trajectory."""
        return {
            'converged': False,
            'convergence_iter': None,
            'final_loss': np.inf,
            'min_loss': np.inf,
            'initial_loss': np.inf,
            'total_reduction': 0,
            'reduction_ratio': 0,
            'convergence_rate_type': 'diverged',
            'convergence_rate_value': np.nan,
            'stagnation_detected': False,
            'stagnation_iter': None,
            'oscillation_metric': np.inf,
            'total_iterations': 0,
            'finite_iterations': 0
        }


def analyze_non_convex_convergence(
    results_df: pd.DataFrame,
    optimizer_col: str = 'optimizer',
    loss_col: str = 'train_loss',  # GAP 38 FIX: Default to train_loss, not test_loss
    seed_col: str = 'seed'
) -> pd.DataFrame:
    """
    Analyze convergence for non-convex problems from experiment results.
    
    GAP 38 FIX - Optimization vs Generalization:
        Convergence rate analysis MUST use TRAINING loss, not test loss.
        
        Scientific reasoning:
        - Optimization: Study of minimizing the Training Loss f(θ)
        - Generalization: Study of minimizing the Test Loss (held-out data)
        
        Using test_loss for convergence rate is scientifically incorrect because:
        1. A fast optimizer drives train_loss → 0 quickly (fast convergence)
        2. But may overfit, causing test_loss to increase (apparent "divergence")
        3. Measuring speed on test_loss penalizes the optimizer for doing its job
        
        Correct approach:
        - Convergence Rate: Analyze on train_loss
        - Final Quality: Report on test_loss separately
        
    Args:
        results_df: DataFrame with experiment results
        optimizer_col: Column name for optimizer
        loss_col: Column name for loss values (default: 'train_loss')
        seed_col: Column name for seed
        
    Returns:
        DataFrame with convergence analysis results
    """
    analyzer = ConvergenceAnalyzer()
    
    # Group by optimizer and seed
    trajectories = {}

    opt_values = results_df[optimizer_col]
    if isinstance(opt_values, pd.Series):
        unique_opts = opt_values.unique()
    else:
        unique_opts = pd.Series(opt_values).unique()

    for opt in unique_opts:
        opt_data = results_df[results_df[optimizer_col] == opt]
        traj_list = []

        seed_values = opt_data[seed_col]
        if isinstance(seed_values, pd.Series):
            unique_seeds = seed_values.unique()
        else:
            unique_seeds = pd.Series(seed_values).unique()

        for seed in unique_seeds:
            subset = opt_data[opt_data[seed_col] == seed]
            if not isinstance(subset, pd.DataFrame):
                subset = pd.DataFrame(subset)
            from typing import cast
            seed_data = cast(pd.DataFrame, subset).sort_values(by=['epoch'])
            losses = seed_data[loss_col].values

            traj = {'losses': losses}

            # Add gradient norms if available
            if 'grad_norm' in seed_data.columns:
                traj['grad_norms'] = seed_data['grad_norm'].values

            traj_list.append(traj)

        trajectories[opt] = traj_list
    
    # Compare optimizers
    comparison_df = analyzer.compare_optimizers(trajectories)
    
    return comparison_df


if __name__ == '__main__':
    # Demo: Analyze synthetic convergence trajectories
    print("="*80)
    print(" "*20 + "CONVERGENCE ANALYSIS DEMO")
    print("="*80)
    
    # Generate synthetic trajectories
    np.random.seed(42)
    
    # Linear convergence (geometric decay)
    linear_conv = 10 * 0.9**np.arange(100) + np.random.randn(100) * 0.01
    
    # Sublinear convergence (1/k decay)
    sublinear_conv = 10 / (np.arange(100) + 1) + np.random.randn(100) * 0.01
    
    # Stagnating
    stagnating = np.concatenate([
        10 * 0.9**np.arange(50),
        np.ones(50) * (10 * 0.9**50) + np.random.randn(50) * 1e-8
    ])
    
    analyzer = ConvergenceAnalyzer()
    
    print("\n1. Linear Convergence:")
    metrics = analyzer.analyze_trajectory(linear_conv)
    print(f"   Type: {metrics['convergence_rate_type']}")
    print(f"   Rate: {metrics['convergence_rate_value']:.6f}")
    print(f"   Converged: {metrics['converged']}")
    
    print("\n2. Sublinear Convergence:")
    metrics = analyzer.analyze_trajectory(sublinear_conv)
    print(f"   Type: {metrics['convergence_rate_type']}")
    print(f"   Rate: {metrics['convergence_rate_value']:.6f}")
    print(f"   Converged: {metrics['converged']}")
    
    print("\n3. Stagnating:")
    metrics = analyzer.analyze_trajectory(stagnating)
    print(f"   Type: {metrics['convergence_rate_type']}")
    print(f"   Stagnation detected: {metrics['stagnation_detected']}")
    print(f"   Stagnation iter: {metrics['stagnation_iter']}")
    
    print("\nDemo complete!")
