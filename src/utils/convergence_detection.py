"""
Adaptive Convergence Detection

This module implements robust, relative convergence detection that adapts
to the scale and characteristics of different optimization problems.

Addresses QA Issue #15: "Relative Convergence Threshold"
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Results from convergence detection."""
    converged: bool
    iteration: Optional[int]
    convergence_value: float
    threshold: float
    criterion: str  # 'absolute', 'relative', 'gradient', 'plateau'


class AdaptiveConvergenceDetector:
    """
    Adaptive convergence detection for optimization algorithms.

    Handles multiple convergence criteria:
    1. Absolute threshold: loss < threshold (for test functions near 0)
    2. Relative threshold: loss < best * (1 + tolerance) (for neural networks)
    3. Gradient norm: ||∇f|| < threshold (first-order stationarity)
    4. Plateau detection: std(recent_losses) < tolerance (no progress)

    The detector automatically selects appropriate criteria based on problem scale.
    """

    def __init__(
        self,
        absolute_loss_threshold: float = 1e-6,
        relative_tolerance: float = 0.01,  # 1% above best
        gradient_threshold: float = 1e-6,
        plateau_window: int = 50,
        plateau_tolerance: float = 1e-8,
        min_iterations: int = 10
    ):
        """
        Args:
            absolute_loss_threshold: Absolute loss threshold (for test functions)
            relative_tolerance: Relative tolerance (fraction above best loss)
            gradient_threshold: Gradient norm threshold
            plateau_window: Window size for plateau detection
            plateau_tolerance: Tolerance for plateau detection
            min_iterations: Minimum iterations before declaring convergence
        """
        self.absolute_threshold = absolute_loss_threshold
        self.relative_tolerance = relative_tolerance
        self.gradient_threshold = gradient_threshold
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        self.min_iterations = min_iterations

    def detect_convergence(
        self,
        losses: np.ndarray,
        grad_norms: Optional[np.ndarray] = None,
        prefer_relative: bool = True
    ) -> ConvergenceResult:
        """
        Detect convergence from loss trajectory and gradient norms.

        Strategy:
        1. If losses never reach near-zero and prefer_relative=True → use relative threshold
        2. If losses reach near-zero → use absolute threshold
        3. If gradient norms available → check gradient threshold
        4. If no convergence found → check for plateau

        Args:
            losses: Loss trajectory
            grad_norms: Gradient norm trajectory (optional)
            prefer_relative: Whether to prefer relative over absolute threshold

        Returns:
            ConvergenceResult with convergence status and details
        """
        if len(losses) < self.min_iterations:
            return ConvergenceResult(
                converged=False,
                iteration=None,
                convergence_value=float('inf'),
                threshold=self.absolute_threshold,
                criterion='insufficient_iterations'
            )

        # Filter out non-finite values
        finite_mask = np.isfinite(losses)
        if not np.any(finite_mask):
            return ConvergenceResult(
                converged=False,
                iteration=None,
                convergence_value=float('inf'),
                threshold=self.absolute_threshold,
                criterion='all_non_finite'
            )

        finite_losses = losses[finite_mask]
        best_loss = np.min(finite_losses)

        # Determine problem scale
        loss_scale = max(abs(best_loss), 1.0)

        # Strategy 1: Check gradient convergence (most reliable)
        if grad_norms is not None and len(grad_norms) > 0:
            grad_result = self._check_gradient_convergence(grad_norms)
            if grad_result.converged:
                return grad_result

        # Strategy 2: Check absolute convergence (for test functions)
        if not prefer_relative or best_loss < 1.0:
            abs_result = self._check_absolute_convergence(losses)
            if abs_result.converged:
                return abs_result

        # Strategy 3: Check relative convergence (for neural networks)
        if prefer_relative:
            rel_result = self._check_relative_convergence(losses, best_loss)
            if rel_result.converged:
                return rel_result

        # Strategy 4: Check plateau (no progress)
        plateau_result = self._check_plateau_convergence(losses)
        if plateau_result.converged:
            return plateau_result

        # No convergence detected
        return ConvergenceResult(
            converged=False,
            iteration=None,
            convergence_value=finite_losses[-1],
            threshold=self.absolute_threshold,
            criterion='not_converged'
        )

    def _check_absolute_convergence(self, losses: np.ndarray) -> ConvergenceResult:
        """Check if loss drops below absolute threshold."""
        for i, loss in enumerate(losses):
            if i < self.min_iterations:
                continue
            if np.isfinite(loss) and loss < self.absolute_threshold:
                return ConvergenceResult(
                    converged=True,
                    iteration=i,
                    convergence_value=loss,
                    threshold=self.absolute_threshold,
                    criterion='absolute_loss'
                )

        finite_losses = losses[np.isfinite(losses)]
        final_loss = finite_losses[-1] if len(finite_losses) > 0 else float('inf')

        return ConvergenceResult(
            converged=False,
            iteration=None,
            convergence_value=final_loss,
            threshold=self.absolute_threshold,
            criterion='absolute_loss'
        )

    def _check_relative_convergence(
        self,
        losses: np.ndarray,
        best_loss: float
    ) -> ConvergenceResult:
        """Check if loss reaches within relative tolerance of best loss."""
        threshold = best_loss * (1.0 + self.relative_tolerance)

        for i, loss in enumerate(losses):
            if i < self.min_iterations:
                continue
            if np.isfinite(loss) and loss < threshold:
                return ConvergenceResult(
                    converged=True,
                    iteration=i,
                    convergence_value=loss,
                    threshold=threshold,
                    criterion='relative_loss'
                )

        finite_losses = losses[np.isfinite(losses)]
        final_loss = finite_losses[-1] if len(finite_losses) > 0 else float('inf')

        return ConvergenceResult(
            converged=False,
            iteration=None,
            convergence_value=final_loss,
            threshold=threshold,
            criterion='relative_loss'
        )

    def _check_gradient_convergence(self, grad_norms: np.ndarray) -> ConvergenceResult:
        """Check if gradient norm drops below threshold."""
        for i, grad_norm in enumerate(grad_norms):
            if i < self.min_iterations:
                continue
            if np.isfinite(grad_norm) and grad_norm < self.gradient_threshold:
                return ConvergenceResult(
                    converged=True,
                    iteration=i,
                    convergence_value=grad_norm,
                    threshold=self.gradient_threshold,
                    criterion='gradient_norm'
                )

        finite_grads = grad_norms[np.isfinite(grad_norms)]
        final_grad = finite_grads[-1] if len(finite_grads) > 0 else float('inf')

        return ConvergenceResult(
            converged=False,
            iteration=None,
            convergence_value=final_grad,
            threshold=self.gradient_threshold,
            criterion='gradient_norm'
        )

    def _check_plateau_convergence(self, losses: np.ndarray) -> ConvergenceResult:
        """Check if optimization has plateaued (no progress)."""
        if len(losses) < self.plateau_window + self.min_iterations:
            return ConvergenceResult(
                converged=False,
                iteration=None,
                convergence_value=float('inf'),
                threshold=self.plateau_tolerance,
                criterion='plateau'
            )

        # Check last N iterations for plateau
        recent_losses = losses[-self.plateau_window:]
        finite_recent = recent_losses[np.isfinite(recent_losses)]

        if len(finite_recent) < self.plateau_window // 2:
            return ConvergenceResult(
                converged=False,
                iteration=None,
                convergence_value=float('inf'),
                threshold=self.plateau_tolerance,
                criterion='plateau'
            )

        # Compute relative standard deviation
        mean_loss = np.mean(finite_recent)
        std_loss = np.std(finite_recent)

        # Plateau if relative std is very small
        if mean_loss > 0:
            relative_std = std_loss / abs(mean_loss)
        else:
            relative_std = std_loss

        if relative_std < self.plateau_tolerance:
            return ConvergenceResult(
                converged=True,
                iteration=len(losses) - self.plateau_window,
                convergence_value=float(mean_loss),
                threshold=self.plateau_tolerance,
                criterion='plateau'
            )

        return ConvergenceResult(
            converged=False,
            iteration=None,
            convergence_value=float(mean_loss),
            threshold=self.plateau_tolerance,
            criterion='plateau'
        )


def detect_convergence_multiple_criteria(
    losses: np.ndarray,
    grad_norms: Optional[np.ndarray] = None,
    problem_type: str = 'neural_network'
) -> Dict[str, ConvergenceResult]:
    """
    Apply multiple convergence criteria and return all results.

    This is useful for research: report convergence under different definitions.

    Args:
        losses: Loss trajectory
        grad_norms: Gradient norm trajectory (optional)
        problem_type: 'test_function' or 'neural_network'

    Returns:
        Dict mapping criterion name to ConvergenceResult
    """
    results = {}

    # Configure detector based on problem type
    if problem_type == 'test_function':
        # Test functions: expect near-zero optima
        detector = AdaptiveConvergenceDetector(
            absolute_loss_threshold=1e-6,
            relative_tolerance=0.01,
            gradient_threshold=1e-6
        )
    else:
        # Neural networks: use relative convergence
        detector = AdaptiveConvergenceDetector(
            absolute_loss_threshold=1e-3,
            relative_tolerance=0.05,  # 5% above best
            gradient_threshold=1e-4
        )

    # Test each criterion individually
    results['absolute'] = detector._check_absolute_convergence(losses)

    finite_losses = losses[np.isfinite(losses)]
    if len(finite_losses) > 0:
        best_loss = np.min(finite_losses)
        results['relative'] = detector._check_relative_convergence(losses, best_loss)

    if grad_norms is not None:
        results['gradient'] = detector._check_gradient_convergence(grad_norms)

    results['plateau'] = detector._check_plateau_convergence(losses)

    return results


def recommend_convergence_threshold(
    losses: np.ndarray,
    problem_type: str = 'auto'
) -> Tuple[str, float]:
    """
    Automatically recommend convergence criterion and threshold.

    Args:
        losses: Loss trajectory
        problem_type: 'test_function', 'neural_network', or 'auto'

    Returns:
        criterion: Recommended criterion ('absolute' or 'relative')
        threshold: Recommended threshold value
    """
    finite_losses = losses[np.isfinite(losses)]
    if len(finite_losses) == 0:
        return 'absolute', 1e-6

    min_loss = np.min(finite_losses)
    max_loss = np.max(finite_losses)
    loss_range = max_loss - min_loss

    # Auto-detect problem type
    if problem_type == 'auto':
        if min_loss < 0.1 and loss_range > 10.0:
            # Looks like a test function (reaches near zero from far away)
            problem_type = 'test_function'
        else:
            # Looks like neural network (bounded range, may not reach zero)
            problem_type = 'neural_network'

    if problem_type == 'test_function':
        return 'absolute', max(1e-6, min_loss * 0.1)
    else:
        # Neural network: 5% above best
        return 'relative', min_loss * 1.05


# Example usage
if __name__ == '__main__':
    # Test function example (Rosenbrock)
    np.random.seed(42)
    test_fn_losses = np.logspace(2, -8, 1000)  # Exponential decay to near-zero
    test_fn_grads = np.logspace(1, -7, 1000)

    detector = AdaptiveConvergenceDetector()
    result = detector.detect_convergence(test_fn_losses, test_fn_grads, prefer_relative=False)
    print(f"Test Function - Converged: {result.converged} at iteration {result.iteration}")
    print(f"  Criterion: {result.criterion}, Value: {result.convergence_value:.2e}, Threshold: {result.threshold:.2e}")

    # Neural network example
    nn_losses = 2.3 - 2.0 * np.exp(-np.arange(1000) / 200) + 0.1 * np.random.randn(1000)
    nn_grads = 0.5 * np.exp(-np.arange(1000) / 300) + 0.01 * np.random.randn(1000)

    result = detector.detect_convergence(nn_losses, nn_grads, prefer_relative=True)
    print(f"\nNeural Network - Converged: {result.converged} at iteration {result.iteration}")
    print(f"  Criterion: {result.criterion}, Value: {result.convergence_value:.4f}, Threshold: {result.threshold:.4f}")

    # Multiple criteria
    all_results = detect_convergence_multiple_criteria(nn_losses, nn_grads, 'neural_network')
    print("\nAll Criteria:")
    for name, res in all_results.items():
        print(f"  {name}: converged={res.converged}, iter={res.iteration}")
