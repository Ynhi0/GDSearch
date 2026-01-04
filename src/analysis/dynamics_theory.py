"""
Theoretical Dynamics Predictions

This module implements theoretical predictions for optimizer dynamics:
- Velocity magnitude (step size effects)
- Oscillation amplitude (momentum effects)
- Trajectory curvature (second-order behavior)

These predictions enable theory-practice validation for dynamics metrics,
completing the scientific analysis loop.

References:
- Polyak 1964: Heavy ball momentum (oscillation theory)
- Nesterov 1983: Accelerated gradient (curvature adaptation)
- Sutskever et al. 2013: Momentum initialization effects
- Wilson et al. 2017: "The Marginal Value of Adaptive Gradient Methods"
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging


def theoretical_velocity_magnitude(
    lr: float,
    L: float,
    momentum: float = 0.0,
    sigma: float = 0.0
) -> Dict[str, float]:
    """
    Predict expected velocity (step size) magnitude.
    
    Theory:
    - SGD (momentum=0): E[||Δx_t||] ≈ η * E[||∇f||] ≈ η√L for normalized gradients
    - Momentum: E[||Δx_t||] ≈ η/(1-β) * E[||∇f||] (amplification by 1/(1-β))
    - Stochastic: Add noise contribution √(η²σ²)
    
    Args:
        lr: Learning rate (η)
        L: Lipschitz smoothness constant
        momentum: Momentum coefficient (β)
        sigma: Gradient noise level
        
    Returns:
        dict with:
          - expected_velocity: Predicted ||Δx_t||
          - deterministic_component: η/(1-β) * √L
          - stochastic_component: η * σ
          - amplification_factor: 1/(1-β) for momentum
    """
    # Deterministic component: step size * typical gradient magnitude
    # Assume typical gradient norm ≈ √L (based on smoothness)
    grad_magnitude = np.sqrt(L)
    
    # Momentum amplification
    if momentum > 0 and momentum < 1:
        amplification = 1.0 / (1.0 - momentum)
    else:
        amplification = 1.0
    
    deterministic_vel = lr * amplification * grad_magnitude
    
    # Stochastic component (noise adds to velocity)
    stochastic_vel = lr * sigma
    
    # Total expected velocity (RMS combination)
    expected_velocity = np.sqrt(deterministic_vel**2 + stochastic_vel**2)
    
    return {
        'expected_velocity': float(expected_velocity),
        'deterministic_component': float(deterministic_vel),
        'stochastic_component': float(stochastic_vel),
        'amplification_factor': float(amplification),
        'theory_source': 'Polyak 1964 (momentum) + stochastic perturbation'
    }


def theoretical_oscillation_amplitude(
    lr: float,
    L: float,
    mu: float,
    momentum: float,
    sigma: float = 0.0
) -> Dict[str, float]:
    """
    Predict oscillation amplitude for momentum-based methods.
    
    Theory (from Polyak 1964 heavy ball analysis):
    - Underdamped regime (β > β_critical): Oscillations with amplitude A ∝ 1/√μ
    - Overdamped regime (β < β_critical): Exponential decay, no oscillations
    - Critical damping: β_critical = (√κ - 1)/(√κ + 1) where κ = L/μ
    
    For stochastic case, noise adds random oscillations ∝ σ.
    
    Args:
        lr: Learning rate
        L: Lipschitz constant
        mu: Strong convexity parameter (use PL constant for non-convex)
        momentum: Momentum coefficient (β)
        sigma: Gradient noise level
        
    Returns:
        dict with:
          - expected_amplitude: Predicted oscillation amplitude
          - regime: 'underdamped', 'overdamped', or 'critically_damped'
          - critical_momentum: Optimal β for critical damping
          - frequency: Oscillation frequency (for underdamped)
    """
    if mu <= 1e-12:
        # Non-convex case: no theory for oscillations without curvature
        return {
            'expected_amplitude': float(sigma * lr),  # Noise-driven only
            'regime': 'non_convex',
            'critical_momentum': None,
            'frequency': None,
            'theory_source': 'No curvature-based theory; noise-driven'
        }
    
    # Condition number
    kappa = L / mu
    sqrt_kappa = np.sqrt(kappa)
    
    # Critical momentum (Polyak optimal)
    beta_critical = (sqrt_kappa - 1) / (sqrt_kappa + 1)
    
    # Determine regime
    if abs(momentum - beta_critical) < 0.05:
        regime = 'critically_damped'
        # Critical damping: minimal overshoot, amplitude ≈ initial error
        amplitude = lr * np.sqrt(L)  # Initial step size
    elif momentum > beta_critical:
        regime = 'underdamped'
        # Underdamped: oscillations with frequency ω = √(μ/L) and amplitude decay
        # Amplitude ∝ (momentum - beta_critical) / √μ
        overshoot_factor = (momentum - beta_critical) / (1 - beta_critical)
        amplitude = lr * np.sqrt(L) * (1 + overshoot_factor * np.sqrt(kappa))
    else:
        regime = 'overdamped'
        # Overdamped: slow exponential decay, no oscillations
        amplitude = lr * np.sqrt(L) * 0.5  # Reduced by slow approach
    
    # Add stochastic contribution
    noise_amplitude = sigma * lr
    total_amplitude = np.sqrt(amplitude**2 + noise_amplitude**2)
    
    # Oscillation frequency (for underdamped)
    if regime == 'underdamped':
        # Natural frequency ω ≈ √(μL) / (1 + momentum)
        frequency = np.sqrt(mu * L) / (1 + momentum)
    else:
        frequency = None
    
    return {
        'expected_amplitude': float(total_amplitude),
        'deterministic_amplitude': float(amplitude),
        'stochastic_amplitude': float(noise_amplitude),
        'regime': regime,
        'critical_momentum': float(beta_critical),
        'frequency': float(frequency) if frequency is not None else None,
        'overshoot_factor': float((momentum - beta_critical) / (1 - beta_critical)) if regime == 'underdamped' else 0.0,
        'theory_source': 'Polyak 1964 Heavy Ball Analysis'
    }


def theoretical_smoothness_index(
    lr: float,
    momentum: float,
    sigma: float = 0.0
) -> Dict[str, float]:
    """
    Predict trajectory smoothness (angle changes between steps).
    
    Theory:
    - Pure SGD: Random walk → angle ≈ 90° (π/2 rad) on average
    - Momentum: Correlation between steps → smaller angles
      Expected angle ≈ arccos(β) (momentum preserves direction)
    - Noise: Adds random deviation ∝ σ/||v||
    
    Args:
        lr: Learning rate
        momentum: Momentum coefficient (β)
        sigma: Gradient noise level
        
    Returns:
        dict with:
          - expected_angle: Mean angle change (radians)
          - angle_std: Standard deviation of angle
          - smoothness_improvement: Factor vs. vanilla SGD
    """
    # Pure SGD case (momentum = 0)
    if momentum < 1e-6:
        # Random walk: directions uncorrelated
        # E[cos(θ)] = 0 → E[θ] ≈ π/2
        expected_angle = np.pi / 2.0
        angle_std = np.pi / 4.0  # High variance
        smoothness_improvement = 1.0
    else:
        # Momentum case: direction correlation
        # E[cos(θ)] ≈ β (momentum preserves direction)
        # E[θ] ≈ arccos(β)
        expected_angle = np.arccos(momentum)
        
        # Noise adds random deviation
        # Approximate: angle_std ∝ (1 - β) * noise_factor
        noise_factor = sigma / (1e-3 + lr)  # Normalized noise
        angle_std = (1 - momentum) * np.pi / 4.0 + noise_factor * 0.1
        
        # Smoothness improvement vs. vanilla SGD
        # (π/2) / arccos(β) ≈ how much smoother
        smoothness_improvement = (np.pi / 2.0) / expected_angle
    
    return {
        'expected_angle': float(expected_angle),
        'expected_angle_degrees': float(np.degrees(expected_angle)),
        'angle_std': float(angle_std),
        'smoothness_improvement': float(smoothness_improvement),
        'momentum_correlation': float(momentum),
        'theory_source': 'Random walk (SGD) vs momentum correlation'
    }


def theoretical_path_efficiency(
    lr: float,
    L: float,
    mu: float,
    momentum: float,
    T: int
) -> Dict[str, float]:
    """
    Predict path efficiency (ratio of direct distance to path length).
    
    Theory:
    - Optimal path: Straight line from x_0 to x* (efficiency = 1.0)
    - SGD: Random walk → efficiency ≈ 1/√T (diffusion scaling)
    - Momentum: Reduces zigzagging → efficiency ≈ √(1-β)
    
    Args:
        lr: Learning rate
        L: Lipschitz constant
        mu: Strong convexity parameter
        momentum: Momentum coefficient
        T: Number of iterations
        
    Returns:
        dict with:
          - expected_efficiency: Path efficiency [0, 1]
          - theoretical_path_length: Expected total distance traveled
          - theoretical_displacement: Expected direct distance
    """
    # Approximate analysis based on random walk vs ballistic motion
    
    if mu <= 1e-12:
        # Non-convex: harder to predict, use empirical heuristics
        efficiency_base = 0.5  # Midpoint estimate
    else:
        # Convex case: efficiency improves with momentum
        # SGD: ~1/√T (random walk)
        # Momentum: ~(1-β) (straighter path)
        if momentum < 1e-6:
            efficiency_base = 1.0 / np.sqrt(T)
        else:
            efficiency_base = np.sqrt(1 - momentum)
    
    # Learning rate and smoothness affect straightness
    # Smaller η → smaller steps → straighter (less overshoot)
    lr_factor = min(1.0, 1.0 / (lr * L))
    
    expected_efficiency = efficiency_base * lr_factor
    expected_efficiency = np.clip(expected_efficiency, 0.0, 1.0)
    
    # Approximate path length and displacement
    # Assume initial error ||x_0 - x*|| = 1 (normalized)
    theoretical_displacement = 1.0  # Direct distance
    theoretical_path_length = theoretical_displacement / (expected_efficiency + 1e-10)
    
    return {
        'expected_efficiency': float(expected_efficiency),
        'theoretical_path_length': float(theoretical_path_length),
        'theoretical_displacement': float(theoretical_displacement),
        'lr_factor': float(lr_factor),
        'theory_source': 'Random walk (SGD) vs ballistic motion (momentum)'
    }


def compare_dynamics_theory_practice(
    measured_velocity: float,
    measured_oscillation: float,
    measured_smoothness: float,
    lr: float,
    L: float,
    momentum: float = 0.0,
    mu: Optional[float] = None,
    sigma: float = 0.0
) -> Dict[str, Dict]:
    """
    Compare measured dynamics with theoretical predictions.
    
    Args:
        measured_velocity: Observed mean velocity ||Δx_t||
        measured_oscillation: Observed oscillation amplitude
        measured_smoothness: Observed smoothness index (mean angle)
        lr: Learning rate used
        L: Lipschitz constant (measured)
        momentum: Momentum coefficient
        mu: Strong convexity / PL constant (if available)
        sigma: Gradient noise (measured)
        
    Returns:
        dict with theory-practice comparison for each metric:
          - velocity: {theory, practice, error, error_pct}
          - oscillation: {theory, practice, error, error_pct}
          - smoothness: {theory, practice, error, error_pct}
    """
    # Predict velocity
    vel_theory = theoretical_velocity_magnitude(lr, L, momentum, sigma)
    vel_error = abs(measured_velocity - vel_theory['expected_velocity'])
    vel_error_pct = (vel_error / (measured_velocity + 1e-10)) * 100
    
    # Predict oscillation (requires curvature)
    if mu is not None and mu > 1e-12:
        osc_theory = theoretical_oscillation_amplitude(lr, L, mu, momentum, sigma)
        osc_error = abs(measured_oscillation - osc_theory['expected_amplitude'])
        osc_error_pct = (osc_error / (measured_oscillation + 1e-10)) * 100
    else:
        osc_theory = {'expected_amplitude': None, 'regime': 'non_convex'}
        osc_error = None
        osc_error_pct = None
    
    # Predict smoothness
    smooth_theory = theoretical_smoothness_index(lr, momentum, sigma)
    smooth_error = abs(measured_smoothness - smooth_theory['expected_angle'])
    smooth_error_pct = (smooth_error / (measured_smoothness + 1e-10)) * 100
    
    return {
        'velocity': {
            'measured': float(measured_velocity),
            'theoretical': float(vel_theory['expected_velocity']),
            'error': float(vel_error),
            'error_pct': float(vel_error_pct),
            'details': vel_theory
        },
        'oscillation': {
            'measured': float(measured_oscillation),
            'theoretical': float(osc_theory['expected_amplitude']) if osc_theory['expected_amplitude'] is not None else None,
            'error': float(osc_error) if osc_error is not None else None,
            'error_pct': float(osc_error_pct) if osc_error_pct is not None else None,
            'details': osc_theory
        },
        'smoothness': {
            'measured': float(measured_smoothness),
            'theoretical': float(smooth_theory['expected_angle']),
            'error': float(smooth_error),
            'error_pct': float(smooth_error_pct),
            'details': smooth_theory
        }
    }


def generate_dynamics_theory_report(comparison: Dict) -> str:
    """
    Generate human-readable report for dynamics theory-practice comparison.
    
    Args:
        comparison: Output from compare_dynamics_theory_practice()
        
    Returns:
        str: Formatted report
    """
    report = []
    report.append("="*80)
    report.append("DYNAMICS THEORY-PRACTICE COMPARISON")
    report.append("="*80)
    report.append("")
    
    # Velocity
    vel = comparison['velocity']
    report.append("1. VELOCITY MAGNITUDE")
    report.append(f"   Measured:     {vel['measured']:.6f}")
    report.append(f"   Theoretical:  {vel['theoretical']:.6f}")
    report.append(f"   Error:        {vel['error']:.6f} ({vel['error_pct']:.2f}%)")
    if vel['error_pct'] < 10:
        report.append("   ✓ Good agreement (<10% error)")
    elif vel['error_pct'] < 25:
        report.append("   ⚠ Moderate agreement (10-25% error)")
    else:
        report.append("   ✗ Poor agreement (>25% error)")
    report.append("")
    
    # Oscillation
    osc = comparison['oscillation']
    report.append("2. OSCILLATION AMPLITUDE")
    if osc['theoretical'] is not None:
        report.append(f"   Measured:     {osc['measured']:.6f}")
        report.append(f"   Theoretical:  {osc['theoretical']:.6f}")
        report.append(f"   Error:        {osc['error']:.6f} ({osc['error_pct']:.2f}%)")
        report.append(f"   Regime:       {osc['details']['regime']}")
        if osc['error_pct'] < 15:
            report.append("   ✓ Good agreement (<15% error)")
        else:
            report.append("   ⚠ Moderate/poor agreement")
    else:
        report.append("   ⚠ No theoretical prediction (non-convex, no curvature)")
    report.append("")
    
    # Smoothness
    smooth = comparison['smoothness']
    report.append("3. TRAJECTORY SMOOTHNESS")
    report.append(f"   Measured:     {smooth['measured']:.4f} rad ({np.degrees(smooth['measured']):.1f}°)")
    report.append(f"   Theoretical:  {smooth['theoretical']:.4f} rad ({np.degrees(smooth['theoretical']):.1f}°)")
    report.append(f"   Error:        {smooth['error']:.4f} rad ({smooth['error_pct']:.2f}%)")
    if smooth['error_pct'] < 20:
        report.append("   ✓ Good agreement (<20% error)")
    else:
        report.append("   ⚠ Moderate/poor agreement")
    report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)


if __name__ == '__main__':
    # Example usage
    print("Dynamics Theory Predictions - Example")
    print("="*80)
    
    # Test case: SGD with momentum on a smooth strongly convex problem
    lr = 0.01
    L = 10.0
    mu = 0.1
    momentum = 0.9
    sigma = 0.01
    
    print(f"\nParameters: lr={lr}, L={L}, μ={mu}, β={momentum}, σ={sigma}")
    print()
    
    # Velocity prediction
    vel_pred = theoretical_velocity_magnitude(lr, L, momentum, sigma)
    print("VELOCITY PREDICTION:")
    print(f"  Expected magnitude: {vel_pred['expected_velocity']:.6f}")
    print(f"  Amplification:      {vel_pred['amplification_factor']:.2f}x")
    print()
    
    # Oscillation prediction
    osc_pred = theoretical_oscillation_amplitude(lr, L, mu, momentum, sigma)
    print("OSCILLATION PREDICTION:")
    print(f"  Expected amplitude: {osc_pred['expected_amplitude']:.6f}")
    print(f"  Regime:            {osc_pred['regime']}")
    print(f"  Critical β:        {osc_pred['critical_momentum']:.4f}")
    print()
    
    # Smoothness prediction
    smooth_pred = theoretical_smoothness_index(lr, momentum, sigma)
    print("SMOOTHNESS PREDICTION:")
    print(f"  Expected angle:    {smooth_pred['expected_angle']:.4f} rad ({smooth_pred['expected_angle_degrees']:.1f}°)")
    print(f"  Improvement:       {smooth_pred['smoothness_improvement']:.2f}x vs vanilla SGD")
    print()
    
    print("="*80)
    print("✓ Dynamics theory module functional")
