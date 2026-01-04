"""
Scientific Integrity Fixes for GDSearch Optimizer Comparison

This module addresses critical methodological flaws identified in thesis review:
1. "Fake SGD" Problem: Adds gradient noise to make SGD truly stochastic
2. Multi-Function Suite: Tests on Rosenbrock, Ackley, SaddlePoint, Ill-Conditioned Quadratic
3. Realistic Conditioning: Uses kappa=1000-10000 to match neural network difficulty
4. Proper Noise Injection: Multiplicative noise that vanishes at stationary points

Author: Senior Principal Software Engineer & Codebase Janitor
Date: 2026-01-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Import test functions
from src.core.test_functions import (
    Rosenbrock, Ackley2D, SaddlePoint, IllConditionedQuadratic
)

# Import optimizers
from src.core.optimizers import SGD, SGDMomentum, Adam, RMSProp, SAM  # Fixed: RMSProp not RMSprop

# Import config loader
from src.utils.config_loader import load_optimizer_config, ConfigurationError


def run_stochastic_2d_experiments(
    results_dir: str = "results/stochastic_2d",
    seeds: Optional[List[int]] = None,
    noise_std: float = 0.1,
    noise_type: str = 'multiplicative',
    max_iter: int = 5000,
    resume: bool = False
) -> pd.DataFrame:
    """
    Run 2D optimization experiments with PROPER stochastic gradient noise.
    
    This fixes the "Fake SGD" problem: deterministic analytical gradients
    are not SGD. Real SGD has gradient noise from mini-batch sampling.
    
    Academic rigor improvements:
    - Injects gradient noise to simulate mini-batch stochasticity
    - Tests multiple test function topologies (valleys, saddles, multi-modal)
    - Uses realistic condition numbers (kappa >= 1000)
    - Applies multiplicative noise (vanishes at stationary points, like real SGD)
    
    Args:
        results_dir: Output directory
        seeds: List of random seeds for reproducibility
        noise_std: Standard deviation of gradient noise (0.1 recommended)
        noise_type: 'multiplicative' (realistic, default) or 'additive' (simple)
        max_iter: Maximum iterations
        resume: Skip if results already exist
    
    Returns:
        DataFrame with convergence results
    """
    print("\n" + "="*80)
    print("🔬 STOCHASTIC 2D OPTIMIZATION EXPERIMENTS (Proper SGD)")
    print("="*80)
    print(f"   Gradient Noise: {noise_type} (std={noise_std})")
    print(f"   Note: Without noise, this is Gradient Descent (GD), NOT SGD!")
    print("="*80)
    
    if seeds is None:
        seeds = [42, 123, 456]
    
    # Check resume
    if resume:
        result_file = Path(results_dir) / "stochastic_2d_results.csv"
        if result_file.exists():
            try:
                df = pd.read_csv(result_file)
                if len(df) > 0:
                    logging.info(f"Skipping Stochastic 2D experiment (already completed)")
                    return df
            except Exception:
                pass
    
    # Test function suite (addresses "One-Trick Pony" problem)
    test_functions = [
        ("Rosenbrock_Easy", Rosenbrock(a=1, b=100), (-1.5, 2.0)),
        ("Rosenbrock_Hard", Rosenbrock(a=1, b=1000), (-1.5, 2.0)),  # Harder conditioning
        ("Ackley2D", Ackley2D(), (-2.0, 2.0)),  # Multi-modal landscape
        ("SaddlePoint", SaddlePoint(), (0.5, 0.5)),  # Tests escape capability
        ("IllConditioned_Easy", IllConditionedQuadratic(kappa=100), (1.0, 1.0)),
        ("IllConditioned_Realistic", IllConditionedQuadratic(kappa=1000), (1.0, 1.0)),  # Matches NN difficulty
        ("IllConditioned_Hard", IllConditionedQuadratic(kappa=10000), (1.0, 1.0)),  # Extreme conditioning
    ]
    
    # Optimizer configurations (load from config with fallback)
    optimizers = []
    try:
        sgd_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'SGD')
        sgdm_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'SGDMomentum')
        adam_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'Adam')
        rmsprop_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'RMSProp')
        # SAM config (if available, otherwise use default)
        try:
            sam_cfg = load_optimizer_config('benchmark_hyperparameters', '2d_optimization', 'SAM')
            sam_opt = SAM(**sam_cfg)
        except (ConfigurationError, FileNotFoundError):
            sam_opt = SAM(lr=0.01, rho=0.05, base_optimizer='SGD')
        
        optimizers = [
            ("SGD", SGD(**sgd_cfg)),
            ("SGD_Momentum", SGDMomentum(**sgdm_cfg)),
            ("Adam", Adam(**adam_cfg)),
            ("RMSProp", RMSProp(**rmsprop_cfg)),
            ("SAM", sam_opt),
        ]
        logging.debug("Loaded optimizer configs from benchmark_hyperparameters.json")
    except (ConfigurationError, FileNotFoundError) as e:
        logging.warning("Failed to load config, using hardcoded defaults: %s", str(e))
        optimizers = [
            ("SGD", SGD(lr=0.01)),
            ("SGD_Momentum", SGDMomentum(lr=0.05, beta=0.9)),
            ("Adam", Adam(lr=0.1)),
            ("RMSProp", RMSProp(lr=0.01)),
            ("SAM", SAM(lr=0.01, rho=0.05, base_optimizer='SGD')),
        ]
    
    results = []
    
    for func_name, func, start_point in test_functions:
        print(f"\n[FUNCTION] {func_name}")
        print("-" * 40)
        
        for opt_name, optimizer in optimizers:
            for seed in seeds:
                np.random.seed(seed)
                
                # Initialize position
                x, y = start_point
                history = []
                
                # Reset optimizer state
                optimizer.reset()
                
                for iteration in range(max_iter):
                    # Compute loss
                    loss = func.compute(x, y)
                    history.append({'iteration': iteration, 'x': x, 'y': y, 'loss': loss})
                    
                    # Compute STOCHASTIC gradient (this is the key fix!)
                    grad_x, grad_y = func.gradient(x, y, noise_std=noise_std, noise_type=noise_type)
                    
                    # Update parameters (with SAM-specific handling)
                    if isinstance(optimizer, SAM):
                        # SAM requires adversarial gradients
                        # Compute adversarial point manually (avoid protected method warning)
                        grad_norm = np.hypot(grad_x, grad_y)
                        if grad_norm >= 1e-12:
                            # Normalize gradient direction
                            grad_dir_x = grad_x / grad_norm
                            grad_dir_y = grad_y / grad_norm
                            # Adversarial step: θ + ρ * (g / ||g||)
                            # SAM guaranteed to have .rho attribute
                            if hasattr(optimizer, 'rho'):
                                rho_value = optimizer.rho
                            else:
                                rho_value = 0.05  # Default SAM rho
                            adv_x = x + rho_value * grad_dir_x
                            adv_y = y + rho_value * grad_dir_y
                        else:
                            adv_x, adv_y = x, y
                        
                        # Compute gradient at adversarial point (with same noise characteristics)
                        adv_grad_x, adv_grad_y = func.gradient(adv_x, adv_y, noise_std=noise_std, noise_type=noise_type)
                        adversarial_gradients = (adv_grad_x, adv_grad_y)
                        
                        # SAM step with adversarial gradients
                        x, y = optimizer.step((x, y), (grad_x, grad_y), adversarial_gradients=adversarial_gradients)
                    else:
                        # Standard optimizer step
                        x, y = optimizer.step((x, y), (grad_x, grad_y))
                    
                    # Convergence check
                    if loss < 1e-8:
                        break
                
                final_loss = history[-1]['loss'] if history else float('nan')
                converged = final_loss < 1e-6
                
                results.append({
                    'function': func_name,
                    'optimizer': opt_name,
                    'seed': seed,
                    'noise_std': noise_std,
                    'noise_type': noise_type,
                    'final_loss': final_loss,
                    'final_x': x,
                    'final_y': y,
                    'iterations': len(history),
                    'converged': converged
                })
                
                print(f"  {opt_name:15s} (seed {seed}): Loss={final_loss:.6e}, Iters={len(history):4d}, Converged={converged}")
    
    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{results_dir}/stochastic_2d_results.csv", index=False)
    
    print(f"\n💾 Results saved to {results_dir}/stochastic_2d_results.csv")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Convergence Rate by Function and Optimizer")
    print("="*80)
    
    summary = df.groupby(['function', 'optimizer']).agg({
        'converged': 'mean',
        'iterations': 'mean',
        'final_loss': 'mean'
    }).round(4)
    
    print(summary)
    
    return df


def compare_deterministic_vs_stochastic(
    results_dir: str = "results/gd_vs_sgd_comparison",
    seeds: Optional[List[int]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compare deterministic GD (noise_std=0) vs. stochastic SGD (noise_std>0).
    
    This experiment demonstrates the critical difference between:
    - Gradient Descent (GD): deterministic, follows exact gradient
    - Stochastic Gradient Descent (SGD): noisy gradients, can escape local minima
    
    Returns:
        Dictionary with 'deterministic' and 'stochastic' DataFrames
    """
    print("\n" + "="*80)
    print("🔬 CRITICAL EXPERIMENT: Gradient Descent (GD) vs. SGD")
    print("="*80)
    print("   This demonstrates why gradient noise matters!")
    print("="*80)
    
    if seeds is None:
        seeds = [42, 123, 456]
    
    # Run deterministic (FAKE SGD)
    print("\n[PHASE 1] Running DETERMINISTIC Gradient Descent (noise_std=0)")
    print("   WARNING: This is NOT SGD, despite being called 'SGD' in code!")
    df_deterministic = run_stochastic_2d_experiments(
        results_dir=f"{results_dir}/deterministic",
        seeds=seeds,
        noise_std=0.0,  # No noise = not stochastic!
        max_iter=5000,
        resume=False
    )
    
    # Run stochastic (REAL SGD)
    print("\n[PHASE 2] Running STOCHASTIC Gradient Descent (noise_std=0.1)")
    print("   This is PROPER SGD with gradient noise!")
    df_stochastic = run_stochastic_2d_experiments(
        results_dir=f"{results_dir}/stochastic",
        seeds=seeds,
        noise_std=0.1,
        noise_type='multiplicative',
        max_iter=5000,
        resume=False
    )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Deterministic GD vs. Stochastic SGD")
    print("="*80)
    
    comparison = []
    for func in df_deterministic['function'].unique():
        for opt in df_deterministic['optimizer'].unique():
            det_subset = df_deterministic[(df_deterministic['function'] == func) & (df_deterministic['optimizer'] == opt)]
            stoch_subset = df_stochastic[(df_stochastic['function'] == func) & (df_stochastic['optimizer'] == opt)]
            
            comparison.append({
                'function': func,
                'optimizer': opt,
                'GD_converged_rate': det_subset['converged'].mean(),
                'SGD_converged_rate': stoch_subset['converged'].mean(),
                'GD_avg_iters': det_subset['iterations'].mean(),
                'SGD_avg_iters': stoch_subset['iterations'].mean(),
                'GD_final_loss': det_subset['final_loss'].mean(),
                'SGD_final_loss': stoch_subset['final_loss'].mean(),
            })
    
    df_comparison = pd.DataFrame(comparison)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(f"{results_dir}/gd_vs_sgd_comparison.csv", index=False)
    
    print(df_comparison)
    
    return {
        'deterministic': df_deterministic,
        'stochastic': df_stochastic,
        'comparison': df_comparison
    }


if __name__ == "__main__":
    # Run the critical experiment
    results = compare_deterministic_vs_stochastic()
    
    print("\n" + "="*80)
    print("✅ SCIENTIFIC INTEGRITY RESTORED")
    print("="*80)
    print("   Key Fixes Applied:")
    print("   1. ✅ Gradient noise injection (multiplicative)")
    print("   2. ✅ Multi-function test suite (Rosenbrock, Ackley, Saddle, Ill-Conditioned)")
    print("   3. ✅ Realistic condition numbers (kappa=1000-10000)")
    print("   4. ✅ Proper GD vs. SGD comparison")
    print("="*80)
