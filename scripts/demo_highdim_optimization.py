"""
Demonstration of high-dimensional test functions with custom optimizers.

This script trains Adam optimizer on various high-dimensional benchmark functions
to demonstrate how well the custom optimizer handles complex optimization landscapes.
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.test_functions import Rastrigin, Ackley, Sphere, Schwefel
from src.core.optimizers import Adam, SGDMomentum


def optimize_function(func, optimizer_class, optimizer_kwargs, max_iters=1000, 
                     tolerance=1e-6, print_every=100):
    """
    Optimize a high-dimensional function.
    
    Args:
        func: High-dimensional function to optimize
        optimizer_class: Optimizer class (Adam, SGDMomentum, etc.)
        optimizer_kwargs: Kwargs for optimizer (lr, etc.)
        max_iters: Maximum number of iterations
        tolerance: Convergence tolerance
        print_every: Print progress every N iterations
        
    Returns:
        Tuple (final_x, final_value, iterations)
    """
    # Initialize at random point within bounds
    lower, upper = func.get_bounds()
    x = np.random.uniform(lower, upper, func.dim)
    
    # Create optimizer
    optimizer = optimizer_class(**optimizer_kwargs)
    
    # Optimization loop
    history_values = []
    history_grad_norms = []
    
    for iteration in range(max_iters):
        # Compute function value and gradient
        value = func.compute(x)
        grad = func.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        history_values.append(value)
        history_grad_norms.append(grad_norm)
        
        # Print progress
        if (iteration + 1) % print_every == 0:
            print(f"  Iter {iteration + 1:4d}: f(x) = {value:12.6f}, ||grad|| = {grad_norm:10.6f}")
        
        # Check convergence
        if grad_norm < tolerance:
            print(f"  Converged at iteration {iteration + 1}: ||grad|| = {grad_norm:.2e}")
            break
        
        # Update parameters
        x = optimizer.step(x, grad)
    
    return x, value, iteration + 1, history_values, history_grad_norms


def main():
    parser = argparse.ArgumentParser(description='High-dimensional function optimization demo')
    parser.add_argument('--dim', type=int, default=10, help='Number of dimensions (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd_momentum'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--max-iters', type=int, default=1000, help='Maximum iterations (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create functions
    functions = [
        Sphere(dim=args.dim),
        Rastrigin(dim=args.dim),
        Ackley(dim=args.dim),
        Schwefel(dim=args.dim),
    ]
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer_class = Adam
        optimizer_kwargs = {'lr': args.lr}
        optimizer_name = f"Adam(lr={args.lr})"
    else:
        optimizer_class = SGDMomentum
        optimizer_kwargs = {'lr': args.lr, 'momentum': 0.9}
        optimizer_name = f"SGD+Momentum(lr={args.lr}, momentum=0.9)"
    
    print("=" * 80)
    print(f"High-Dimensional Function Optimization Demo")
    print("=" * 80)
    print(f"Dimensions: {args.dim}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()
    
    # Optimize each function
    results = []
    
    for func in functions:
        print(f"\n{'─' * 80}")
        print(f"Optimizing {func.name}")
        print(f"{'─' * 80}")
        
        # Get known optimum
        x_opt, f_opt = func.get_optimum()
        print(f"Known optimum: f(x*) = {f_opt}")
        
        # Optimize
        x_final, f_final, iters, hist_vals, hist_grads = optimize_function(
            func, optimizer_class, optimizer_kwargs, 
            max_iters=args.max_iters, print_every=100
        )
        
        # Compute distance to optimum
        dist_to_opt = np.linalg.norm(x_final - x_opt)
        value_error = abs(f_final - f_opt)
        
        print(f"\nResults:")
        print(f"  Final value: f(x) = {f_final:.6f}")
        print(f"  Distance to optimum: ||x - x*|| = {dist_to_opt:.6f}")
        print(f"  Value error: |f(x) - f(x*)| = {value_error:.6f}")
        print(f"  Iterations: {iters}")
        
        # Initial vs final comparison
        initial_value = hist_vals[0]
        improvement = initial_value - f_final
        improvement_pct = (improvement / abs(initial_value)) * 100 if initial_value != 0 else 0
        print(f"  Improvement: {initial_value:.6f} → {f_final:.6f} ({improvement_pct:.1f}%)")
        
        results.append({
            'function': func.name,
            'initial': initial_value,
            'final': f_final,
            'optimum': f_opt,
            'error': value_error,
            'iterations': iters,
            'converged': iters < args.max_iters
        })
    
    # Summary table
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"{'Function':<30} {'Initial':>12} {'Final':>12} {'Error':>12} {'Iters':>8} {'Conv':>6}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 6}")
    
    for result in results:
        conv_str = "X" if result['converged'] else "-"
        print(f"{result['function']:<30} {result['initial']:>12.6f} {result['final']:>12.6f} "
              f"{result['error']:>12.6f} {result['iterations']:>8d} {conv_str:>6}")
    
    print(f"{'=' * 80}")
    
    # Success metrics
    successful = sum(1 for r in results if r['error'] < 1.0)
    converged = sum(1 for r in results if r['converged'])
    
    print(f"\nSuccess rate: {successful}/{len(results)} functions reached error < 1.0")
    print(f"Convergence rate: {converged}/{len(results)} functions converged")
    
    # Recommendations
    print(f"\nObservations:")
    print(f"- Sphere (convex): Should converge quickly")
    print(f"- Rastrigin (multimodal): Challenging due to many local minima")
    print(f"- Ackley (nearly flat): Requires careful tuning")
    print(f"- Schwefel (deceptive): Global optimum far from most local minima")
    
    if args.optimizer == 'adam':
        print(f"\nTip: Adam typically works well on these functions.")
        print(f"     Try different learning rates: --lr 0.01, --lr 0.1, --lr 1.0")
    else:
        print(f"\nTip: SGD+Momentum may need smaller learning rates.")
        print(f"     Try: --lr 0.01 or --lr 0.001")


if __name__ == '__main__':
    main()
