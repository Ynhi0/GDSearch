"""
Main script to run optimization algorithm comparison experiments.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import logging
logging.basicConfig(level=logging.INFO)

from src.core.test_functions import Rosenbrock, IllConditionedQuadratic, SaddlePoint, Ackley2D
from src.core.optimizers import SGD, SGDMomentum, SGDNesterov, RMSProp, Adam, AdamW, AMSGrad


def run_single_experiment(optimizer_config, function_config, initial_point, num_iterations, seed):
    """
    Run a single experiment with specified configuration.

    Args:
        optimizer_config: Dictionary configuring optimizer
            {'type': 'SGD'|'SGDMomentum'|'RMSProp'|'Adam', 'params': {...}}
        function_config: Dictionary configuring test function
            {'type': 'Rosenbrock'|'IllConditionedQuadratic'|'SaddlePoint', 'params': {...}}
        initial_point: Tuple (x0, y0) - starting point
        num_iterations: Number of iterations
        seed: Seed for random number generator

    Returns:
        DataFrame containing optimization process history
    """
    # Set seed to ensure reproducibility across libraries
    from src.core.training_utils import set_seed
    set_seed(seed)

    # Initialize test function
    func_type = function_config['type']
    func_params = function_config.get('params', {})

    if func_type == 'Rosenbrock':
        test_function = Rosenbrock(**func_params)
    elif func_type == 'IllConditionedQuadratic':
        test_function = IllConditionedQuadratic(**func_params)
    elif func_type == 'SaddlePoint':
        test_function = SaddlePoint(**func_params)
    elif func_type == 'Ackley':
        test_function = Ackley2D(**func_params)
    else:
        raise ValueError(f"Invalid test function type: {func_type}")

    # Initialize optimizer
    opt_type = optimizer_config['type']
    opt_params = optimizer_config.get('params', {})

    # Use factory to support aliases and canonical names
    from src.core.optimizers import create_optimizer_instance

    optimizer = create_optimizer_instance(opt_type, **opt_params)

    # Reset optimizer to initial state
    optimizer.reset()

    # Initialize parameters
    current_x, current_y = initial_point

    # List to store history
    history = []

    # Start timing and track GPU (if available)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Optimization loop
    for i in range(num_iterations):
        # Calculate function value and gradient
        loss = test_function.compute(current_x, current_y)
        grad_x, grad_y = test_function.gradient(current_x, current_y)

        # Calculate gradient norm (use hypot to avoid overflow and handle large values safely)
        import math
        try:
            grad_norm = float(math.hypot(float(grad_x), float(grad_y)))
        except Exception as e:
            # Robust fallback and logging
            grad_norm = float('inf')
            logging.warning("Gradient norm computation overflowed or invalid: grad_x=%s, grad_y=%s", grad_x, grad_y, exc_info=True)

        # Compute Hessian eigenvalues for curvature analysis
        hessian = test_function.hessian(current_x, current_y)
        eigenvalues = np.linalg.eigvalsh(hessian)  # Returns sorted eigenvalues
        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[1]
        condition_number = abs(lambda_max / lambda_min) if abs(lambda_min) > 1e-10 else np.inf

        # Perform update step
        new_x, new_y = optimizer.step((current_x, current_y), (grad_x, grad_y))

        # Guard against non-finite or explosively large updates
        try:
            if (not np.isfinite(new_x)) or (not np.isfinite(new_y)) or abs(new_x) > 1e12 or abs(new_y) > 1e12:
                logging.warning("Optimizer produced invalid update: new=(%s,%s), skipping update and keeping previous parameters.", new_x, new_y)
                new_x, new_y = current_x, current_y
        except Exception as e:
            # Log the unexpected exception with traceback; ensure we fail safe by reverting parameters
            logging.error("Error while validating optimizer update; reverting to previous parameters.", exc_info=True)
            new_x, new_y = current_x, current_y

        # Calculate update norm (use hypot for robustness)
        try:
            update_norm = float(math.hypot(float(new_x - current_x), float(new_y - current_y)))
        except Exception as e:
            update_norm = float('inf')
            logging.warning("Update norm computation overflowed or invalid: new=(%s,%s), old=(%s,%s)", new_x, new_y, current_x, current_y, exc_info=True)

        # Save information to history (including Hessian eigenvalues)
        history.append({
            'iteration': i,
            'x': current_x,
            'y': current_y,
            'loss': loss,
            'grad_norm': grad_norm,
            'update_norm': update_norm,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'condition_number': condition_number,
            # Metadata for reproducibility/aggregation
            'seed': seed,
            'optimizer': getattr(optimizer, 'name', str(opt_type))
        })

        # Update parameters
        current_x, current_y = new_x, new_y

    # Convert history to DataFrame
    df = pd.DataFrame(history)

    # Add dynamics metrics (step sizes, angles, oscillations)
    try:
        from src.analysis.dynamics import add_dynamics_metrics
        df, dynamics_summary = add_dynamics_metrics(df, x_col='x', y_col='y')
        # Optionally log or attach summary
        logging.info("Dynamics summary: %s", dynamics_summary)
    except Exception as e:
        logging.debug("dynamics analysis unavailable: %s", e, exc_info=True)

    # End timing and record GPU statistics
    elapsed_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None

    # Add timing and memory info to DataFrame (constant for all rows)
    df['elapsed_time'] = elapsed_time
    df['peak_memory_MB'] = peak_memory

    # Compute PL mu estimates for points where applicable
    try:
        from src.analysis.pl_condition import pl_mu_estimate
        df['pl_mu_hat'] = df.apply(lambda r: pl_mu_estimate(r['loss'], (r['grad_norm'] ** 2), f_star=0.0), axis=1)
    except Exception as e:
        logging.debug("PL condition helper unavailable: %s", e, exc_info=True)

    return df


def create_experiment_configs():
    """
    Create list of experiment configurations according to Design Matrix.

    Returns:
        List of experiment configuration dictionaries
    """
    configs = []

    # Starting points for functions
    initial_rosenbrock = (-1.5, 2.0)
    initial_quad = (1.0, 1.0)
    initial_saddle = (0.5, 0.5)

    # ========== SGD Momentum on Rosenbrock ==========
    # SGDM-R-1: beta=0.5
    configs.append({
        'experiment_id': 'SGDM-R-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.5}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # SGDM-R-2: beta=0.9
    configs.append({
        'experiment_id': 'SGDM-R-2',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # SGDM-R-3: beta=0.99
    configs.append({
        'experiment_id': 'SGDM-R-3',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.99}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== Adam on Rosenbrock ==========
    # ADAM-R-1: beta1=0.9, beta2=0.999 (default)
    configs.append({
        'experiment_id': 'ADAM-R-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ADAM-R-2: beta1=0.5, beta2=0.999
    configs.append({
        'experiment_id': 'ADAM-R-2',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.5, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ADAM-R-3: beta1=0.9, beta2=0.9
    configs.append({
        'experiment_id': 'ADAM-R-3',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ADAM-R-4: beta1=0.5, beta2=0.9
    configs.append({
        'experiment_id': 'ADAM-R-4',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.5, 'beta2': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== Nesterov on Rosenbrock ==========
    configs.append({
        'experiment_id': 'NAG-R-1',
        'optimizer_config': {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })
    configs.append({
        'experiment_id': 'NAG-R-2',
        'optimizer_config': {'type': 'SGDNesterov', 'params': {'lr': 0.01, 'beta': 0.5}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== AdamW on Rosenbrock (compare weight decay) ==========
    for wd, exp_id in [(0.0, 'ADAMW-R-0'), (0.01, 'ADAMW-R-1'), (0.05, 'ADAMW-R-5')]:
        configs.append({
            'experiment_id': exp_id,
            'optimizer_config': {'type': 'AdamW', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'weight_decay': wd}},
            'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
            'initial_point': initial_rosenbrock,
            'num_iterations': 10000,
            'seed': 42
        })

    # ========== AMSGrad on Rosenbrock ==========
    configs.append({
        'experiment_id': 'AMSG-R-1',
        'optimizer_config': {'type': 'AMSGrad', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== SGD on other functions ==========
    # SGD on Rosenbrock
    configs.append({
        'experiment_id': 'SGD-R-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.001}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # SGD on IllConditionedQuadratic
    configs.append({
        'experiment_id': 'SGD-Q-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.001}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })

    # SGD on SaddlePoint
    configs.append({
        'experiment_id': 'SGD-S-1',
        'optimizer_config': {'type': 'SGD', 'params': {'lr': 0.01}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== RMSProp on functions ==========
    # RMSProp on Rosenbrock
    configs.append({
        'experiment_id': 'RMS-R-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'Rosenbrock', 'params': {'a': 1, 'b': 100}},
        'initial_point': initial_rosenbrock,
        'num_iterations': 10000,
        'seed': 42
    })

    # RMSProp on IllConditionedQuadratic
    configs.append({
        'experiment_id': 'RMS-Q-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })

    # RMSProp on SaddlePoint
    configs.append({
        'experiment_id': 'RMS-S-1',
        'optimizer_config': {'type': 'RMSProp', 'params': {'lr': 0.01, 'decay_rate': 0.9, 'epsilon': 1e-8}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })

    # ========== Add experiments on other functions ==========
    # SGDMomentum on IllConditionedQuadratic
    configs.append({
        'experiment_id': 'SGDM-Q-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })

    # SGDMomentum on SaddlePoint
    configs.append({
        'experiment_id': 'SGDM-S-1',
        'optimizer_config': {'type': 'SGDMomentum', 'params': {'lr': 0.01, 'beta': 0.9}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })

    # Adam on IllConditionedQuadratic
    configs.append({
        'experiment_id': 'ADAM-Q-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'IllConditionedQuadratic', 'params': {'kappa': 100}},
        'initial_point': initial_quad,
        'num_iterations': 10000,
        'seed': 42
    })

    # Adam on SaddlePoint
    configs.append({
        'experiment_id': 'ADAM-S-1',
        'optimizer_config': {'type': 'Adam', 'params': {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}},
        'function_config': {'type': 'SaddlePoint', 'params': {}},
        'initial_point': initial_saddle,
        'num_iterations': 10000,
        'seed': 42
    })

    return configs


def generate_filename(config):
    """
    Generate unique filename for experiment results.

    Args:
        config: Full experiment configuration dictionary

    Returns:
        Filename (string)
    """
    # Use experiment_id if available, otherwise create from parameters
    if 'experiment_id' in config:
        exp_id = config['experiment_id']
        filename = f"{exp_id}.csv"
    else:
        # Fallback for experiments without ID
        opt_type = config['optimizer_config']['type']
        func_type = config['function_config']['type']
        seed = config['seed']
        filename = f"{opt_type}_{func_type}_seed{seed}.csv"

    return filename


def main():
    """Main function to run all experiments."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create list of experiment configurations
    configs = create_experiment_configs()

    print(f"Total experiments: {len(configs)}")
    print("Starting experiments...\n")

    # Run all experiments
    for config in tqdm(configs, desc="Running experiments"):
        # Run experiment
        df = run_single_experiment(
            optimizer_config=config['optimizer_config'],
            function_config=config['function_config'],
            initial_point=config['initial_point'],
            num_iterations=config['num_iterations'],
            seed=config['seed']
        )

        # Generate filename
        filename = generate_filename(config)

        # Save results
        filepath = os.path.join('results', filename)
        df.to_csv(filepath, index=False)

        # Add experiment_id to metadata if available
        if 'experiment_id' in config:
            # Can add experiment_id to DataFrame if needed
            pass

    print("\nCompleted all experiments!")
    print("Results saved in 'results/' directory")
    print(f"Total files: {len(configs)}")


if __name__ == '__main__':
    main()
