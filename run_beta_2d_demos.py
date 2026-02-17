# Beta Sensitivity 2D Experiments Integration
# -----------------------------------------------------------------------------
# Quick script to run beta sensitivity analysis on 2D test functions
# Suitable for thesis trajectory visualizations and dynamics analysis

import subprocess
import sys
from pathlib import Path

def run_beta_sensitivity_2d_demos():
    """
    Run demo beta sensitivity experiments for visualization.
    These are fast (<5 minutes) and generate publication-quality plots.
    """
    print("="*80)
    print("BETA SENSITIVITY 2D VISUALIZATIONS")
    print("="*80)
    print("Running quick visualizations for thesis figures...")
    print()
    
    demos = [
        {
            'name': 'Momentum β Sweep on Rosenbrock',
            'args': [
                '--optimizer', 'Momentum',
                '--function', 'rosenbrock',
                '--beta-values', '0.5,0.7,0.9,0.95,0.99',
                '--max-iters', '300'
            ],
            'description': 'Shows β impact on trajectory smoothness and convergence speed'
        },
        {
            'name': 'Adam β1×β2 on Saddle Point',
            'args': [
                '--optimizer', 'Adam',
                '--function', 'saddle_point',
                '--beta1-values', '0.8,0.9',
                '--beta2-values', '0.9,0.99',
                '--max-iters', '200'
            ],
            'description': 'Demonstrates Adam\'s saddle point escape dynamics'
        }
    ]
    
    for demo in demos:
        print(f"\n{'='*80}")
        print(f"Running: {demo['name']}")
        print(f"Description: {demo['description']}")
        print(f"{'='*80}\n")
        
        cmd = [sys.executable, 'src/experiments/beta_sensitivity_2d.py'] + demo['args']
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n✅ {demo['name']} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {demo['name']} failed with error code {e.returncode}")
            print("Continuing with remaining demonstrations...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
    
    print("\n" + "="*80)
    print("BETA SENSITIVITY 2D DEMONSTRATIONS COMPLETE")
    print("="*80)
    print("\nGenerated visualizations can be found in:")
    print("  - results/beta_sensitivity_2d/rosenbrock/momentum/")
    print("  - results/beta_sensitivity_2d/saddle_point/adam/")
    print("\nThese plots are suitable for thesis inclusion.")
   print("="*80)

if __name__ == '__main__':
    run_beta_sensitivity_2d_demos()
