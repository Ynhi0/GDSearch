"""
Demo script for parallel experiment execution.

This script demonstrates:
1. GPU detection
2. Parallel vs sequential execution
3. Checkpoint saving and resume
4. Result validation
"""

import torch
import sys
from pathlib import Path
import logging
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.parallel_experiment_runner import ParallelExperimentRunner
from src.core.optimizers import SGD, Adam

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def demo_gpu_detection():
    """Demo GPU detection."""
    print("\n" + "="*80)
    print("DEMO 1: GPU DETECTION")
    print("="*80)
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count >= 2:
        print("\n✅ Multi-GPU detected!")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute: {props.major}.{props.minor}")
        
        print(f"\n🚀 Parallel execution will use {gpu_count} GPUs")
    elif gpu_count == 1:
        print(f"\nSingle GPU detected: {torch.cuda.get_device_name(0)}")
        print("Sequential execution mode")
    else:
        print("\n⚠️  No GPU detected - CPU mode")
    
    return gpu_count


def demo_parallel_runner_initialization():
    """Demo parallel runner initialization."""
    print("\n" + "="*80)
    print("DEMO 2: PARALLEL RUNNER INITIALIZATION")
    print("="*80)
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count >= 2:
        print(f"Initializing ParallelExperimentRunner with {gpu_count} GPUs...")
        runner = ParallelExperimentRunner(num_gpus=gpu_count)
        
        print(f"✅ Runner initialized:")
        print(f"  GPUs: {runner.num_gpus}")
        print(f"  Results dir: {runner.results_dir}")
        
        return runner
    else:
        print("⚠️  Need 2+ GPUs for parallel execution")
        return None


def demo_experiment_config():
    """Demo experiment configuration."""
    print("\n" + "="*80)
    print("DEMO 3: EXPERIMENT CONFIGURATION")
    print("="*80)
    
    # Create sample experiment configs
    experiments = []
    
    for seed in [42, 123, 456]:
        for optimizer_name in ['SGD', 'Adam']:
            exp = {
                'name': f'mnist_{optimizer_name}_seed{seed}',
                'experiment_name': 'demo',
                'dataset': 'MNIST',
                'model': 'SimpleMLP',
                'optimizer': optimizer_name,
                'lr': 0.01 if optimizer_name == 'SGD' else 0.001,
                'epochs': 2,  # Quick demo
                'batch_size': 128,
                'seed': seed,
                'device': 'cuda:0'  # Will be overridden by parallel runner
            }
            experiments.append(exp)
    
    print(f"Created {len(experiments)} experiment configurations:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    
    return experiments


def demo_timing_comparison():
    """Demo timing comparison between sequential and parallel."""
    print("\n" + "="*80)
    print("DEMO 4: TIMING COMPARISON")
    print("="*80)
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count >= 2:
        # Estimate times
        experiments_per_seed = 2  # SGD + Adam
        seeds = 3
        total_experiments = experiments_per_seed * seeds
        time_per_experiment = 30  # seconds (estimate for 2 epochs)
        
        sequential_time = total_experiments * time_per_experiment
        parallel_time = (total_experiments / gpu_count) * time_per_experiment
        
        print(f"Configuration:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Time per experiment: {time_per_experiment}s")
        
        print(f"\nSequential execution (1 GPU):")
        print(f"  Time: {sequential_time}s ({sequential_time/60:.1f} minutes)")
        
        print(f"\nParallel execution ({gpu_count} GPUs):")
        print(f"  Time: {parallel_time}s ({parallel_time/60:.1f} minutes)")
        print(f"  Speedup: {sequential_time/parallel_time:.2f}x")
        
        print(f"\n⏱️  Time saved: {sequential_time - parallel_time}s ({(sequential_time - parallel_time)/60:.1f} minutes)")
    else:
        print("⚠️  Need 2+ GPUs for parallel execution demo")


def demo_quick_run():
    """Run a quick demo experiment."""
    print("\n" + "="*80)
    print("DEMO 5: QUICK RUN (Optional - Press Enter to skip)")
    print("="*80)
    
    response = input("Run a quick 2-epoch demo experiment? (y/N): ")
    
    if response.lower() != 'y':
        print("Skipping quick run demo")
        return
    
    print("\nRunning quick demo...")
    print("This will run 2 epochs of MNIST with SGD")
    
    # Import here to avoid slow startup
    from src.experiments.run_nn_experiment import run_experiment
    
    config = {
        'experiment_name': 'demo',
        'dataset': 'MNIST',
        'model': 'SimpleMLP',
        'optimizer': 'SGD',
        'lr': 0.01,
        'epochs': 2,
        'batch_size': 128,
        'seed': 42,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    
    results_dir = Path('results/demo')
    
    start_time = time.time()
    result = run_experiment(config, device=config['device'], results_dir=results_dir)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Demo completed in {elapsed:.1f}s")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Final test accuracy: {result.get('final_test_acc', 'N/A')}")
        print(f"Final train loss: {result.get('final_train_loss', 'N/A')}")
        print(f"Result file: {result.get('result_file', 'N/A')}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("PARALLEL EXECUTION DEMO")
    print("GDSearch Codebase - February 2026")
    print("="*80)
    
    # Run demos
    gpu_count = demo_gpu_detection()
    runner = demo_parallel_runner_initialization()
    experiments = demo_experiment_config()
    demo_timing_comparison()
    demo_quick_run()
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    
    if gpu_count >= 2:
        print("\n✅ Your system is ready for parallel execution!")
        print("\nTo run experiments in parallel:")
        print("  python run_all_kaggle.py --experiments mnist --seeds 42,123,456 --parallel")
    else:
        print("\nℹ️  Your system will use sequential execution")
        print("  For parallel execution, use a system with 2+ GPUs (e.g., Kaggle T4x2)")
    
    print("\nNext steps:")
    print("  1. Run comprehensive tests: pytest tests/test_bug_fixes_comprehensive.py")
    print("  2. Try a full experiment: python run_all_kaggle.py --experiments mnist --seeds 42")
    print("  3. Check the Kaggle notebook for integrated workflow")


if __name__ == '__main__':
    main()
