#!/usr/bin/env python3
"""
Comprehensive reproducibility script for GDSearch project.
Runs the entire pipeline: 2D experiments → NN tuning → summaries → reports.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-2d          # Skip 2D experiments
    python run_all.py --skip-tuning      # Skip NN hyperparameter tuning
    python run_all.py --quick            # Quick mode: reduced iterations
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path


def log(msg: str):
    """Print timestamped log message."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def run_command(cmd: str, description: str, check=True):
    """Run a shell command with logging."""
    log(f"Starting: {description}")
    log(f"Command: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, check=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"⚠️  Warning: {description} failed with exit code {result.returncode}")
        if check:
            log("Stopping pipeline due to error")
            sys.exit(1)
    else:
        log(f"✅ Completed: {description} (took {elapsed:.1f}s)")
    return result.returncode


def run_python_script(script: str, description: str, check=True):
    """Run a Python script."""
    return run_command(f"python {script}", description, check=check)


def ensure_directories():
    """Create necessary directories."""
    log("Creating necessary directories...")
    for d in ['results', 'plots', 'configs']:
        Path(d).mkdir(exist_ok=True)
    log("✅ Directories ready")


def run_2d_experiments(quick=False):
    """Run 2D test function experiments."""
    log("=" * 60)
    log("PHASE 1: 2D Test Function Experiments")
    log("=" * 60)
    
    # Check if src/experiments/run_experiment.py exists
    if not Path("src/experiments/run_experiment.py").exists():
        log("⚠️  src/experiments/run_experiment.py not found, skipping 2D experiments")
        return
    
    run_python_script("src/experiments/run_experiment.py", "2D baseline experiments")
    
    # Generate advanced visualizations
    if Path("generate_advanced_plots.py").exists():
        run_python_script("generate_advanced_plots.py", 
                         "Advanced 2D visualizations (grids, dynamics, 3D)", 
                         check=False)


def run_nn_tuning(quick=False):
    """Run neural network hyperparameter tuning."""
    log("=" * 60)
    log("PHASE 2: Neural Network Hyperparameter Tuning")
    log("=" * 60)
    
    # Check for tuning script
    if Path("scripts/tune_nn.py").exists():
        run_python_script("scripts/tune_nn.py", "NN hyperparameter tuning (2-stage sweeps + final runs)")
    else:
        log("⚠️  tune_nn.py not found, trying nn_workflow.py")
        if Path("scripts/nn_workflow.py").exists():
            run_python_script("scripts/nn_workflow.py", "NN workflow (demo runs)")
        else:
            log("⚠️  No NN training scripts found, skipping")


def run_cifar10_experiments(quick=False):
    """Run CIFAR-10 experiments."""
    log("=" * 60)
    log("PHASE 3: CIFAR-10 Experiments")
    cmd = "python src/experiments/run_cifar10.py"
    if quick:
        cmd += " --quick"
    run_python_script(cmd, "CIFAR-10 Experiments")


def run_loss_landscape():
    """Generate loss landscape visualizations."""
    log("=" * 60)
    log("PHASE 4: Loss Landscape Analysis")
    log("=" * 60)
    
    # No standalone script provided; provide guidance and skip.
    if Path("src/visualization/loss_landscape.py").exists():
        log("ℹ️  Loss landscape utilities available in src/visualization/loss_landscape.py (call from notebooks or custom script). Skipping.")
    else:
        log("⚠️  src/visualization/loss_landscape.py not found, skipping")


def generate_summaries():
    """Generate quantitative and qualitative summary tables."""
    log("=" * 60)
    log("PHASE 5: Summary Generation")
    log("=" * 60)
    
    if Path("scripts/generate_summaries.py").exists():
        run_python_script("scripts/generate_summaries.py", 
                         "Quantitative & qualitative summaries + plots")
    else:
        log("⚠️  scripts/generate_summaries.py not found, skipping")


def run_statistical_reports():
    """Generate statistical reports for MNIST and CIFAR-10 if results exist."""
    log("=" * 60)
    log("PHASE 5.1: Statistical Reports")
    if Path("scripts/generate_statistical_report.py").exists():
        run_python_script("scripts/generate_statistical_report.py", "MNIST statistical report", check=False)
    if Path("scripts/generate_cifar10_statistical_report.py").exists():
        run_python_script("scripts/generate_cifar10_statistical_report.py", "CIFAR-10 statistical report", check=False)


def compute_tradeoffs():
    """Compute Accuracy vs Time/Memory trade-offs and save plots."""
    log("=" * 60)
    log("PHASE 5.2: Trade-off Analysis")
    if Path("scripts/compute_tradeoffs.py").exists():
        run_python_script("scripts/compute_tradeoffs.py", "Trade-offs (Accuracy vs Time/Memory)", check=False)
    else:
        log("⚠️  scripts/compute_tradeoffs.py not found, skipping")


def list_outputs():
    """List generated outputs."""
    log("=" * 60)
    log("PHASE 6: Output Summary")
    log("=" * 60)
    
    results_count = len(list(Path("results").glob("*.csv"))) if Path("results").exists() else 0
    plots_count = len(list(Path("plots").glob("*.png"))) if Path("plots").exists() else 0
    
    log(f"📊 Generated {results_count} CSV files in results/")
    log(f"📈 Generated {plots_count} PNG plots in plots/")
    
    # List key artifacts
    key_files = [
        "results/summary_quantitative.csv",
        "results/summary_qualitative.csv",
        "REPORT.md",
        "hypothesis_matrix.md"
    ]
    
    log("\n🎯 Key Artifacts:")
    for f in key_files:
        status = "✅" if Path(f).exists() else "❌"
        log(f"  {status} {f}")


def main():
    parser = argparse.ArgumentParser(description="Run complete GDSearch pipeline")
    parser.add_argument("--skip-2d", action="store_true", 
                       help="Skip 2D test function experiments")
    parser.add_argument("--skip-tuning", action="store_true", 
                       help="Skip NN hyperparameter tuning")
    parser.add_argument("--skip-landscape", action="store_true",
                       help="Skip loss landscape analysis")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode with reduced iterations")
    parser.add_argument("--summaries-only", action="store_true",
                       help="Only regenerate summaries from existing results")
    
    args = parser.parse_args()
    
    start_time = time.time()
    log("🚀 Starting GDSearch Complete Pipeline")
    log(f"Working directory: {os.getcwd()}")
    
    # Ensure directories exist
    ensure_directories()
    
    if args.summaries_only:
        log("Running in summaries-only mode")
        generate_summaries()
        list_outputs()
        return
    
    # Phase 1: 2D experiments
    if not args.skip_2d:
        run_2d_experiments(quick=args.quick)
    else:
        log("⏭️  Skipping 2D experiments (--skip-2d)")
    
    # Phase 2: NN tuning
    if not args.skip_tuning:
        run_nn_tuning(quick=args.quick)
    else:
        log("⏭️  Skipping NN tuning (--skip-tuning)")
    
    # Phase 3: CIFAR-10 experiments
    if not args.skip_tuning:
        run_cifar10_experiments(quick=args.quick)
    else:
        log("⏭️  Skipping CIFAR-10 experiments (--skip-tuning)")
    
    # Phase 4: Loss landscape
    if not args.skip_landscape:
        run_loss_landscape()
    else:
        log("⏭️  Skipping loss landscape (--skip-landscape)")
    
    # Phase 5: Summaries and analyses
    generate_summaries()
    run_statistical_reports()
    compute_tradeoffs()
    
    # Phase 6: Report outputs
    list_outputs()
    
    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"✅ Pipeline completed successfully in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log("=" * 60)
    log("\n📖 Next steps:")
    log("  1. Review summary tables: results/summary_*.csv")
    log("  2. Check visualizations: plots/")
    log("  3. Read synthesis report: REPORT.md")
    log("  4. Explore hypothesis matrix: hypothesis_matrix.md")


if __name__ == "__main__":
    main()
