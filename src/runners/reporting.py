"""
Reporting Module for GDSearch.

Handles result aggregation, report generation, and visualization.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np


def generate_experiment_summary(results_dir: Path, experiment_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive summary report for all experiments.
    
    Args:
        results_dir: Directory containing experiment results
        experiment_results: Dictionary mapping experiment names to results
    
    Returns:
        Path to generated report file
    """
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    report_path = reports_dir / "experiment_summary_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# GDSearch Benchmark Suite - Comprehensive Experiment Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Experiments completed
        f.write("## Experiments Completed\n\n")
        for exp_name, exp_df in experiment_results.items():
            if exp_df is not None:
                n_points = len(exp_df) if hasattr(exp_df, '__len__') else 'N/A'
                f.write(f"- **{exp_name.upper()}**: {n_points} data points\n")
        
        # Directory structure
        f.write("\n## Results Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{results_dir.name}/\n")
        f.write("|-- experiments/           # Experiment-specific results\n")
        f.write("|   |-- mnist/             # MNIST classification results\n")
        f.write("|   |-- cifar10/           # CIFAR-10 image classification\n")
        f.write("|   |-- nlp/               # NLP sentiment analysis\n")
        f.write("|   `-- medical/           # Medical image segmentation\n")
        f.write("|-- visualizations/        # Interactive HTML plots\n")
        f.write("|   `-- *.html             # Open in browser for interactive charts\n")
        f.write("|-- analysis/              # Statistical & convergence analysis\n")
        f.write("|   |-- convergence_rates.csv\n")
        f.write("|   |-- cross_experiment_statistics.csv\n")
        f.write("|   `-- aggregated_optimizer_performance.csv\n")
        f.write("|-- reports/               # Summary reports\n")
        f.write("|   `-- experiment_summary_report.md  # This file\n")
        f.write("`-- checkpoints/           # Model checkpoints (if enabled)\n")
        f.write("```\n\n")
        
        # Statistical summary
        f.write("## Statistical Analysis Summary\n\n")
        
        # Check for cross-experiment statistics
        stats_path = results_dir / "analysis" / "cross_experiment_statistics.csv"
        if stats_path.exists():
            try:
                stats_df = pd.read_csv(stats_path)
                f.write("### Pairwise Optimizer Comparisons\n\n")
                f.write("| Comparison | p-value | p-adj | Cohen's d | CI Lower | CI Upper | Interpretation |\n")
                f.write("|------------|---------|-------|-----------|----------|----------|----------------|\n")
                
                for _, row in stats_df.iterrows():
                    comp = f"{row['optimizer_a']} vs {row['optimizer_b']}"
                    
                    # Extract scalar values from row - handle NA/None gracefully
                    try:
                        p_val_raw = row['p_value']
                        p_val = f"{float(p_val_raw):.4f}" if (p_val_raw is not None and not np.isnan(float(p_val_raw))) else 'N/A'
                    except (ValueError, TypeError, KeyError):
                        p_val = 'N/A'
                    
                    # Check if column exists and value is not NA
                    if 'p_value_adjusted' in stats_df.columns:
                        try:
                            p_adj_raw = row['p_value_adjusted']
                            p_adj = f"{float(p_adj_raw):.4f}" if (p_adj_raw is not None and not np.isnan(float(p_adj_raw))) else 'N/A'
                        except (ValueError, TypeError):
                            p_adj = 'N/A'
                    else:
                        p_adj = 'N/A'
                    
                    try:
                        d_raw = row['cohens_d']
                        d = f"{float(d_raw):.3f}" if (d_raw is not None and not np.isnan(float(d_raw))) else 'N/A'
                    except (ValueError, TypeError, KeyError):
                        d = 'N/A'
                    
                    if 'cohens_d_ci_lower' in stats_df.columns:
                        try:
                            ci_low_raw = row['cohens_d_ci_lower']
                            ci_low = f"{float(ci_low_raw):.3f}" if (ci_low_raw is not None and not np.isnan(float(ci_low_raw))) else 'N/A'
                        except (ValueError, TypeError):
                            ci_low = 'N/A'
                    else:
                        ci_low = 'N/A'
                    
                    if 'cohens_d_ci_upper' in stats_df.columns:
                        try:
                            ci_high_raw = row['cohens_d_ci_upper']
                            ci_high = f"{float(ci_high_raw):.3f}" if (ci_high_raw is not None and not np.isnan(float(ci_high_raw))) else 'N/A'
                        except (ValueError, TypeError):
                            ci_high = 'N/A'
                    else:
                        ci_high = 'N/A'
                    
                    interp = str(row.get('effect_interpretation', 'N/A'))
                    sig = '*' if bool(row.get('significant', False)) else ''
                    
                    f.write(f"| {comp} | {p_val} | {p_adj}{sig} | {d} | {ci_low} | {ci_high} | {interp} |\n")
                
                f.write("\n*Asterisk (*) indicates statistical significance after FDR correction (Î±=0.05)\n\n")
            except Exception as e:
                f.write(f"Could not load statistics: {e}\n\n")
        
        # Best performers
        f.write("## Best Performing Optimizers\n\n")
        
        agg_path = results_dir / "analysis" / "aggregated_optimizer_performance.csv"
        if agg_path.exists():
            try:
                agg_df = pd.read_csv(agg_path)
                if 'avg_accuracy_across_experiments' in agg_df.columns:
                    best_optimizers = agg_df.nlargest(5, 'avg_accuracy_across_experiments')
                    
                    f.write("| Rank | Optimizer | Avg Accuracy | Std Dev | # Experiments |\n")
                    f.write("|------|-----------|--------------|---------|---------------|\n")
                    
                    for rank, (_, row) in enumerate(best_optimizers.iterrows(), 1):
                        opt = str(row['optimizer'])
                        
                        try:
                            acc_raw = row['avg_accuracy_across_experiments']
                            acc = f"{float(acc_raw):.2f}%" if (acc_raw is not None and not np.isnan(float(acc_raw))) else 'N/A'
                        except (ValueError, TypeError, KeyError):
                            acc = 'N/A'
                        
                        if 'std_accuracy_across_experiments' in agg_df.columns:
                            try:
                                std_raw = row['std_accuracy_across_experiments']
                                std = f"{float(std_raw):.2f}" if (std_raw is not None and not np.isnan(float(std_raw))) else 'N/A'
                            except (ValueError, TypeError):
                                std = 'N/A'
                        else:
                            std = 'N/A'
                        
                        try:
                            n_exp_raw = row.get('experiments_count', 0)
                            n_exp = int(float(n_exp_raw)) if (n_exp_raw is not None and not np.isnan(float(n_exp_raw))) else 0
                        except (ValueError, TypeError):
                            n_exp = 0
                        
                        f.write(f"| {rank} | {opt} | {acc} | {std} | {n_exp} |\n")
                    
                    f.write("\n")
            except Exception as e:
                f.write(f"Could not load aggregated results: {e}\n\n")
        
        # Reproducibility information
        f.write("## Reproducibility Information\n\n")
        f.write("### Random Seeds\n")
        f.write("All experiments use explicit random seeds for reproducibility.\n\n")
        
        f.write("### Dependencies\n")
        f.write("- PyTorch\n")
        f.write("- NumPy\n")
        f.write("- SciPy (for statistical tests)\n")
        f.write("- Optional: MLflow (experiment tracking), medmnist/MONAI (medical experiments)\n\n")
        
        f.write("### Data Splits\n")
        f.write("- Train/Val/Test splits use deterministic seeds\n")
        f.write("- No data leakage between splits\n")
        f.write("- Test set used only for final evaluation\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write("*Report generated by GDSearch Benchmark Suite*\n")
    
    logging.info(f"Summary report generated: {report_path}")
    return str(report_path)


def create_results_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save CSV file
    """
    if not results:
        logging.warning("No results to save")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")


def print_experiment_summary(results: Dict[str, Any], experiment_name: str) -> None:
    """
    Print experiment results to console.
    
    Args:
        results: Results dictionary
        experiment_name: Name of the experiment
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name.upper()}")
    print(f"{'='*80}")
    
    if 'test_accuracy' in results:
        print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    
    if 'test_loss' in results:
        print(f"Test Loss: {results['test_loss']:.4f}")
    
    if 'epochs_trained' in results:
        print(f"Epochs Trained: {results['epochs_trained']}")
    
    if 'best_epoch' in results:
        print(f"Best Epoch: {results['best_epoch']}")
    
    print(f"{'='*80}\n")


def visualize_training_history(history: Dict[str, List], output_path: Path) -> None:
    """
    Create and save training history plot.
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy')
        if 'val_accuracy' in history and history['val_accuracy']:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training history plot saved to {output_path}")
    except Exception as e:
        logging.warning(f"Could not create training history plot: {e}")
