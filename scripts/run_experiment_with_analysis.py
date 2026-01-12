"""
Integrated Experiment Runner with Convergence Analysis and Provenance Tracking

This script demonstrates the full pipeline:
1. Run multi-seed experiments with dataset provenance logging
2. Compute empirical convergence rates
3. Compare to theoretical bounds
4. Generate high-quality reports and figures

Usage:
    python scripts/run_experiment_with_analysis.py --dataset MNIST --optimizers SGD Adam AdamW --seeds 42,123,456
"""
import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import pandas as pd
import numpy as np

from src.core.dataset_provenance import (
    get_dataset_provenance,
    create_experiment_manifest
)
from src.analysis.convergence_rate_analyzer import (
    compute_empirical_rate,
    compare_to_theoretical_bounds,
    generate_convergence_report,
    plot_convergence_comparison
)
from src.experiments.run_nn_experiment import build_model_and_data, build_optimizer
from src.core.training_utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_experiment(config, seed, output_dir):
    """Run a single experiment with full logging."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and data (support both 3-tuple and 4-tuple return shapes)
    model_and_data = build_model_and_data(
        dataset=config['dataset'],
        model_name=config['model'],
        batch_size=config.get('batch_size', 128),
        device=device,
        seed=seed
    )
    # Support (model, train_loader, test_loader) and (model, train_loader, val_loader, test_loader)
    if isinstance(model_and_data, tuple) and len(model_and_data) == 4:
        model, train_loader, val_loader, test_loader = model_and_data
    else:
        model, train_loader, test_loader = model_and_data  # type: ignore[assignment]
        val_loader = None

    # Build optimizer
    optimizer = build_optimizer(
        optimizer_name=config['optimizer'],
        model=model,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.0),
        momentum=config.get('momentum', 0.9)
    )

    # Training loop
    criterion = torch.nn.CrossEntropyLoss()
    loss_history = []

    epochs = config.get('epochs', 10)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx >= 100:  # Limit batches for speed
                break

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        logger.info(f"Seed {seed}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    # test_loader should always be provided; assert to narrow type for static analyzer
    assert test_loader is not None, "test_loader must be provided by build_model_and_data"
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total

    return {
        'seed': seed,
        'optimizer': config['optimizer'],
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'test_accuracy': accuracy,
        'config': config
    }


def main():
    parser = argparse.ArgumentParser(description='Run experiments with convergence analysis')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset (MNIST, CIFAR-10, CIFAR-100)')
    parser.add_argument('--model', type=str, default='SimpleMLP',
                        help='Model architecture')
    parser.add_argument('--optimizers', nargs='+', default=['SGD', 'Adam', 'AdamW'],
                        help='List of optimizers to test')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='results/convergence_analysis',
                        help='Output directory for results')

    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Optimizers: {args.optimizers}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)

    # Get dataset provenance
    provenance = get_dataset_provenance(
        dataset_name=args.dataset,
        split='train',
        data_root='./data'
    )

    logger.info("\nDATASET PROVENANCE:")
    for key, value in provenance.items():
        logger.info(f"  {key}: {value}")

    # Run experiments for each optimizer
    all_results = []
    convergence_results = {}

    for optimizer_name in args.optimizers:
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZER: {optimizer_name}")
        logger.info(f"{'='*80}")

        config = {
            'dataset': args.dataset,
            'model': args.model,
            'optimizer': optimizer_name,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': 128,
            'momentum': 0.9 if 'SGD' in optimizer_name.upper() else 0.0,
            'weight_decay': 0.0001
        }

        optimizer_results = []
        loss_histories = []

        for seed in seeds:
            logger.info(f"\n  Running seed {seed}...")
            result = run_single_experiment(config, seed, output_dir)
            optimizer_results.append(result)
            loss_histories.append(result['loss_history'])
            all_results.append(result)

        # Compute mean loss history across seeds
        min_length = min(len(h) for h in loss_histories)
        loss_histories_trimmed = [h[:min_length] for h in loss_histories]
        mean_loss_history = np.mean(loss_histories_trimmed, axis=0).tolist()

        # Analyze convergence
        logger.info(f"\n  Analyzing convergence rate for {optimizer_name}...")
        convergence_result = compute_empirical_rate(mean_loss_history, method='auto')

        if convergence_result.get('success'):
            best_fit = convergence_result.get('best_fit', 'power_law')
            r2 = convergence_result.get('best_r_squared', 0)
            logger.info(f"  Best fit: {best_fit} (R² = {r2:.4f})")

            if best_fit == 'power_law':
                alpha = convergence_result['power_law'].get('alpha', 0)
                logger.info(f"  Convergence rate α = {alpha:.4f}")

                # Compare to theory
                comparison = compare_to_theoretical_bounds(
                    empirical_rate=alpha,
                    optimizer_name=optimizer_name,
                    problem_type='nonconvex',  # Neural networks are nonconvex
                    lr=args.lr
                )
                logger.info(f"  Theoretical rate: {comparison['theoretical_exponent']:.4f}")
                logger.info(f"  Relative deviation: {comparison['relative_deviation']:.2%}")
                logger.info(f"  Within theory: {comparison['within_theory']}")
            else:
                beta = convergence_result['exponential'].get('beta', 0)
                logger.info(f"  Convergence rate β = {beta:.4f}")

        convergence_results[optimizer_name] = convergence_result

    # Generate comprehensive report
    logger.info(f"\n{'='*80}")
    logger.info("GENERATING REPORTS")
    logger.info(f"{'='*80}")

    # Save raw results
    results_df = pd.DataFrame([
        {
            'optimizer': r['optimizer'],
            'seed': r['seed'],
            'final_loss': r['final_loss'],
            'test_accuracy': r['test_accuracy']
        }
        for r in all_results
    ])
    results_path = output_dir / 'experiment_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"✓ Saved results to {results_path}")

    # Generate convergence report
    conv_report = generate_convergence_report(
        convergence_results,
        output_path=output_dir / 'convergence_rates.csv'
    )
    logger.info(f"✓ Saved convergence analysis to {output_dir / 'convergence_rates.csv'}")
    print("\nConvergence Rate Summary:")
    print(conv_report.to_string(index=False))

    # Plot convergence comparison
    plot_convergence_comparison(
        convergence_results,
        output_path=output_dir / 'convergence_comparison.png',
        title=f'Convergence Rate Comparison - {args.dataset}'
    )
    logger.info(f"✓ Saved convergence plot to {output_dir / 'convergence_comparison.png'}")

    # Create experiment manifest
    manifest_config = {
        'dataset': args.dataset,
        'model': args.model,
        'optimizers': args.optimizers,
        'seeds': seeds,
        'epochs': args.epochs,
        'lr': args.lr
    }
    create_experiment_manifest(
        experiment_name=f"{args.dataset}_{'-'.join(args.optimizers)}",
        config=manifest_config,
        dataset_provenance=provenance,
        output_path=output_dir / 'experiment_manifest.json'
    )
    logger.info(f"✓ Saved experiment manifest to {output_dir / 'experiment_manifest.json'}")

    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*80}")
    summary = results_df.groupby('optimizer').agg({
        'final_loss': ['mean', 'std'],
        'test_accuracy': ['mean', 'std']
    }).round(4)
    print(summary)

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"All results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
