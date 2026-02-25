"""
Generate Analysis Artifacts for Theory-Practice Validation

This script runs Hessian analysis and gradient noise analysis to generate
the artifacts required for scientific theory-practice validation.

Usage:
    # Generate artifacts for all experiments
    python scripts/generate_analysis_artifacts.py

    # Generate for specific experiment
    python scripts/generate_analysis_artifacts.py --experiment mnist

    # Generate only Hessian analysis
    python scripts/generate_analysis_artifacts.py --analysis hessian

    # Dry run (show what would be done)
    python scripts/generate_analysis_artifacts.py --dry-run

This ensures theory-practice validation uses MEASURED constants (L, sigma^2)
instead of hardcoded fallbacks.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def find_model_checkpoints(results_dir: Path, experiment: str):
    """Find trained model checkpoints for analysis."""
    checkpoint_dirs = [
        results_dir / experiment / 'checkpoints',
        results_dir / 'checkpoints',
        Path('artifacts') / 'checkpoints',
    ]

    checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints.extend(list(checkpoint_dir.glob('*.pt')))

    return checkpoints


def run_hessian_analysis(experiment: str, checkpoint: Path, output_dir: Path, dry_run: bool = False):
    """Run Hessian analysis on a checkpoint."""
    logging.info(f"  Running Hessian analysis on {checkpoint.name}...")

    if dry_run:
        logging.info(f"    [DRY RUN] Would analyze: {checkpoint}")
        return True

    # Check if Hessian analysis module exists
    hessian_module = Path('src/analysis/hessian_analysis.py')
    if not hessian_module.exists():
        logging.warning(f"    Hessian analysis module not found: {hessian_module}")
        return False

    try:
        # Run Hessian analysis (assuming it has a main entry point)
        cmd = [
            sys.executable,
            '-m', 'src.analysis.hessian_analysis',
            '--checkpoint', str(checkpoint),
            '--output-dir', str(output_dir / experiment / 'hessian_analysis')
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logging.info("    [OK] Hessian analysis completed")
            return True
        else:
            logging.warning(f"    [WARN] Hessian analysis failed: {result.stderr[:200]}")
            return False

    except FileNotFoundError:
        logging.warning("    [WARN] Python module execution failed (module may not have CLI)")
        return False
    except subprocess.TimeoutExpired:
        logging.warning("    [WARN] Hessian analysis timed out")
        return False
    except Exception as e:
        logging.warning(f"    [WARN] Hessian analysis error: {e}")
        return False


def run_gradient_noise_analysis(experiment: str, training_csv: Path, output_dir: Path, dry_run: bool = False):
    """Run gradient noise analysis on training results."""
    logging.info(f"  Running gradient noise analysis on {training_csv.name}...")

    if dry_run:
        logging.info(f"    [DRY RUN] Would analyze: {training_csv}")
        return True

    # Check if gradient noise analysis module exists
    noise_module = Path('src/analysis/gradient_noise_analysis.py')
    if not noise_module.exists():
        logging.warning(f"    Gradient noise module not found: {noise_module}")
        return False

    try:
        # Run gradient noise analysis
        cmd = [
            sys.executable,
            '-m', 'src.analysis.gradient_noise_analysis',
            '--training-csv', str(training_csv),
            '--output-dir', str(output_dir / experiment / 'gradient_noise')
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )

        if result.returncode == 0:
            logging.info("    [OK] Gradient noise analysis completed")
            return True
        else:
            logging.warning(f"    [WARN] Gradient noise analysis failed: {result.stderr[:200]}")
            return False

    except FileNotFoundError:
        logging.warning("    [WARN] Python module execution failed (module may not have CLI)")
        return False
    except subprocess.TimeoutExpired:
        logging.warning("    [WARN] Gradient noise analysis timed out")
        return False
    except Exception as e:
        logging.warning(f"    [WARN] Gradient noise analysis error: {e}")
        return False


def create_mock_artifacts(experiment: str, output_dir: Path):
    """Create mock artifacts for testing if analysis modules unavailable."""
    logging.info(f"  Creating mock artifacts for {experiment}...")

    # Create directories
    (output_dir / experiment / 'hessian_analysis').mkdir(parents=True, exist_ok=True)
    (output_dir / experiment / 'gradient_noise').mkdir(parents=True, exist_ok=True)

    # Mock Hessian artifact
    hessian_artifact = {
        'max_eigenvalue': 10.0,
        'min_eigenvalue': -0.01,
        'condition_number': 1000.0,
        'note': 'Mock artifact for testing - replace with real analysis'
    }

    hessian_file = output_dir / experiment / 'hessian_analysis' / f'{experiment}_mock_hessian.json'
    with open(hessian_file, 'w') as f:
        json.dump(hessian_artifact, f, indent=2)
    logging.info(f"    Created: {hessian_file}")

    # Mock gradient noise artifact
    noise_artifact = {
        'sigma_squared': 0.01,
        'gradient_variance': 0.01,
        'noise_to_signal_ratio': 0.1,
        'note': 'Mock artifact for testing - replace with real analysis'
    }

    noise_file = output_dir / experiment / 'gradient_noise' / f'{experiment}_mock_noise.json'
    with open(noise_file, 'w') as f:
        json.dump(noise_artifact, f, indent=2)
    logging.info(f"    Created: {noise_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate analysis artifacts for theory-practice validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Specific experiment to analyze (default: all experiments)'
    )
    parser.add_argument(
        '--analysis',
        choices=['hessian', 'gradient_noise', 'all'],
        default='all',
        help='Type of analysis to run (default: all)'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Results directory (default: results/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Output directory for artifacts (default: results/)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Create mock artifacts instead of running analysis'
    )

    args = parser.parse_args()

    print("="*80)
    print("ANALYSIS ARTIFACT GENERATION")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis types: {args.analysis}")
    if args.dry_run:
        print("Mode: DRY RUN (no execution)")
    if args.mock:
        print("Mode: MOCK (creating placeholder artifacts)")
    print()

    # Find experiments
    if args.experiment:
        experiments = [args.experiment]
    else:
        # Auto-detect experiment directories
        if args.results_dir.exists():
            experiments = [d.name for d in args.results_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')]
        else:
            logging.error(f"Results directory not found: {args.results_dir}")
            logging.info("Run experiments first, or specify --results-dir")
            return 1

    if not experiments:
        logging.warning("No experiments found to analyze")
        return 1

    print(f"Found {len(experiments)} experiment(s): {', '.join(experiments)}")
    print()

    # Process each experiment
    stats = {'hessian_success': 0, 'noise_success': 0, 'total': 0}

    for experiment in experiments:
        print(f"Processing experiment: {experiment}")
        print("-" * 80)

        if args.mock:
            # Create mock artifacts for testing
            create_mock_artifacts(experiment, args.output_dir)
            stats['hessian_success'] += 1
            stats['noise_success'] += 1
            stats['total'] += 1
            continue

        # Find checkpoints for Hessian analysis
        if args.analysis in ['hessian', 'all']:
            checkpoints = find_model_checkpoints(args.results_dir, experiment)

            if checkpoints:
                logging.info(f"Found {len(checkpoints)} checkpoint(s)")
                for checkpoint in checkpoints[:3]:  # Analyze first 3 checkpoints
                    success = run_hessian_analysis(
                        experiment, checkpoint, args.output_dir, args.dry_run
                    )
                    if success:
                        stats['hessian_success'] += 1
            else:
                logging.warning(f"No checkpoints found for {experiment}")

        # Find training CSVs for gradient noise analysis
        if args.analysis in ['gradient_noise', 'all']:
            exp_dir = args.results_dir / experiment
            if exp_dir.exists():
                csv_files = list(exp_dir.glob('*.csv'))

                if csv_files:
                    logging.info(f"Found {len(csv_files)} training CSV(s)")
                    for csv_file in csv_files[:3]:  # Analyze first 3 CSVs
                        success = run_gradient_noise_analysis(
                            experiment, csv_file, args.output_dir, args.dry_run
                        )
                        if success:
                            stats['noise_success'] += 1
                else:
                    logging.warning(f"No training CSVs found in {exp_dir}")

        stats['total'] += 1
        print()

    # Summary
    print("="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"Experiments processed: {stats['total']}")
    if args.analysis in ['hessian', 'all']:
        print(f"Hessian analyses: {stats['hessian_success']} succeeded")
    if args.analysis in ['gradient_noise', 'all']:
        print(f"Gradient noise analyses: {stats['noise_success']} succeeded")
    print()

    if args.dry_run:
        print("[OK] DRY RUN COMPLETE - No artifacts generated")
        print("  Remove --dry-run flag to execute")
    elif args.mock:
        print("[OK] MOCK ARTIFACTS CREATED")
        print("  Replace with real analysis when models are trained")
    elif stats['hessian_success'] > 0 or stats['noise_success'] > 0:
        print("[OK] ARTIFACTS GENERATED")
        print("  Theory-practice validation will now use measured constants")
        print()
        print("Next step: Run theory-practice validation")
        print("  python -m src.experiments.theory_practice_validation")
    else:
        print("[WARN] NO ARTIFACTS GENERATED")
        print("  Possible causes:")
        print("  - Analysis modules don't have CLI interface")
        print("  - No trained models/checkpoints available")
        print("  - Module execution failed")
        print()
        print("Solution: Use --mock flag to create placeholder artifacts")

    return 0


if __name__ == '__main__':
    sys.exit(main())
