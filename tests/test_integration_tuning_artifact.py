import os
from pathlib import Path
import run_all_kaggle as rag


def test_end_to_end_tuning_and_artifact(tmp_path):
    # CI-safe quick integration test: run minimal tuning + one optimizer
    os.environ['GDSEARCH_TUNE_EVAL_ALL_CANDIDATES'] = 'false'
    os.environ['GDSEARCH_TUNE_SEED_COUNT'] = '1'
    os.environ['GDSEARCH_TUNE_TOPK'] = '1'
    os.environ['GDSEARCH_ULTRA_QUICK_LIMIT'] = '1'  # limit to first optimizer for speed

    # Ensure ULTRA_QUICK_MODE for fast runs
    rag.ULTRA_QUICK_MODE = True

    results_dir = str(tmp_path / "results_integration")

    # Run MNIST experiment with a single seed (fast)
    rag.run_mnist_experiment(results_dir=results_dir, seeds=[42], quick=True, skip_tuning=False, resume=False)

    # Check that tuning artifacts were created
    optuna_dir = Path(results_dir) / 'optuna_results' / 'MNIST_Benchmark'
    assert optuna_dir.exists()
    tuning_files = list(optuna_dir.glob('*.tuning.json'))
    assert len(tuning_files) >= 1

    # Check that run metadata references a tuning artifact
    experiments_dir = Path(results_dir) / 'experiments' / 'mnist'
    meta_files = list(experiments_dir.glob('*.metadata.json'))
    assert len(meta_files) >= 1

    found = False
    for mf in meta_files:
        data = mf.read_text()
        if 'tuning_artifact' in data:
            found = True
            break

    assert found, "No per-run metadata contained 'tuning_artifact' reference"
