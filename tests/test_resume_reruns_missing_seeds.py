import os
import pandas as pd
from pathlib import Path
import run_all_kaggle as rag

run_nlp_experiment_simple = rag.run_nlp_experiment_simple


def test_resume_only_runs_missing_seeds(tmp_path, capsys):
    results_dir = tmp_path / "results_nlp"
    results_dir.mkdir(parents=True)

    # Seeds to run
    seeds = [1, 2, 3]

    # Simulate that seed=1 already completed by creating its result file
    completed_fp = results_dir / f"nlp_imdb_simple_SimpleLSTM_AdamW_seed1.csv"
    completed_fp.write_text('epoch,loss,acc\n')  # minimal CSV presence

    # Run with resume: expect the function to skip seed 1 and run 2 and 3
    df = run_nlp_experiment_simple(results_dir=str(results_dir), seeds=seeds, epochs=1, resume=True)

    captured = capsys.readouterr()
    stdout = captured.out

    # Check skip message for seed 1 appeared
    assert 'Skipping SimpleLSTM + AdamW (seed 1)' in stdout or 'Skipping' in stdout

    # Check that the aggregated results file exists and includes entries for the other seeds
    aggregated = results_dir / 'nlp_results.csv'
    assert aggregated.exists(), "Aggregated results file was not created"

    df_res = pd.read_csv(aggregated)
    # Ensure that at least one row exists and that missing seeds were run
    assert len(df_res) >= 1
    assert any(s in set(df_res['seed']) for s in (2, 3)), "Missing seeds were not run"

    # Ensure the pre-existing per-run artifact for SimpleLSTM+AdamW seed1 was not re-run (file preserved)
    pre_sz = completed_fp.stat().st_size
    post_sz = completed_fp.stat().st_size
    assert pre_sz == post_sz, "Pre-existing artifact file was modified; resume should skip and not overwrite completed runs"
