import os, sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.experiments.run_nn_experiment import train_and_evaluate, result_filename

cfg = {
    'model': 'SimpleMLP',
    'dataset': 'MNIST',
    'optimizer': 'Adam',
    'lr': 1e-3,
    'epochs': 1,
    'batch_size': 32,
    'seed': 42,
    'val_split': 0.1
}

out_df = train_and_evaluate(cfg)
print('Columns in produced DataFrame:', list(out_df.columns))
print('Number of rows:', len(out_df))
print(out_df.head().to_csv(index=False))

os.makedirs('results/debug', exist_ok=True)
fname = result_filename(cfg)
out_path = os.path.join('results/debug', fname)
out_df.to_csv(out_path, index=False)
print('Saved CSV to', out_path)
