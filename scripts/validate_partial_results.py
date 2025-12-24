import json, csv, sys, argparse, os, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
parser = argparse.ArgumentParser(description='Validate partial results directory')
parser.add_argument('--results-dir', default=os.environ.get('GDSEARCH_RESULTS_DIR', 'results/results_full'), help='Path to results directory')
args = parser.parse_args()

root = Path(args.results_dir)

issues = []
summary = {'total_csv':0,'total_meta':0,'tainted':0,'partial_runs':0,'small_runs':0,'missing_checkpoints':0}
for exp_dir in (root/'experiments').rglob('*.csv'):
    summary['total_csv'] += 1
    try:
        # check rows
        with exp_dir.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
            nrows = len(rows)-1 if len(rows)>0 else 0
        meta_path = exp_dir.with_suffix('.metadata.json')
        if not meta_path.exists():
            issues.append((str(exp_dir), 'missing metadata'))
            continue
        summary['total_meta'] += 1
        meta = json.loads(meta_path.read_text())
        expected_epochs = meta.get('params', {}).get('epochs', None)
        tainted = meta.get('params', {}).get('tainted', False)
        if tainted:
            summary['tainted'] += 1
            issues.append((str(exp_dir), 'tainted true in metadata'))
        if nrows < 2:
            summary['small_runs'] += 1
            issues.append((str(exp_dir), f'small run: rows={nrows}'))
        if expected_epochs and nrows < expected_epochs:
            summary['partial_runs'] += 1
            issues.append((str(exp_dir), f'partial: rows={nrows}, expected={expected_epochs}'))
        # check checkpoint exists
        parts = exp_dir.name.split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            model = parts[1]
            opt = parts[2]
            seed_part = parts[-1]
            seed = seed_part.replace('.csv','')
            # Some naming differences: many CSVs are like MNIST_SimpleMLP_AdaBound_seed42.csv
            ckpt_name = f"{dataset}_{opt}_seed{seed}.pt" if dataset=='CIFAR10' else f"{dataset}_{opt}_seed{seed}.pt"
            ckpt_path = root/'checkpoints'/ckpt_name
            if not ckpt_path.exists():
                summary['missing_checkpoints'] += 1
                issues.append((str(exp_dir), f'missing checkpoint {ckpt_name}'))
    except Exception as e:
        logging.exception(f"Error validating {exp_dir}")
        issues.append((str(exp_dir), f'error: {e}'))

print('SCAN SUMMARY')
print('Total CSV files:', summary['total_csv'])
print('Total metadata:', summary['total_meta'])
print('Tainted runs:', summary['tainted'])
print('Partial runs (rows < expected epochs):', summary['partial_runs'])
print('Small runs (<2 rows):', summary['small_runs'])
print('Missing checkpoints:', summary['missing_checkpoints'])
print('\nISSUES (first 50):')
for i,it in enumerate(issues[:50]):
    print(i+1,'-',it[0],':',it[1])
