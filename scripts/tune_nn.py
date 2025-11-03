import os
import json
from typing import Dict, Any, Tuple, List

import pandas as pd

from src.experiments.run_nn_experiment import train_and_evaluate, result_filename
from src.visualization.plot_results import plot_generalization_gap, plot_layer_grad_norms

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def run_and_save(cfg: Dict[str, Any], tag: str) -> Tuple[str, pd.DataFrame]:
    cfg = dict(cfg)
    cfg['tag'] = tag
    df = train_and_evaluate(cfg)
    out = os.path.join(RESULTS_DIR, result_filename(cfg))
    df.to_csv(out, index=False)
    return out, df


def best_by_eval(csv_paths: List[str], prefer: str = 'accuracy') -> Tuple[str, float]:
    """
    Return best CSV path by final eval metric.
    prefer: 'accuracy' or 'loss' (min loss). Default accuracy.
    """
    best_path = None
    best_score = None
    for p in csv_paths:
        df = pd.read_csv(p)
        ev = df[df.get('phase', '') == 'eval']
        if ev.empty:
            continue
        if prefer == 'accuracy':
            score = ev['test_accuracy'].iloc[-1]
            better = (best_score is None) or (score > best_score)
        else:
            score = ev['test_loss'].iloc[-1]
            better = (best_score is None) or (score < best_score)
        if better:
            best_score = float(score)
            best_path = p
    return best_path, (best_score if best_score is not None else float('nan'))


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def tune_optimizer(base: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Two-stage tuning: LR sweep, then optimizer-specific params at best LR.
    Returns best config dict.
    """
    csvs = []
    # Stage 1: LR sweep
    for lr in spec.get('lr_values', []):
        cfg = {
            'dataset': base['dataset'],
            'model': base['model'],
            'seed': base['seed'],
            'batch_size': base['batch_size'],
            'optimizer': spec['optimizer'],
            'lr': lr,
            'epochs': spec.get('epochs', 3),
        }
        if spec['optimizer'].upper().startswith('ADAM') and 'weight_decay_values' in spec:
            # Try default wd 0.0 in LR sweep
            cfg['weight_decay'] = 0.0
        if spec['optimizer'].upper().startswith('SGD') and 'momentum_values' in spec:
            cfg['momentum'] = 0.9
        out, _ = run_and_save(cfg, tag='sweepLR')
        csvs.append(out)
    best_lr_path, _ = best_by_eval(csvs, prefer='accuracy')
    if best_lr_path is None:
        # fallback: choose middle lr
        best_lr = spec.get('lr_values', [1e-3])[min(1, len(spec.get('lr_values', [1e-3])) // 2)]
    else:
        best_lr = float(pd.read_csv(best_lr_path).iloc[0]['lr']) if 'lr' in pd.read_csv(best_lr_path).columns else spec.get('lr_values', [1e-3])[0]

    # Stage 2: Optimizer-specific grid at best lr
    best_cfg = None
    best_score = None
    csvs_stage2 = []
    if spec['optimizer'].upper().startswith('ADAM'):
        for wd in spec.get('weight_decay_values', [0.0]):
            cfg = {
                'dataset': base['dataset'],
                'model': base['model'],
                'seed': base['seed'],
                'batch_size': base['batch_size'],
                'optimizer': spec['optimizer'],
                'lr': best_lr,
                'epochs': spec.get('epochs', 3),
                'weight_decay': wd,
            }
            out, _ = run_and_save(cfg, tag='sweepWD')
            csvs_stage2.append(out)
    elif spec['optimizer'].upper().startswith('SGD'):
        for mom in spec.get('momentum_values', [0.0, 0.9]):
            cfg = {
                'dataset': base['dataset'],
                'model': base['model'],
                'seed': base['seed'],
                'batch_size': base['batch_size'],
                'optimizer': spec['optimizer'],
                'lr': best_lr,
                'epochs': spec.get('epochs', 3),
                'momentum': mom,
            }
            out, _ = run_and_save(cfg, tag='sweepMOM')
            csvs_stage2.append(out)
    # pick best from stage2 (or from stage1 if none)
    candidate_csvs = csvs_stage2 if csvs_stage2 else csvs
    best_path, _ = best_by_eval(candidate_csvs, prefer='accuracy')
    if best_path is None:
        # default best config
        best_cfg = {
            'dataset': base['dataset'],
            'model': base['model'],
            'seed': base['seed'],
            'batch_size': base['batch_size'],
            'optimizer': spec['optimizer'],
            'lr': best_lr,
            'epochs': spec.get('epochs', 3),
        }
    else:
        # reconstruct config from filename is brittle; instead read csv meta: use the first train row for lr & get final epoch for selection
        df = pd.read_csv(best_path)
        best_cfg = {
            'dataset': base['dataset'],
            'model': base['model'],
            'seed': base['seed'],
            'batch_size': base['batch_size'],
            'optimizer': spec['optimizer']
        }
        # best lr
        if 'lr' in df.columns:
            first_train = df[df.get('phase','')=='train']
            if not first_train.empty:
                best_cfg['lr'] = float(first_train['lr'].iloc[0])
        # mom/wd
        if spec['optimizer'].upper().startswith('SGD'):
            # cannot recover momentum from csv; infer from filename if present, fallback to 0.9
            name = os.path.basename(best_path)
            if 'mom' in name:
                try:
                    frag = name.split('mom')[1]
                    val = float(frag.split('_')[0].replace('.csv',''))
                    best_cfg['momentum'] = val
                except Exception:
                    best_cfg['momentum'] = 0.9
            else:
                best_cfg['momentum'] = 0.9
        else:
            # Adam/AdamW
            best_cfg['weight_decay'] = float(df.get('weight_decay', pd.Series([0.0])).iloc[0]) if 'weight_decay' in df.columns else 0.0

    return best_cfg


def run_final(cfg: Dict[str, Any], final_spec: Dict[str, Any], conv_spec: Dict[str, Any]):
    fcfg = dict(cfg)
    fcfg['epochs'] = final_spec.get('epochs', 20)
    fcfg['capture_layer_grad_epochs'] = final_spec.get('capture_layer_grad_epochs', [1, fcfg['epochs']])
    # convergence settings
    fcfg['convergence_grad_norm_threshold'] = conv_spec.get('grad_norm_threshold', 0.0)
    fcfg['convergence_loss_delta_threshold'] = conv_spec.get('loss_delta_threshold', 0.0)
    fcfg['convergence_loss_window'] = conv_spec.get('loss_window', 0)
    out, df = run_and_save(fcfg, tag='final')
    # plots
    base = os.path.splitext(os.path.basename(out))[0]
    plot_generalization_gap(df, title=f"GenGap & TestAcc: {base}", save_path=os.path.join(PLOTS_DIR, f"{base}_gen_gap.png"))
    plot_layer_grad_norms(df, title=f"Per-layer Grad Norms: {base}", save_path=os.path.join(PLOTS_DIR, f"{base}_layer_grads.png"))


def main():
    ensure_dirs()
    cfg = load_config('configs/nn_tuning.json')
    base = {
        'dataset': cfg['dataset'],
        'model': cfg['model'],
        'seed': cfg.get('seed', 1),
        'batch_size': cfg.get('batch_size', 128),
    }
    finals = []
    for spec in cfg['sweeps']:
        best_cfg = tune_optimizer(base, spec)
        finals.append(best_cfg)
    for c in finals:
        run_final(c, cfg['final'], cfg['convergence'])


if __name__ == '__main__':
    main()
