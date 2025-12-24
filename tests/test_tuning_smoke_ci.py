import os
import torch
from torch.utils.data import TensorDataset
from run_all_kaggle import quick_tune_optimizer, make_dataloader, SimpleMLP, ULTRA_QUICK_MODE


def test_quick_tune_optimizer_ultra_quick_mode():
    # Ensure ULTRA_QUICK_MODE is respected and tuning runs quickly
    os.environ['GDSEARCH_TUNE_EVAL_ALL_CANDIDATES'] = 'false'
    os.environ['GDSEARCH_TUNE_SEED_COUNT'] = '1'

    # Prepare tiny dataset
    X = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    ds = TensorDataset(X, y)

    train_loader = make_dataloader(ds, batch_size=4, shuffle=True, seed=42, num_workers=0, split_type='train')
    val_loader = make_dataloader(ds, batch_size=4, shuffle=False, seed=42, num_workers=0, split_type='validation')

    # Force ULTRA_QUICK_MODE in-memory (not altering global by import); if variable exists, ensure it's True for the test
    try:
        # If module variable exists, set it
        import run_all_kaggle as rag
        rag.ULTRA_QUICK_MODE = True
    except Exception:
        pass

    best_params, best_val = quick_tune_optimizer('SGD', SimpleMLP, train_loader, val_loader, device=torch.device('cpu'), epochs=1, n_trials=1, seed=42)

    assert isinstance(best_params, dict)
    assert isinstance(best_val, float)
    assert best_val >= 0.0
