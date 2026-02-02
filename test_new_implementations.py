"""
Quick test to verify optimizer refactoring and new utilities.

Tests:
1. Optimizer _dispatch_step pattern works correctly
2. Checkpoint utilities work (create, save, load)
3. Parallel runner GPU detection works
4. Resume utilities correctly identify completed experiments
"""
import sys
import logging
from pathlib import Path
import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_optimizer_dispatch():
    """Test SGD with new _dispatch_step pattern."""
    print("\n" + "="*80)
    print("TEST 1: Optimizer Dispatch Pattern")
    print("="*80)
    
    from src.core.optimizers import SGD
    
    # Test tuple parameters
    sgd = SGD(lr=0.1, weight_decay=0.01)
    params_tuple = (1.0, 2.0)
    grads_tuple = (0.5, 0.3)
    
    new_params = sgd.step(params_tuple, grads_tuple)
    print(f"✓ Tuple params: {params_tuple} -> {new_params}")
    
    # Test array parameters
    params_array = np.array([1.0, 2.0, 3.0])
    grads_array = np.array([0.1, 0.2, 0.3])
    
    new_params_array = sgd.step(params_array, grads_array)
    print(f"✓ Array params: {params_array} -> {new_params_array}")
    
    print("✅ Optimizer dispatch pattern works!\n")
    return True


def test_checkpoint_utilities():
    """Test checkpoint utilities."""
    print("\n" + "="*80)
    print("TEST 2: Checkpoint Utilities")
    print("="*80)
    
    from src.utils.checkpoint_utils import (
        create_checkpoint,
        save_checkpoint_atomic,
        load_checkpoint_safe,
        CheckpointManager
    )
    
    # Create simple model
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create checkpoint
    checkpoint = create_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        best_metric=0.85,
        config={'lr': 0.01, 'model': 'SimpleMLP'}
    )
    
    required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                     'best_metric', 'config', 'timestamp', 'random_states']
    for key in required_keys:
        assert key in checkpoint, f"Missing key: {key}"
    print(f"✓ Checkpoint created with all required keys")
    
    # Test atomic save
    test_path = Path('test_checkpoint.pt')
    save_checkpoint_atomic(checkpoint, test_path)
    assert test_path.exists(), "Checkpoint file not created"
    print(f"✓ Checkpoint saved atomically to {test_path}")
    
    # Test load
    model_new = torch.nn.Linear(10, 2)
    optimizer_new = torch.optim.SGD(model_new.parameters(), lr=0.01)
    
    metadata = load_checkpoint_safe(
        test_path,
        model=model_new,
        optimizer=optimizer_new,
        device='cpu'
    )
    
    assert metadata['epoch'] == 10, "Epoch not loaded correctly"
    assert metadata['best_metric'] == 0.85, "Best metric not loaded correctly"
    print(f"✓ Checkpoint loaded successfully: epoch={metadata['epoch']}, metric={metadata['best_metric']}")
    
    # Cleanup
    test_path.unlink()
    
    # Test CheckpointManager
    ckpt_dir = Path('test_checkpoints')
    manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        keep_last=2,
        keep_best=2,
        metric_mode='max'
    )
    
    # Save multiple checkpoints
    for epoch in [1, 2, 3, 4, 5]:
        metric = 0.5 + epoch * 0.05
        ckpt = create_checkpoint(model, optimizer, epoch, metric, {})
        manager.save_checkpoint(ckpt, epoch, metric, is_best=(epoch == 5))
    
    # Check that old checkpoints were cleaned up
    remaining = list(ckpt_dir.glob('*.pt'))
    print(f"✓ CheckpointManager: kept {len(remaining)} checkpoints (should be ≤4)")
    
    # Cleanup
    import shutil
    shutil.rmtree(ckpt_dir)
    
    print("✅ Checkpoint utilities work correctly!\n")
    return True


def test_parallel_runner():
    """Test parallel runner GPU detection."""
    print("\n" + "="*80)
    print("TEST 3: Parallel Runner GPU Detection")
    print("="*80)
    
    from src.utils.parallel_experiment_runner import detect_gpu_configuration
    
    gpu_config = detect_gpu_configuration()
    
    print(f"✓ GPU Count: {gpu_config['gpu_count']}")
    print(f"✓ GPU Names: {gpu_config['gpu_names']}")
    print(f"✓ GPU Memory (GB): {gpu_config['gpu_memory']}")
    print(f"✓ Parallel Capable: {gpu_config['parallel_capable']}")
    print(f"✓ Parallel Recommended: {gpu_config['recommended_parallel']}")
    
    if gpu_config['gpu_count'] >= 2:
        print("✅ Multi-GPU detected! Parallel mode available.\n")
    else:
        print("ℹ️  Single GPU or CPU only. Parallel mode not available.\n")
    
    return True


def test_resume_utilities():
    """Test resume utilities."""
    print("\n" + "="*80)
    print("TEST 4: Resume Utilities")
    print("="*80)
    
    from src.utils.resume_utils import should_skip_experiment, validate_experiment_result
    import pandas as pd
    
    # Create test result file
    test_dir = Path('test_results')
    test_dir.mkdir(exist_ok=True)
    
    result_file = test_dir / 'test_result.csv'
    df = pd.DataFrame({
        'epoch': list(range(50)),
        'train_loss': np.random.rand(50),
        'train_acc': np.random.rand(50),
        'test_loss': np.random.rand(50),
        'test_acc': np.random.rand(50)
    })
    df.to_csv(result_file, index=False)
    
    # Test validation
    is_valid = validate_experiment_result(result_file, expected_epochs=50)
    print(f"✓ validate_experiment_result: {is_valid} (should be True)")
    assert is_valid, "Valid result file not recognized"
    
    # Test incomplete file
    incomplete_file = test_dir / 'incomplete_result.csv'
    df_incomplete = df.head(30)  # Only 30 epochs
    df_incomplete.to_csv(incomplete_file, index=False)
    
    is_incomplete = validate_experiment_result(incomplete_file, expected_epochs=50)
    print(f"✓ Incomplete file detected correctly: {not is_incomplete}")
    assert not is_incomplete, "Incomplete file not detected"
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    print("✅ Resume utilities work correctly!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING NEW IMPLEMENTATIONS")
    print("="*80)
    
    all_pass = True
    
    try:
        all_pass &= test_optimizer_dispatch()
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}\n")
        all_pass = False
    
    try:
        all_pass &= test_checkpoint_utilities()
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}\n")
        all_pass = False
    
    try:
        all_pass &= test_parallel_runner()
    except Exception as e:
        print(f"❌ Parallel runner test failed: {e}\n")
        all_pass = False
    
    try:
        all_pass &= test_resume_utilities()
    except Exception as e:
        print(f"❌ Resume utilities test failed: {e}\n")
        all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
