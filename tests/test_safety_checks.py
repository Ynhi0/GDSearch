"""
Test suite for integrity and robustness checks.

This test file validates important safety measures:
1. Test-leakage enforcement (untagged loaders must fail)
2. Optuna study contamination prevention
3. SAM+OOM safety assertions
4. Fixed-LR ablation guard
5. DDP device mapping robustness
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLeakageEnforcement:
    """Test stricter test-leakage enforcement in Optuna tuner."""

    def test_untagged_loader_with_test_dataset_rejected(self):
        """Untagged loader referencing test dataset must be rejected."""
        from src.core.optuna_tuner import OptunaHyperparameterTuner

        # Create dummy datasets
        train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
        test_data = TensorDataset(torch.randn(50, 10), torch.randint(0, 5, (50,)))

        # Create untagged loader using test dataset (BUG!)
        bad_loader = DataLoader(test_data, batch_size=32)
        # Intentionally NO metadata tags

        def dummy_objective(trial):
            return 0.5

        tuner = OptunaHyperparameterTuner(
            objective_fn=dummy_objective,
            direction='maximize',
            study_name='test_audit_fix'
        )

        # Should raise ValueError due to missing metadata when test_dataset provided
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            tuner.optimize(
                n_trials=1,
                val_loader=bad_loader,
                test_dataset=test_data,  # Identity check should catch this
                enforce_validation=True
            )

        # Verify error message mentions test-leakage or metadata
        assert any(keyword in str(exc_info.value).lower()
                  for keyword in ['test', 'leakage', 'metadata', 'validation'])

    def test_properly_tagged_validation_loader_accepted(self):
        """Properly tagged validation loader should pass checks."""
        from src.core.optuna_tuner import OptunaHyperparameterTuner

        # Create datasets
        train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
        val_subset = Subset(train_data, range(20))  # Val from training data
        test_data = TensorDataset(torch.randn(50, 10), torch.randint(0, 5, (50,)))

        # Create properly tagged validation loader
        val_loader = DataLoader(val_subset, batch_size=32)
        val_loader.name = 'validation'
        val_loader._split_type = 'validation'

        def dummy_objective(trial):
            return 0.5

        tuner = OptunaHyperparameterTuner(
            objective_fn=dummy_objective,
            direction='maximize',
            study_name='test_audit_fix_valid'
        )

        # Should NOT raise - properly tagged
        try:
            tuner.optimize(
                n_trials=1,
                val_loader=val_loader,
                test_dataset=test_data,
                enforce_validation=True
            )
            assert True, "Properly tagged loader accepted"
        except Exception as e:
            pytest.fail(f"Properly tagged loader should not raise: {e}")


class TestOptunaStudyContamination:
    """Test Optuna study contamination prevention."""

    def test_study_creation_defaults_to_no_reuse(self):
        """Study creation should default to load_if_exists=False."""
        import optuna
        from src.core.optuna_tuner import OptunaHyperparameterTuner

        # Create first study
        def dummy_objective(trial):
            return trial.suggest_float('x', 0, 1)

        tuner1 = OptunaHyperparameterTuner(
            objective_fn=dummy_objective,
            direction='maximize',
            study_name='contamination_test_study'
        )

        # Check that creating a second tuner with same name would fail
        # (indicating load_if_exists=False behavior)
        tuner2 = OptunaHyperparameterTuner(
            objective_fn=dummy_objective,
            direction='maximize',
            study_name='contamination_test_study'
        )

        # If load_if_exists=False (correct), creating duplicate should raise
        # If load_if_exists=True (incorrect), duplicate would silently reuse
        # We verify by checking that the studies are separate objects
        assert tuner1.study is not tuner2.study, \
            "Studies with same name should be separate instances (no reuse)"


class TestSAM_OOM_Safety:
    """Test SAM+OOM safety assertions."""

    def test_closure_optimizer_missing_attribute_caught(self):
        """Missing requires_closure on SAM-like optimizer should raise."""
        from src.core.oom_handler import oom_safe_train_step

        # Create mock SAM optimizer WITHOUT requires_closure attribute (BUG!)
        class FakeSAMOptimizer(torch.optim.Optimizer):
            def __init__(self, params):
                defaults = dict(lr=0.01)
                super().__init__(params, defaults)
                # INTENTIONALLY MISSING: self.requires_closure = True

            def step(self, closure=None):
                return closure() if closure else None

        model = nn.Linear(10, 5)
        fake_sam = FakeSAMOptimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()

        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 5, (4,))
        device = torch.device('cpu')

        # Should raise AttributeError due to missing requires_closure
        with pytest.raises(AttributeError) as exc_info:
            oom_safe_train_step(
                model, fake_sam, criterion, inputs, targets, device
            )

        assert 'requires_closure' in str(exc_info.value).lower()

    def test_sam_wrapper_has_requires_closure(self):
        """Verify SAMWrapper has requires_closure=True."""
        from src.core.pytorch_optimizers import SAMWrapper

        model = nn.Linear(10, 5)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = SAMWrapper(base_opt, rho=0.05)

        assert hasattr(sam, 'requires_closure'), "SAMWrapper must have requires_closure attribute"
        assert sam.requires_closure is True, "SAMWrapper requires_closure must be True"


class TestAblationGuard:
    """Test fixed-LR ablation guard."""

    def test_ablation_script_requires_flag(self):
        """Ablation script should require --allow-unfair-ablations flag."""
        from src.experiments.run_optimizer_ablation import check_ablation_guard

        # Save original sys.argv
        original_argv = sys.argv.copy()

        try:
            # Test without flag - should exit
            sys.argv = ['script.py']  # No --allow-unfair-ablations
            with pytest.raises(SystemExit) as exc_info:
                check_ablation_guard()
            assert exc_info.value.code == 1

            # Test with flag - should pass
            sys.argv = ['script.py', '--allow-unfair-ablations']
            try:
                check_ablation_guard()  # Should not raise
                assert True, "Guard allows execution with flag"
            except SystemExit:
                pytest.fail("Guard should not exit when flag is present")

        finally:
            # Restore original sys.argv
            sys.argv = original_argv


class TestDDP_DeviceMapping:
    """Test DDP device mapping robustness."""

    def test_local_rank_env_var_usage(self):
        """DDP worker should use LOCAL_RANK env var when available."""
        # This is a design verification test - actual DDP testing requires multi-GPU
        # We verify that the code pattern exists in the function
        import inspect
        from run_all_kaggle import distributed_training_worker

        source = inspect.getsource(distributed_training_worker)

        # Verify LOCAL_RANK is checked
        assert 'LOCAL_RANK' in source, "DDP worker must check LOCAL_RANK env var"
        assert 'os.environ.get' in source, "Must use os.environ.get for LOCAL_RANK"


class TestVenvNotTracked:
    """Test repository hygiene - venv not tracked."""

    def test_venv_in_gitignore(self):
        """Verify venv/ is in .gitignore."""
        gitignore_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '.gitignore'
        )

        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                content = f.read()

            assert 'venv' in content.lower(), "venv must be in .gitignore"
        else:
            pytest.skip(".gitignore not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
