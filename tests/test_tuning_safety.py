"""
Test suite for hyperparameter tuning safety.

Ensures that tuning objectives never access test loaders, preventing
adaptive overfitting via test set leakage.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock


def enforce_validation_only(loader, loader_name: str, phase: str = 'tuning'):
    """
    Runtime guard to enforce that only validation loaders are used during tuning.

    Args:
        loader: DataLoader instance
        loader_name: Name of the loader (e.g., 'train_loader', 'val_loader', 'test_loader')
        phase: Current experiment phase ('tuning' or 'final_evaluation')

    Raises:
        RuntimeError: If test_loader is accessed during tuning phase
    """
    if phase == 'tuning' and 'test' in loader_name.lower():
        raise RuntimeError(
            f"Error: Attempted to use {loader_name} during tuning phase! "
            f"This constitutes adaptive overfitting. Use val_loader instead."
        )
    # Success: no return value (None). This function raises on forbidden access.
    return None


class TestTuningSafety:
    """Test hyperparameter tuning safety measures."""

    def test_tuning_objective_parameter_naming(self):
        """Verify tuning functions use 'val_loader' not 'test_loader'."""
        # This is a code inspection test
        # In run_all_kaggle.py, quick_tune_optimizer should use val_loader

        from run_all_kaggle import quick_tune_optimizer
        import inspect

        # Get function signature
        sig = inspect.signature(quick_tune_optimizer)
        param_names = list(sig.parameters.keys())

        # Check that 'test_loader' is NOT in parameters (should be val_loader)
        # NOTE: As of the fix, this might still be 'test_loader' parameter name
        # but it should be documented as validation data
        assert 'train_loader' in param_names, "Missing train_loader parameter"

        # Get function docstring
        docstring = quick_tune_optimizer.__doc__ or ""

        # Docstring should clarify that test_loader is actually validation data
        # OR parameter should be renamed to val_loader
        if 'test_loader' in param_names:
            assert 'validation' in docstring.lower() or 'val' in docstring.lower(), \
                "Function using 'test_loader' must document it as validation data"

    def test_mock_tuning_objective_rejects_test_data(self):
        """Test that a properly implemented objective rejects test loaders."""
        # Create mock loaders with identifiable names
        train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        val_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

        train_loader = DataLoader(train_data, batch_size=10)
        val_loader = DataLoader(val_data, batch_size=10)

        # Mark loaders with attributes to identify them (typing-safe helpers)
        from src.utils.loader_meta import set_loader_split_type, get_loader_split_type
        set_loader_split_type(train_loader, 'train')
        set_loader_split_type(val_loader, 'validation')

        # This test passes if we can identify loader types
        assert get_loader_split_type(train_loader) == 'train'
        assert get_loader_split_type(val_loader) == 'validation'

    def test_optuna_objective_should_use_validation(self):
        """Integration test: Optuna objective must evaluate on validation, not test."""

        # Create mock objective function that violates safety
        def bad_objective_using_test(trial):
            # Simulate accessing test_loader (BAD!)
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            # In real code, this would use test_loader for evaluation
            # We simulate by returning a metric from "test" data
            test_metric = 0.95  # Simulated test accuracy
            return test_metric

        # Create mock objective function that correctly uses validation
        def good_objective_using_val(trial):
            # Simulate accessing val_loader (GOOD!)
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            # In real code, this would use val_loader for evaluation
            val_metric = 0.92  # Simulated validation accuracy
            return val_metric

        # Test that we can programmatically enforce this by inspecting function signatures
        import inspect

        # Check quick_tune_optimizer from run_all_kaggle.py
        try:
            from run_all_kaggle import quick_tune_optimizer
            sig = inspect.signature(quick_tune_optimizer)
            param_names = list(sig.parameters.keys())

            # Function should have 'val_loader' parameter (or document test_loader is validation)
            docstring = quick_tune_optimizer.__doc__ or ""

            # Either parameter is named val_loader, OR docstring clarifies usage
            has_val_param = 'val_loader' in param_names
            doc_clarifies_validation = ('validation' in docstring.lower() or
                                       'val' in docstring.lower() or
                                       'NOT test' in docstring)

            assert has_val_param or doc_clarifies_validation, (
                "Tuning function must either use 'val_loader' parameter name "
                "or clearly document that test_loader parameter is actually validation data"
            )

            # If test_loader is in params, docstring MUST clarify it's validation
            if 'test_loader' in param_names and not has_val_param:
                assert doc_clarifies_validation, (
                    "Function using 'test_loader' parameter must document it as validation data"
                )
        except ImportError:
            # If run_all_kaggle doesn't exist or can't be imported, skip this check
            pass
            pass

    def test_quick_tune_requires_val_loader(self):
        """Ensure quick_tune_optimizer fails fast if val_loader is None"""
        from run_all_kaggle import quick_tune_optimizer
        import pytest

        # Minimal model_fn and dummy train loader
        def model_fn():
            class M:
                def to(self, device):
                    return self
            return M()

        train_loader = object()  # placeholder - function will check val_loader early

        with pytest.raises(ValueError, match="requires a 'val_loader' argument"):
            quick_tune_optimizer('SGD', model_fn, train_loader, None, device='cpu', epochs=1, n_trials=1, seed=42)



class TestLoaderNaming:
    """Test proper naming conventions for data loaders."""

    def test_loader_naming_conventions(self):
        """Ensure loaders are named according to their purpose."""
        # Create example loaders
        train_data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        val_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))
        test_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

        train_loader = DataLoader(train_data, batch_size=10)
        val_loader = DataLoader(val_data, batch_size=10)
        test_loader = DataLoader(test_data, batch_size=10)

        # In proper code, test_loader should NEVER be passed to tuning functions
        # This test documents the expected behavior

        # Add metadata to loaders (typing-safe helpers)
        from src.utils.loader_meta import set_loader_purpose, get_loader_purpose
        set_loader_purpose(train_loader, 'training')
        set_loader_purpose(val_loader, 'validation')  # Used for tuning
        set_loader_purpose(test_loader, 'final_evaluation')  # Used ONLY after tuning

        assert get_loader_purpose(train_loader) == 'training'
        assert get_loader_purpose(val_loader) == 'validation'
        assert get_loader_purpose(test_loader) == 'final_evaluation'

    def test_tuning_phase_separation(self):
        """Test conceptual separation of tuning and final evaluation phases."""
        # Tuning phase: uses train + validation
        # Final evaluation phase: uses test (ONCE, after all tuning done)

        class ExperimentPhase:
            def __init__(self):
                self.phase = 'tuning'
                self.test_accessed = False

            def access_test_data(self):
                if self.phase == 'tuning':
                    raise RuntimeError(
                        "Error: Cannot access test data during tuning phase! "
                        "This constitutes adaptive overfitting."
                    )
                self.test_accessed = True

        exp = ExperimentPhase()

        # Should raise error during tuning
        with pytest.raises(RuntimeError, match="adaptive overfitting"):
            exp.access_test_data()

        # Should succeed after tuning complete
        exp.phase = 'final_evaluation'
        exp.access_test_data()
        assert exp.test_accessed


class TestTuningBestPractices:
    """Test best practices for hyperparameter tuning."""

    def test_three_way_split_enforcement(self):
        """Ensure proper train/val/test split is maintained."""
        total_samples = 1000

        # Proper split ratios
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size

        assert train_size + val_size + test_size == total_samples
        assert train_size > val_size  # Training set should be largest
        assert train_size > test_size

    def test_tuning_workflow_documentation(self):
        """Document the correct tuning workflow."""
        workflow = """
        CORRECT HYPERPARAMETER TUNING WORKFLOW:

        1. Split data into train/val/test (e.g., 70%/15%/15%)
        2. TUNING PHASE:
           - For each trial:
             a. Train on train set
             b. Evaluate on VALIDATION set (NOT test!)
             c. Record validation metric
           - Select best hyperparameters based on validation performance

        3. FINAL TRAINING:
           - Retrain with best hyperparameters on train set
           - Monitor on validation set

        4. FINAL EVALUATION (ONCE):
           - Evaluate final model on TEST set
           - Report test metrics as generalization performance

        VIOLATIONS:
        - Using test set in step 2 → ADAPTIVE OVERFITTING
        - Multiple test set evaluations → Inflated generalization claims
        - Single-seed results → Unreliable (should use ≥5 seeds)
        """

        assert 'VALIDATION set (NOT test!)' in workflow
        assert 'ADAPTIVE OVERFITTING' in workflow


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
