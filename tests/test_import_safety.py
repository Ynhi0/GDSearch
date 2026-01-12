"""
Test that importing modules does not have side effects.

This test ensures that:
1. Importing top-level scripts does not mutate sys.stdout or sys.stderr
2. Import-time side effects don't break test harnesses (pytest, etc.)
3. Global state is not modified at import time

This is important for CI/CD and programmatic consumption of the codebase.
"""

import sys
import io
import pytest


def test_run_all_kaggle_import_safety():
    """Test that importing run_all_kaggle does not mutate stdout/stderr."""
    # Capture original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_stdout_type = type(sys.stdout)
    original_stderr_type = type(sys.stderr)

    # Import the module
    import run_all_kaggle  # noqa: F401

    # Verify stdout/stderr were not replaced
    assert sys.stdout is original_stdout, \
        "Importing run_all_kaggle mutated sys.stdout (breaks pytest capture)"
    assert sys.stderr is original_stderr, \
        "Importing run_all_kaggle mutated sys.stderr (breaks pytest capture)"

    # Verify types didn't change (additional safety check)
    assert type(sys.stdout) == original_stdout_type, \
        f"sys.stdout type changed from {original_stdout_type} to {type(sys.stdout)}"
    assert type(sys.stderr) == original_stderr_type, \
        f"sys.stderr type changed from {original_stderr_type} to {type(sys.stderr)}"


def test_quick_validation_import_safety():
    """Test that importing scripts.quick_validation_test doesn't mutate streams."""
    # Capture original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Import the module
    from scripts import quick_validation_test  # noqa: F401

    # Verify stdout/stderr were not replaced
    assert sys.stdout is original_stdout, \
        "Importing quick_validation_test mutated sys.stdout"
    assert sys.stderr is original_stderr, \
        "Importing quick_validation_test mutated sys.stderr"


def test_pytest_capture_compatibility():
    """
    Test that the codebase is compatible with pytest capture mechanisms.

    This simulates what pytest does internally: replacing stdout/stderr with
    capture objects that have .buffer attributes, then reading from them later.
    """
    # Simulate pytest's capture system
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create mock capture objects similar to pytest's
    class MockCapture:
        """Mock pytest capture object with buffer."""
        def __init__(self, stream):
            self.stream = stream
            self.buffer = stream.buffer if hasattr(stream, 'buffer') else None

        def write(self, text):
            return self.stream.write(text)

        def flush(self):
            return self.stream.flush()

    try:
        # Replace with mock captures (simulating pytest)
        sys.stdout = MockCapture(original_stdout)
        sys.stderr = MockCapture(original_stderr)

        # Now import modules - they should NOT mutate these
        import run_all_kaggle  # noqa: F401, F811
        from scripts import quick_validation_test  # noqa: F401, F811

        # Verify the mock captures are still in place
        assert isinstance(sys.stdout, MockCapture), \
            "Module import replaced pytest's stdout capture"
        assert isinstance(sys.stderr, MockCapture), \
            "Module import replaced pytest's stderr capture"

    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def test_no_global_state_mutation():
    """Test that importing modules doesn't modify global configuration."""
    import os

    # Capture original environment
    original_env = os.environ.copy()

    # These env vars might be set but shouldn't cause issues
    allowed_env_vars = {
        'TF_CPP_MIN_LOG_LEVEL', 'CUDA_VISIBLE_DEVICES_ORDER',
        'GRPC_VERBOSITY', 'GLOG_minloglevel',
        'PYTHONIOENCODING', 'PYTHONUTF8',  # Windows encoding vars
        'CUBLAS_WORKSPACE_CONFIG'  # Deterministic CUDA
    }

    # Import modules
    import run_all_kaggle  # noqa: F401, F811

    # Check for unexpected environment changes
    new_vars = set(os.environ.keys()) - set(original_env.keys())
    unexpected_vars = new_vars - allowed_env_vars

    assert not unexpected_vars, \
        f"Import added unexpected env vars: {unexpected_vars}"

    # Verify sys.path wasn't polluted with relative paths
    for path in sys.path[:5]:  # Check first 5 entries
        assert not path.startswith('.'), \
            f"Import added relative path to sys.path: {path}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
