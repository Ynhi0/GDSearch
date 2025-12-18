"""
Pytest configuration for GDSearch test suite.

Configures warning filters to reduce noise from known deprecations
and third-party library warnings while maintaining visibility of
project-specific issues.

AUDIT FIX (Dec 18, 2025): Added to reduce test warning noise from 216k to manageable levels.
"""
import warnings
import pytest


def pytest_configure(config):
    """Configure pytest with warning filters."""
    # Suppress specific deprecation warnings from third-party libraries
    
    # 1. Pillow 'mode' parameter deprecation in torchvision MNIST
    # Will be removed in Pillow 13 (2026-10-15)
    warnings.filterwarnings(
        "ignore",
        message="'mode' parameter is deprecated",
        category=DeprecationWarning,
        module="PIL"
    )
    
    # 2. Matplotlib boxplot 'labels' parameter (already fixed in our code)
    # This filter is for any remaining instances in test fixtures
    warnings.filterwarnings(
        "ignore",
        message="The 'labels' parameter of boxplot.*has been renamed",
        category=DeprecationWarning
    )
    
    # 3. PyTorch LR scheduler ordering (already documented/fixed in training code)
    # Suppress for tests that intentionally check edge cases
    warnings.filterwarnings(
        "ignore",
        message="Detected call of.*lr_scheduler.step.*before.*optimizer.step",
        category=UserWarning,
        module="torch.optim.lr_scheduler"
    )
    
    # 4. SciPy Shapiro-Wilk warnings for zero-variance data (already fixed with guards)
    # This filter is for any edge-case tests that intentionally use degenerate data
    warnings.filterwarnings(
        "ignore",
        message="Input data has range zero",
        category=UserWarning,
        module="scipy.stats._axis_nan_policy"
    )
    
    # Keep other warnings visible for debugging
    # Don't suppress:
    # - RuntimeWarnings (numerical issues)
    # - Our custom warnings (tuning safety, config validation)
    # - Errors and exceptions


@pytest.fixture(autouse=False)
def suppress_test_warnings():
    """
    Optional fixture to suppress warnings for specific tests.
    
    Usage:
        def test_something(suppress_test_warnings):
            # Test code here
            pass
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
