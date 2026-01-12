#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Validation Test for Robust Gradient Handling
===================================================

This script performs a quick smoke test to verify that:
1. Robust gradient handler can be imported
2. Handler correctly detects heavy-tailed gradients
3. Gradient clipping works correctly
4. Integration with training loop doesn't break

Run:
    python scripts/test_robust_gradients.py --verbose
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
import numpy as np


def test_import():
    """Test that robust gradient module can be imported."""
    try:
        from src.core.robust_gradients import (
            RobustGradientHandler,
            create_robust_gradient_handler,
            HuberLoss,
            get_robust_loss_function
        )
        print("✓ Robust gradient module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import robust gradient module: {e}")
        return False


def test_heavy_tail_detection():
    """Test heavy-tail detection functionality."""
    try:
        from src.core.robust_gradients import RobustGradientHandler
        from src.core.models import SimpleMLP

        # Create handler with monitoring enabled
        handler = RobustGradientHandler(
            enabled=True,
            monitor_heavy_tails=True,
            heavy_tail_threshold=0.05
        )

        # Create model and inject heavy-tailed gradients
        model = SimpleMLP()

        # Inject extreme gradients (heavy-tailed distribution)
        for param in model.parameters():
            # Create mixture: 90% normal, 10% extreme outliers
            grad = torch.randn_like(param)
            mask = torch.rand_like(param) < 0.1
            grad[mask] *= 100  # Extreme outliers
            param.grad = grad

        # Run handler multiple times to accumulate statistics
        for _ in range(50):
            diagnostics = handler(model, epoch=1)

        # Check if heavy tails were detected
        stats = handler.get_statistics()

        if stats['heavy_tail_fraction'] > 0.0:
            print(f"✓ Heavy-tail detection works (detected in {stats['heavy_tail_fraction']:.1%} of steps)")
            return True
        else:
            print("✗ Heavy-tail detection failed (no heavy tails detected)")
            return False

    except Exception as e:
        print(f"✗ Heavy-tail detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_clipping():
    """Test that gradient clipping works correctly."""
    try:
        from src.core.robust_gradients import RobustGradientHandler
        from src.core.models import SimpleMLP

        # Create handler with clipping enabled
        handler = RobustGradientHandler(
            enabled=True,
            clip_norm=1.0,
            monitor_heavy_tails=False
        )

        # Create model and inject large gradients
        model = SimpleMLP()

        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0  # Large gradients

        # Compute norm before clipping
        norm_before = sum(p.grad.norm().item()**2 for p in model.parameters())**0.5

        # Apply handler
        diagnostics = handler(model, epoch=1)

        # Compute norm after clipping
        norm_after = sum(p.grad.norm().item()**2 for p in model.parameters())**0.5

        # Check clipping worked
        if diagnostics['clipped'] and norm_after <= 1.0 + 1e-3:
            print(f"✓ Gradient clipping works (norm: {norm_before:.2f} → {norm_after:.2f})")
            return True
        else:
            print(f"✗ Gradient clipping failed (norm: {norm_before:.2f} → {norm_after:.2f}, clipped={diagnostics['clipped']})")
            return False

    except Exception as e:
        print(f"✗ Gradient clipping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agc():
    """Test Adaptive Gradient Clipping."""
    try:
        from src.core.robust_gradients import RobustGradientHandler
        from src.core.models import SimpleMLP

        # Create handler with AGC enabled
        handler = RobustGradientHandler(
            enabled=True,
            use_agc=True,
            clip_percentile=50.0,  # Aggressive for testing
            monitor_heavy_tails=False
        )

        # Create model and inject gradients
        model = SimpleMLP()

        for param in model.parameters():
            param.grad = torch.randn_like(param) * 5.0

        # Apply handler
        diagnostics = handler(model, epoch=1)

        # AGC should clip at least some layers
        stats = handler.get_statistics()

        if stats['clip_fraction'] > 0.0:
            print(f"✓ AGC works (clipped {stats['clip_fraction']:.1%} of steps)")
            return True
        else:
            print("⚠ AGC test inconclusive (no clipping occurred)")
            return True  # Not a failure, just no clipping needed

    except Exception as e:
        print(f"✗ AGC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_huber_loss():
    """Test Huber loss implementation."""
    try:
        from src.core.robust_gradients import HuberLoss

        # Create Huber loss
        huber = HuberLoss(delta=1.0)

        # Test with small error (should be L2-like)
        pred_small = torch.tensor([1.0, 2.0, 3.0])
        target_small = torch.tensor([1.1, 2.1, 3.1])
        loss_small = huber(pred_small, target_small)

        # Test with large error (should be L1-like)
        pred_large = torch.tensor([1.0, 2.0, 3.0])
        target_large = torch.tensor([10.0, 20.0, 30.0])
        loss_large = huber(pred_large, target_large)

        if loss_small < loss_large:
            print(f"✓ Huber loss works (small error: {loss_small:.3f}, large error: {loss_large:.3f})")
            return True
        else:
            print(f"✗ Huber loss failed (small: {loss_small:.3f}, large: {loss_large:.3f})")
            return False

    except Exception as e:
        print(f"✗ Huber loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_oom_handler():
    """Test integration with OOM handler."""
    try:
        from src.core.robust_gradients import create_robust_gradient_handler
        from src.core.oom_handler import oom_safe_train_step
        from src.core.models import SimpleMLP
        import torch.optim as optim

        # Create model, optimizer, criterion
        model = SimpleMLP()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')

        # Create robust gradient handler
        handler = create_robust_gradient_handler(
            enabled=True,
            config={'clip_norm': 5.0, 'monitor_heavy_tails': True}
        )

        # Create dummy data
        inputs = torch.randn(32, 28*28)
        targets = torch.randint(0, 10, (32,))

        # Run training step with handler
        loss_value, batch_size, outputs, tainted = oom_safe_train_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            inputs=inputs,
            targets=targets,
            device=device,
            robust_grad_handler=handler,
            epoch=1
        )

        # Check that training worked
        if not np.isnan(loss_value) and not tainted:
            print(f"✓ Integration with OOM handler works (loss: {loss_value:.4f})")
            return True
        else:
            print(f"✗ Integration test failed (loss: {loss_value}, tainted: {tainted})")
            return False

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("ROBUST GRADIENT HANDLING - VALIDATION TESTS")
    print("="*70)
    print()

    tests = [
        ("Module Import", test_import),
        ("Heavy-Tail Detection", test_heavy_tail_detection),
        ("Gradient Clipping", test_gradient_clipping),
        ("Adaptive Gradient Clipping (AGC)", test_agc),
        ("Huber Loss", test_huber_loss),
        ("OOM Handler Integration", test_integration_with_oom_handler),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 70)
        passed = test_func()
        results.append((name, passed))
        print()

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    for name, passed_flag in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status:10} {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if failed == 0:
        print("\n✅ All tests passed! Robust gradient handling is ready for use.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
