"""
Cross-process checkpoint tests for optimizer wrappers.

CRITICAL: Tests that optimizer state survives checkpoint save/load across
different Python processes (simulating Kaggle kernel restarts).

This ensures the fix for id(p)-based serialization is working correctly.
"""

import pytest
import torch
import torch.nn as nn
import subprocess
import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pytorch_optimizers import (
    SGDMomentumWrapper,
    AdamWrapper,
    SGDNesterovWrapper,
    RMSPropWrapper,
    AdamWWrapper
)


def create_simple_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )


def create_test_data():
    """Create test input/target data."""
    x = torch.randn(8, 10, dtype=torch.float32)
    y = torch.randint(0, 5, (8,))
    return x, y


def _train_and_get_state(optimizer_cls, kwargs, state_before):
    """Train optimizer in separate process and return final state."""
    import torch
    import torch.nn as nn
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and optimizer
    model = create_simple_model()
    optimizer = optimizer_cls(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()
    
    # Load initial state
    optimizer.load_state_dict(state_before)
    
    # Train for 5 steps
    for _ in range(5):
        x = torch.randn(4, 10, dtype=torch.float32)
        y = torch.randint(0, 2, (4,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    return optimizer.state_dict()


class TestCrossProcessCheckpoint:
    """Test checkpoint serialization across process boundaries."""
    
    @pytest.mark.parametrize("optimizer_class,kwargs", [
        (SGDMomentumWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (AdamWrapper, {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}),
        (SGDNesterovWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (RMSPropWrapper, {'lr': 0.01, 'alpha': 0.99}),
        (AdamWWrapper, {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01}),
    ])
    def test_in_process_state_preservation(self, optimizer_class, kwargs):
        """Test that optimizer state is preserved in same process (baseline)."""
        model = create_simple_model()
        x, y = create_test_data()
        
        # Create optimizer and take some steps
        optimizer = optimizer_class(model.parameters(), **kwargs)
        criterion = nn.CrossEntropyLoss()
        
        # Take 3 training steps to build up optimizer state
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Save state
        model_state = model.state_dict()
        opt_state = optimizer.state_dict()
        
        # Create new model and optimizer
        new_model = create_simple_model()
        new_optimizer = optimizer_class(new_model.parameters(), **kwargs)
        
        # Load state
        new_model.load_state_dict(model_state)
        new_optimizer.load_state_dict(opt_state)
        
        # Take one more step with both
        for opt, mdl in [(optimizer, model), (new_optimizer, new_model)]:
            opt.zero_grad()
            out = mdl(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
        
        # Compare model parameters after step
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-7,
                msg=f"{optimizer_class.__name__} state not preserved correctly")
    
    @pytest.mark.parametrize("optimizer_class,kwargs", [
        (SGDMomentumWrapper, {'lr': 0.01, 'momentum': 0.9}),
        (AdamWrapper, {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}),
        (AdamWWrapper, {'lr': 0.001, 'betas': (0.9, 0.999), 'weight_decay': 0.01}),
    ])
    def test_disk_checkpoint_roundtrip(self, optimizer_class, kwargs):
        """Test checkpoint save/load through disk (critical for Kaggle)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Phase 1: Train and save checkpoint
            model = create_simple_model()
            x, y = create_test_data()
            optimizer = optimizer_class(model.parameters(), **kwargs)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few steps
            for _ in range(5):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # Save checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_data': {'x': x, 'y': y}
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Get current model state for comparison
            params_before_save = [p.clone() for p in model.parameters()]
            
            # Phase 2: Load checkpoint in same process
            loaded_checkpoint = torch.load(checkpoint_path)
            new_model = create_simple_model()
            new_optimizer = optimizer_class(new_model.parameters(), **kwargs)
            
            new_model.load_state_dict(loaded_checkpoint['model'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer'])
            
            x_test = loaded_checkpoint['test_data']['x']
            y_test = loaded_checkpoint['test_data']['y']
            
            # Take one step with original and loaded
            optimizer.zero_grad()
            output = model(x_test)
            loss = criterion(output, y_test)
            loss.backward()
            optimizer.step()
            
            new_optimizer.zero_grad()
            output_new = new_model(x_test)
            loss_new = criterion(output_new, y_test)
            loss_new.backward()
            new_optimizer.step()
            
            # Compare parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-6,
                    msg=f"{optimizer_class.__name__} checkpoint roundtrip failed")
    
    def test_subprocess_checkpoint_roundtrip(self):
        """
        CRITICAL TEST: Verify checkpoint works across different Python processes.
        
        This simulates Kaggle kernel restart scenario where process memory is cleared.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            result_path = Path(tmpdir) / "result.json"
            
            # Step 1: Create checkpoint in this process
            model = create_simple_model()
            x, y = create_test_data()
            optimizer = AdamWrapper(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Train
            for _ in range(5):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # Save checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'x': x,
                'y': y
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Take one more step in current process (reference)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            params_reference = [p.detach().cpu().numpy().tolist() for p in model.parameters()]
            
            # Step 2: Create subprocess that loads checkpoint and takes same step
            subprocess_script = f"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, r'{Path(__file__).parent.parent}')
from src.core.pytorch_optimizers import AdamWrapper

# Define model architecture
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Load checkpoint
checkpoint = torch.load(r'{checkpoint_path}')
optimizer = AdamWrapper(model.parameters(), lr=0.001)

model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

x = checkpoint['x']
y = checkpoint['y']

# Take same step as parent process
criterion = nn.CrossEntropyLoss()
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()

# Save results
params_subprocess = [p.detach().cpu().numpy().tolist() for p in model.parameters()]
result = {{'params': params_subprocess}}

with open(r'{result_path}', 'w') as f:
    json.dump(result, f)
"""
            
            # Run subprocess
            result = subprocess.run(
                [sys.executable, '-c', subprocess_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check subprocess succeeded
            assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
            
            # Load subprocess results
            with open(result_path, 'r') as f:
                subprocess_result = json.load(f)
            
            params_subprocess = subprocess_result['params']
            
            # Compare parameters from both processes
            import numpy as np
            for i, (ref, sub) in enumerate(zip(params_reference, params_subprocess)):
                ref_arr = np.array(ref)
                sub_arr = np.array(sub)
                np.testing.assert_allclose(
                    ref_arr, sub_arr,
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Parameter {i} differs between processes - optimizer state not preserved!"
                )


def test_optimizer_state_keys_are_serializable():
    """Verify that all optimizer state dict keys are JSON-serializable."""
    model = create_simple_model()
    x, y = create_test_data()
    
    optimizers_to_test = [
        SGDMomentumWrapper(model.parameters(), lr=0.01, momentum=0.9),
        AdamWrapper(model.parameters(), lr=0.001),
        AdamWWrapper(model.parameters(), lr=0.001, weight_decay=0.01),
    ]
    
    criterion = nn.CrossEntropyLoss()
    
    for optimizer in optimizers_to_test:
        # Train to build state
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Get state dict
        state = optimizer.state_dict()
        
        # Try to JSON serialize (this would fail with non-serializable keys)
        try:
            json_str = json.dumps(state, default=lambda o: str(o) if isinstance(o, torch.Tensor) else o.__class__.__name__)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"{optimizer.__class__.__name__} has non-serializable state keys: {e}")


if __name__ == '__main__':
    # Run quick test
    print("Running cross-process checkpoint tests...")
    pytest.main([__file__, '-v', '--tb=short'])
