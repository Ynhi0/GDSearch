# Error Handling Quick Reference

**Quick guide for using error handling best practices in GDSearch**

## Import the Utilities

```python
from src.utils.error_handling_patterns import (
    gpu_safe_operation,
    model_cleanup_guard,
    validate_preconditions,
    atomic_save_checkpoint,
    ErrorContext,
    safe_gpu_operation,
    log_and_reraise
)
```

## Common Patterns

### 1. GPU Training with Automatic Cleanup

**Problem:** GPU memory leaks when training crashes  
**Solution:** Use `model_cleanup_guard`

```python
with model_cleanup_guard(model):
    train_loop(model, data_loader)
# Model and GPU cache always cleaned up, even on error
```

### 2. OOM-Safe GPU Operations

**Problem:** CUDA OOM errors crash training  
**Solution:** Use `gpu_safe_operation`

```python
with gpu_safe_operation("Forward pass"):
    output = model(batch)
    loss = criterion(output, target)
# Catches OOM, clears cache, re-raises with context
```

### 3. Validate Before Training

**Problem:** Training fails hours in due to bad config  
**Solution:** Validate preconditions early

```python
validate_preconditions(
    model=model,
    data_loader=train_loader,
    epochs=100,
    learning_rate=0.001,
    batch_size=32
)
# Raises informative error if anything is invalid
```

### 4. Atomic Checkpoint Saves

**Problem:** Checkpoint corruption on crash  
**Solution:** Use `atomic_save_checkpoint`

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
atomic_save_checkpoint(checkpoint, 'checkpoints/model.pt')
# Either fully written or not at all (never corrupted)
```

### 5. Add Context to Errors

**Problem:** Generic errors hard to debug  
**Solution:** Use `ErrorContext`

```python
with ErrorContext("Training epoch 5, batch 100"):
    train_step(model, batch)
# Error messages include context automatically
```

### 6. Decorator for GPU Functions

**Problem:** Repeating try/except for GPU code  
**Solution:** Use `@safe_gpu_operation` decorator

```python
@safe_gpu_operation
def train_step(model, batch):
    output = model(batch)
    loss.backward()
    return loss.item()
# Automatic OOM handling and cleanup
```

### 7. Log Before Re-raising

**Problem:** Lost context when errors propagate  
**Solution:** Use `@log_and_reraise` decorator

```python
@log_and_reraise("Model training", context={"epoch": 5})
def train_epoch(model, loader):
    for batch in loader:
        train_step(model, batch)
# Logs full context before re-raising
```

## Complete Example

```python
from src.utils.error_handling_patterns import *

def robust_training_loop(model, train_loader, epochs, lr):
    """Training loop with comprehensive error handling."""
    
    # 1. Validate preconditions early
    validate_preconditions(
        model=model,
        data_loader=train_loader,
        epochs=epochs,
        learning_rate=lr
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Ensure cleanup on any error
    with model_cleanup_guard(model):
        model = model.to(device)
        
        for epoch in range(epochs):
            # 3. Add context for debugging
            with ErrorContext(f"Training epoch {epoch+1}/{epochs}"):
                
                # 4. GPU-safe operations
                with gpu_safe_operation("Epoch training"):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
            
            # 5. Atomic checkpoint save
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
                atomic_save_checkpoint(
                    checkpoint, 
                    f'checkpoints/epoch_{epoch+1}.pt',
                    operation_name=f"epoch {epoch+1} checkpoint"
                )
    
    # Model and GPU memory automatically cleaned up
```

## When to Use Each Pattern

| Pattern | Use When | Benefit |
|---------|----------|---------|
| `model_cleanup_guard` | Any GPU training | Prevents memory leaks |
| `gpu_safe_operation` | GPU operations | OOM recovery |
| `validate_preconditions` | Function entry | Fail fast |
| `atomic_save_checkpoint` | Saving checkpoints | Prevent corruption |
| `ErrorContext` | Multi-step operations | Better debugging |
| `@safe_gpu_operation` | Reusable GPU functions | Clean code |
| `@log_and_reraise` | Long-running functions | Preserve context |

## Integration with Existing Code

### Existing Utilities (Keep Using)
```python
# Atomic file writes
from src.utils.atomic_io import safe_write_csv, safe_write_json

# Safe device transfers
from src.core.device_utils import safe_to_device, clear_gpu_memory

# OOM-safe training
from src.core.oom_handler import oom_safe_train_step

# Checkpoint management
from src.core.checkpoint_manager import RobustCheckpointManager
```

### New Utilities (Optional Enhancement)
```python
# Error handling patterns (new)
from src.utils.error_handling_patterns import (
    gpu_safe_operation,
    model_cleanup_guard,
    validate_preconditions,
    atomic_save_checkpoint
)
```

**Both work together!** The new utilities complement existing infrastructure.

## Common Mistakes to Avoid

### ❌ Don't: Silent failures
```python
try:
    train_model()
except:
    pass  # Error hidden!
```

### ✅ Do: Catch specific errors
```python
try:
    train_model()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logging.error(f"OOM: {e}")
        torch.cuda.empty_cache()
        raise
    else:
        raise
```

### ❌ Don't: Leak GPU memory
```python
model = create_model().to("cuda")
train_loop(model)  # Crash = memory leak
```

### ✅ Do: Always cleanup
```python
with model_cleanup_guard(model):
    train_loop(model)
# Always cleaned up
```

### ❌ Don't: Corrupt checkpoints
```python
torch.save(checkpoint, path)  # Partial write on crash
```

### ✅ Do: Use atomic writes
```python
atomic_save_checkpoint(checkpoint, path)
# Either complete or not at all
```

## Testing Error Handling

```python
def test_oom_recovery():
    """Test that OOM is handled gracefully."""
    model = HugeModel()  # Too big for GPU
    
    with model_cleanup_guard(model):
        with gpu_safe_operation("Testing"):
            try:
                model = model.to("cuda")
            except RuntimeError as e:
                assert "out of memory" in str(e).lower()
                # Verify cache was cleared
                assert torch.cuda.memory_allocated() < initial_memory

def test_checkpoint_atomic():
    """Test checkpoint atomicity."""
    checkpoint = {'data': 'test'}
    path = 'test_checkpoint.pt'
    
    # Simulate crash during save
    with pytest.raises(Exception):
        with mock.patch('torch.save', side_effect=Exception("Crash!")):
            atomic_save_checkpoint(checkpoint, path)
    
    # Verify: no corrupted file left behind
    assert not Path(path).exists()
```

## Further Reading

- **Full documentation:** [ERROR_HANDLING_IMPROVEMENTS.md](../ERROR_HANDLING_IMPROVEMENTS.md)
- **Working examples:** [examples/error_handling_demo.py](../examples/error_handling_demo.py)
- **Source code:** [src/utils/error_handling_patterns.py](../src/utils/error_handling_patterns.py)
- **Existing patterns:** [run_all_kaggle.py](../run_all_kaggle.py) (lines 3060-3420)

---

**Remember:** The codebase already has excellent error handling. These utilities make good patterns even easier to apply consistently.
