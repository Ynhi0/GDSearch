"""
Example usage of error handling utilities.

This script demonstrates how to use the new error handling patterns
for robust PyTorch training code.

Run:
    python examples/error_handling_demo.py
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

# Import new error handling utilities
from src.utils.error_handling_patterns import (
    gpu_safe_operation,
    model_cleanup_guard,
    validate_preconditions,
    atomic_save_checkpoint,
    ErrorContext,
    safe_gpu_operation
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Example 1: GPU-safe training with automatic cleanup
def example_gpu_safe_training():
    """Demonstrate GPU-safe operation with automatic cleanup."""
    print("\n" + "="*60)
    print("Example 1: GPU-Safe Training with Cleanup")
    print("="*60)
    
    model = nn.Linear(100, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use cleanup guard to ensure resources are freed even on error
    with model_cleanup_guard(model):
        with gpu_safe_operation("Model training"):
            # This will automatically cleanup GPU memory if OOM occurs
            batch = torch.randn(32, 100).to(device)
            output = model(batch)
            loss = output.sum()
            loss.backward()
            
            print("✓ Training completed successfully")
            print("✓ GPU memory will be cleaned up automatically")


# Example 2: Precondition validation
def example_precondition_validation():
    """Demonstrate early precondition validation."""
    print("\n" + "="*60)
    print("Example 2: Precondition Validation")
    print("="*60)
    
    model = nn.Linear(100, 10)
    
    # Simulate a data loader (for demo)
    class DummyLoader:
        def __len__(self):
            return 10
    
    data_loader = DummyLoader()
    
    # Validate preconditions before starting expensive training
    try:
        validate_preconditions(
            model=model,
            data_loader=data_loader,
            epochs=100,
            learning_rate=0.001,
            batch_size=32
        )
        print("✓ All preconditions valid - safe to start training")
    except (ValueError, TypeError) as e:
        print(f"✗ Validation failed: {e}")
        return


# Example 3: Atomic checkpoint saving
def example_atomic_checkpoint():
    """Demonstrate atomic checkpoint saving."""
    print("\n" + "="*60)
    print("Example 3: Atomic Checkpoint Saving")
    print("="*60)
    
    model = nn.Linear(100, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5
    }
    
    # Create temp directory for demo
    checkpoint_dir = Path("temp_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_epoch_10.pt"
    
    try:
        atomic_save_checkpoint(
            checkpoint=checkpoint,
            path=str(checkpoint_path),
            operation_name="training checkpoint"
        )
        print(f"✓ Checkpoint saved atomically: {checkpoint_path}")
        print("✓ No corruption even if crash occurs during save")
        
        # Cleanup demo files
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_dir.rmdir()
    except Exception as e:
        print(f"✗ Checkpoint save failed: {e}")


# Example 4: Error context for debugging
def example_error_context():
    """Demonstrate error context for better debugging."""
    print("\n" + "="*60)
    print("Example 4: Error Context for Debugging")
    print("="*60)
    
    try:
        with ErrorContext("Data loading phase"):
            print("Loading data...")
            # Simulate some work
            
        with ErrorContext("Model initialization"):
            print("Creating model...")
            model = nn.Linear(100, 10)
            
        with ErrorContext("Training epoch 5, batch 100"):
            print("Training...")
            # If an error occurs here, the context will be in the error message
            
        print("✓ All operations completed with context tracking")
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")


# Example 5: Decorator for GPU operations
@safe_gpu_operation
def train_step_with_decorator(model, batch):
    """Example training step with automatic GPU error handling."""
    output = model(batch)
    loss = output.sum()
    loss.backward()
    return loss.item()


def example_gpu_decorator():
    """Demonstrate safe GPU operation decorator."""
    print("\n" + "="*60)
    print("Example 5: GPU Operation Decorator")
    print("="*60)
    
    model = nn.Linear(100, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    batch = torch.randn(32, 100).to(device)
    
    try:
        loss = train_step_with_decorator(model, batch)
        print(f"✓ Training step completed, loss: {loss:.4f}")
        print("✓ GPU errors would be caught and cleaned up automatically")
    except RuntimeError as e:
        print(f"✗ Training failed with proper error handling: {e}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Example 6: Complete training loop with all patterns
def example_complete_training():
    """Demonstrate complete training loop with all error handling patterns."""
    print("\n" + "="*60)
    print("Example 6: Complete Training Loop")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(100, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Validate preconditions
    validate_preconditions(
        model=model,
        epochs=5,
        learning_rate=0.01,
        batch_size=32
    )
    
    # Training with comprehensive error handling
    with model_cleanup_guard(model):
        with ErrorContext("Model initialization"):
            model = model.to(device)
        
        for epoch in range(5):
            with gpu_safe_operation(f"Training epoch {epoch+1}"):
                # Simulate training
                batch = torch.randn(32, 100).to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = output.sum()
                loss.backward()
                optimizer.step()
            
            print(f"✓ Epoch {epoch+1}/5 completed")
        
        # Save checkpoint atomically
        checkpoint_dir = Path("temp_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "final_model.pt"
        
        atomic_save_checkpoint(
            checkpoint={
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },
            path=str(checkpoint_path),
            operation_name="final checkpoint"
        )
        
        print("✓ Training completed successfully with full error protection")
        
        # Cleanup demo files
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_dir.rmdir()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ERROR HANDLING UTILITIES DEMONSTRATION")
    print("="*70)
    print("\nThese examples show robust error handling patterns for PyTorch.")
    print("All patterns ensure proper cleanup even when errors occur.\n")
    
    # Run all examples
    example_gpu_safe_training()
    example_precondition_validation()
    example_atomic_checkpoint()
    example_error_context()
    example_gpu_decorator()
    example_complete_training()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nKey Benefits:")
    print("  ✓ Automatic GPU memory cleanup on errors")
    print("  ✓ Early validation prevents wasted computation")
    print("  ✓ Atomic saves prevent checkpoint corruption")
    print("  ✓ Error context improves debugging")
    print("  ✓ No resource leaks even on crashes")
    print("\nFor more details, see: ERROR_HANDLING_IMPROVEMENTS.md\n")


if __name__ == "__main__":
    main()
