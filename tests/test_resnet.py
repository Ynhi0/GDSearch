"""
Tests for deeper models (ResNet-18) and residual connections.
"""

import pytest
import torch
import torch.nn as nn
from src.core.models import BasicBlock, ResNet18
from src.core.pytorch_optimizers import SGDWrapper, AdamWrapper


class TestBasicBlock:
    """Test BasicBlock (residual block)."""
    
    def test_identity_shortcut(self):
        """Test basic block with identity shortcut."""
        block = BasicBlock(64, 64, stride=1)
        
        # Create dummy input
        x = torch.randn(2, 64, 32, 32)
        
        # Forward pass
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_projection_shortcut(self):
        """Test basic block with projection shortcut."""
        # Need downsample when channels change
        downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        block = BasicBlock(64, 128, stride=2, downsample=downsample)
        
        # Create dummy input
        x = torch.randn(2, 64, 32, 32)
        
        # Forward pass
        output = block(x)
        
        # Should halve spatial dimensions and double channels
        assert output.shape == (2, 128, 16, 16)
        assert not torch.isnan(output).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through residual connection."""
        block = BasicBlock(64, 64, stride=1)
        
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        output = block(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestResNet18:
    """Test ResNet-18 architecture."""
    
    def test_model_creation(self):
        """Test creating ResNet-18."""
        model = ResNet18(num_classes=10)
        
        # Check model is created
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with CIFAR-10 input size."""
        model = ResNet18(num_classes=10)
        
        # Create dummy CIFAR-10 batch
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        model = ResNet18(num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        # Forward
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward
        loss.backward()
        
        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = ResNet18(num_classes=10)
        
        num_params = model.get_num_parameters()
        
        # ResNet-18 should have ~11M parameters
        assert num_params > 10_000_000
        assert num_params < 12_000_000
    
    def test_dropout(self):
        """Test dropout functionality."""
        model = ResNet18(num_classes=10, dropout=0.5)
        
        x = torch.randn(2, 3, 32, 32)
        
        # Training mode (dropout active)
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)
        
        # Eval mode (dropout inactive)
        model.eval()
        output3 = model(x)
        output4 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output3, output4)
    
    def test_batch_sizes(self):
        """Test with different batch sizes."""
        model = ResNet18(num_classes=10)
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 10)
    
    def test_residual_connections(self):
        """Test that residual connections help gradient flow."""
        model = ResNet18(num_classes=10)
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 10, (2,))
        
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Input should have gradient (proves gradient flows back)
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Gradient should not vanish (not all zeros)
        assert x.grad.abs().max() > 1e-6


class TestResNetTraining:
    """Test training ResNet-18 with custom optimizers."""
    
    def test_training_step_sgd(self):
        """Test one training step with SGD."""
        model = ResNet18(num_classes=10)
        optimizer = SGDWrapper(model.parameters(), lr=0.01)
        
        # Dummy batch
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Save initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Optimizer step
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                changed = True
                break
        
        assert changed, "Parameters did not update"
    
    def test_training_step_adam(self):
        """Test one training step with Adam."""
        model = ResNet18(num_classes=10)
        optimizer = AdamWrapper(model.parameters(), lr=0.001)
        
        # Dummy batch
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # Loss should be valid
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        model = ResNet18(num_classes=10)
        optimizer = AdamWrapper(model.parameters(), lr=0.001)
        
        # Create consistent data
        torch.manual_seed(42)
        x = torch.randn(8, 3, 32, 32)
        target = torch.randint(0, 10, (8,))
        
        losses = []
        
        # Train for several steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should generally decrease (may fluctuate)
        # Check that we're not stuck (loss changes)
        assert len(set(losses)) > 1, "Loss not changing"
    
    def test_model_save_load(self):
        """Test saving and loading model state."""
        model = ResNet18(num_classes=10)
        
        # Save state
        state_dict = model.state_dict()
        
        # Create new model and load
        new_model = ResNet18(num_classes=10)
        new_model.load_state_dict(state_dict)
        
        # Compare outputs
        x = torch.randn(2, 3, 32, 32)
        
        model.eval()
        new_model.eval()
        
        output1 = model(x)
        output2 = new_model(x)
        
        assert torch.allclose(output1, output2)


class TestResNetComparison:
    """Compare ResNet-18 with simpler models."""
    
    def test_resnet_deeper_than_simple_cnn(self):
        """ResNet-18 should have more parameters than SimpleCNN."""
        from src.core.models import SimpleCNN
        
        simple_cnn = SimpleCNN(num_classes=10)
        resnet18 = ResNet18(num_classes=10)
        
        simple_params = sum(p.numel() for p in simple_cnn.parameters())
        resnet_params = resnet18.get_num_parameters()
        
        # ResNet should be much deeper
        assert resnet_params > simple_params * 10
    
    def test_resnet_has_skip_connections(self):
        """Verify ResNet has skip connections by checking BasicBlock usage."""
        model = ResNet18(num_classes=10)
        
        # Check that model contains BasicBlock modules
        has_basic_blocks = False
        for module in model.modules():
            if isinstance(module, BasicBlock):
                has_basic_blocks = True
                break
        
        assert has_basic_blocks, "ResNet-18 should contain BasicBlock modules"
