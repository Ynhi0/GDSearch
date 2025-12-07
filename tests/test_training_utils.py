"""
Tests for advanced training utilities.
"""

import pytest
import torch
import torch.nn as nn
from src.core.training_utils import (
    LabelSmoothingCrossEntropy,
    ModelEMA,
    AMPWrapper,
    get_loss_function,
    create_amp_wrapper,
    create_model_ema
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class TestLabelSmoothingCrossEntropy:
    """Test Label Smoothing Cross Entropy loss."""
    
    def test_zero_smoothing_equals_standard_ce(self):
        """Test that zero smoothing equals standard cross entropy."""
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.0)
        standard_loss = nn.CrossEntropyLoss()
        
        loss1 = smooth_loss(pred, target)
        loss2 = standard_loss(pred, target)
        
        assert torch.allclose(loss1, loss2, atol=1e-6)
    
    def test_smoothing_reduces_confidence(self):
        """Test that smoothing reduces overconfidence."""
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        standard_loss = nn.CrossEntropyLoss()
        
        loss_smooth = smooth_loss(pred, target)
        loss_standard = standard_loss(pred, target)
        
        # Smoothed loss should be different
        assert not torch.allclose(loss_smooth, loss_standard)
    
    def test_reduction_mean(self):
        """Test mean reduction."""
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, reduction='mean')
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0  # Scalar output
    
    def test_reduction_sum(self):
        """Test sum reduction."""
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, reduction='sum')
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0  # Scalar output
    
    def test_reduction_none(self):
        """Test no reduction."""
        batch_size = 4
        pred = torch.randn(batch_size, 10)
        target = torch.randint(0, 10, (batch_size,))
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, reduction='none')
        loss = loss_fn(pred, target)
        
        assert loss.shape == (batch_size,)
    
    def test_backward_pass(self):
        """Test that gradients can be computed."""
        pred = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 10, (4,))
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestModelEMA:
    """Test Model EMA (Exponential Moving Average)."""
    
    def test_initialization(self):
        """Test EMA initialization."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)
        
        assert ema.decay == 0.999
        assert ema.shadow is not None
    
    def test_shadow_weights_update(self):
        """Test that shadow weights update correctly."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9)
        
        # Get initial shadow weights
        initial_shadow = {name: param.clone() 
                         for name, param in ema.shadow.named_parameters()}
        
        # Update model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Update EMA
        ema.update(model)
        
        # Check shadow weights changed
        changed = False
        for name, param in ema.shadow.named_parameters():
            if not torch.allclose(param, initial_shadow[name]):
                changed = True
                break
        
        assert changed, "Shadow weights did not update"
    
    def test_ema_decay(self):
        """Test EMA decay mechanism."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.99)
        
        # Perform multiple updates
        for _ in range(10):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update(model)
        
        # Shadow should be different from model
        different = False
        for model_param, shadow_param in zip(model.parameters(), ema.shadow.parameters()):
            if not torch.allclose(model_param, shadow_param, atol=1e-3):
                different = True
                break
        
        assert different, "Shadow should differ from model after updates"
    
    def test_state_dict(self):
        """Test state dict save/load."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)
        
        # Update a few times
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update(model)
        
        # Save state
        state = ema.state_dict()
        
        # Create new EMA and load
        ema2 = ModelEMA(SimpleModel(), decay=0.9)
        ema2.load_state_dict(state)
        
        # Check weights match
        for p1, p2 in zip(ema.shadow.parameters(), ema2.shadow.parameters()):
            assert torch.allclose(p1, p2)
        
        assert ema2.decay == 0.999


class TestAMPWrapper:
    """Test Automatic Mixed Precision wrapper."""
    
    def test_initialization_cpu(self):
        """Test AMP wrapper on CPU (should disable)."""
        amp = AMPWrapper(enabled=False)
        assert amp.enabled is False
        assert amp.scaler is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_cuda(self):
        """Test AMP wrapper on CUDA."""
        amp = AMPWrapper(enabled=True)
        assert amp.enabled is True
        assert amp.scaler is not None
    
    def test_autocast_context(self):
        """Test autocast context manager."""
        amp = AMPWrapper(enabled=False)
        
        x = torch.randn(4, 10)
        with amp.autocast():
            y = x * 2
        
        assert y.dtype == x.dtype
    
    def test_backward_without_amp(self):
        """Test backward pass without AMP."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        amp = AMPWrapper(enabled=False)
        
        x = torch.randn(4, 10)
        target = torch.randint(0, 5, (4,))
        
        with amp.autocast():
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
        
        amp.backward(loss, optimizer)
        amp.step(optimizer)
        amp.update()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
    
    def test_state_dict(self):
        """Test state dict for AMP wrapper."""
        amp = AMPWrapper(enabled=False)
        state = amp.state_dict()
        
        assert 'enabled' in state
        assert state['enabled'] is False
        
        amp2 = AMPWrapper(enabled=False)
        amp2.load_state_dict(state)
        
        assert amp2.enabled == amp.enabled


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_get_loss_function_cross_entropy(self):
        """Test getting standard cross entropy."""
        loss_fn = get_loss_function('cross_entropy', label_smoothing=0.0)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
    
    def test_get_loss_function_label_smoothing(self):
        """Test getting label smoothing cross entropy."""
        loss_fn = get_loss_function('cross_entropy', label_smoothing=0.1)
        assert isinstance(loss_fn, LabelSmoothingCrossEntropy)
    
    def test_get_loss_function_bce(self):
        """Test getting BCE loss."""
        loss_fn = get_loss_function('bce')
        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)
    
    def test_get_loss_function_mse(self):
        """Test getting MSE loss."""
        loss_fn = get_loss_function('mse')
        assert isinstance(loss_fn, nn.MSELoss)
    
    def test_get_loss_function_invalid(self):
        """Test invalid loss type raises error."""
        with pytest.raises(ValueError):
            get_loss_function('invalid_loss')
    
    def test_create_amp_wrapper(self):
        """Test AMP wrapper creation."""
        amp = create_amp_wrapper(enabled=False)
        assert isinstance(amp, AMPWrapper)
        assert amp.enabled is False
    
    def test_create_model_ema(self):
        """Test Model EMA creation."""
        model = SimpleModel()
        ema = create_model_ema(model, decay=0.999)
        
        assert isinstance(ema, ModelEMA)
        assert ema.decay == 0.999


class TestIntegration:
    """Integration tests for training utilities."""
    
    def test_full_training_loop_with_amp_and_ema(self):
        """Test full training loop with AMP and EMA."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = get_loss_function('cross_entropy', label_smoothing=0.1)
        amp = create_amp_wrapper(enabled=False)  # CPU testing
        ema = create_model_ema(model, decay=0.999)
        
        # Training loop
        for _ in range(3):
            x = torch.randn(4, 10)
            target = torch.randint(0, 5, (4,))
            
            with amp.autocast():
                output = model(x)
                loss = criterion(output, target)
            
            amp.backward(loss, optimizer)
            amp.step(optimizer)
            amp.update()
            
            # Update EMA
            ema.update(model)
        
        # Check everything ran successfully
        assert loss.item() > 0
        
        # EMA shadow should exist
        assert ema.shadow is not None
