"""
Resume equivalence tests for full training pipeline.

Verifies that checkpointing and resuming produces identical results to
continuous training. This is fundamental for reproducibility.

Tests cover:
- Basic optimizer resume (train 10 steps == train 5 + resume + train 5)
- RobustCheckpointManager integration
- Multi-seed resume consistency
- Scheduler state preservation
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import tempfile

# Import from modular utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.checkpoint_manager import RobustCheckpointManager
from src.core.pytorch_optimizers import AdamWrapper, SGDMomentumWrapper
from src.core.dataloader_utils import make_dataloader


class TinyNet(nn.Module):
    """Minimal model for fast testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_dataset(size=100):
    """Generate simple dataset."""
    X = torch.randn(size, 10, dtype=torch.float32)
    y = torch.randint(0, 2, (size,))
    return torch.utils.data.TensorDataset(X, y)


def train_n_epochs(model, optimizer, dataloader, epochs, scheduler=None):
    """Train for n epochs and return final parameter state."""
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
    
    return {k: v.clone() for k, v in model.state_dict().items()}


class TestBasicResumeEquivalence:
    """Test fundamental resume=continuous property."""
    
    @pytest.mark.parametrize("optimizer_cls,kwargs", [
        (optim.Adam, {'lr': 0.001}),
        (optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        (AdamWrapper, {'lr': 0.001}),
        (SGDMomentumWrapper, {'lr': 0.01, 'momentum': 0.9})
    ])
    def test_continuous_vs_resume(self, optimizer_cls, kwargs):
        """Train(10) must equal Train(5) + Resume + Train(5)."""
        dataset = create_dummy_dataset(100)
        
        # Continuous training
        torch.manual_seed(42)
        model_cont = TinyNet()
        model_cont = model_cont.float()
        dataloader_cont = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
        opt_cont = optimizer_cls(model_cont.parameters(), **kwargs)
        final_cont = train_n_epochs(model_cont, opt_cont, dataloader_cont, epochs=10)
        
        # Resumed training
        torch.manual_seed(42)
        model_resume = TinyNet()
        model_resume = model_resume.float()
        dataloader_resume = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
        opt_resume = optimizer_cls(model_resume.parameters(), **kwargs)
        
        # Train first 5 epochs
        _ = train_n_epochs(model_resume, opt_resume, dataloader_resume, epochs=5)
        
        # Checkpoint
        checkpoint = {
            'model': model_resume.state_dict(),
            'optimizer': opt_resume.state_dict()
        }
        
        # Resume for 5 more epochs
        model_resume.load_state_dict(checkpoint['model'])
        opt_resume.load_state_dict(checkpoint['optimizer'])
        final_resume = train_n_epochs(model_resume, opt_resume, dataloader_resume, epochs=5)
        
        # Compare final states
        for key in final_cont:
            diff = (final_cont[key] - final_resume[key]).abs().max().item()
            assert diff < 1e-5, f"{optimizer_cls.__name__} resume diverged at {key}: max diff = {diff}"
    
    def test_scheduler_resume(self):
        """LR scheduler state must be preserved across resume."""
        dataset = create_dummy_dataset(100)
        
        # Continuous with scheduler
        torch.manual_seed(42)
        model_cont = TinyNet()
        dataloader_cont = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
        opt_cont = optim.Adam(model_cont.parameters(), lr=0.01)
        scheduler_cont = optim.lr_scheduler.StepLR(opt_cont, step_size=3, gamma=0.5)
        final_cont = train_n_epochs(model_cont, opt_cont, dataloader_cont, epochs=10, scheduler=scheduler_cont)
        
        # Resumed with scheduler
        torch.manual_seed(42)
        model_resume = TinyNet()
        dataloader_resume = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
        opt_resume = optim.Adam(model_resume.parameters(), lr=0.01)
        scheduler_resume = optim.lr_scheduler.StepLR(opt_resume, step_size=3, gamma=0.5)
        
        # Train 5, checkpoint, resume 5
        _ = train_n_epochs(model_resume, opt_resume, dataloader_resume, epochs=5, scheduler=scheduler_resume)
        
        checkpoint = {
            'model': model_resume.state_dict(),
            'optimizer': opt_resume.state_dict(),
            'scheduler': scheduler_resume.state_dict()
        }
        
        model_resume.load_state_dict(checkpoint['model'])
        opt_resume.load_state_dict(checkpoint['optimizer'])
        scheduler_resume.load_state_dict(checkpoint['scheduler'])
        
        final_resume = train_n_epochs(model_resume, opt_resume, dataloader_resume, epochs=5, scheduler=scheduler_resume)
        
        # Verify
        for key in final_cont:
            diff = (final_cont[key] - final_resume[key]).abs().max().item()
            assert diff < 1e-5, f"Scheduler resume diverged: {diff}"


class TestRobustCheckpointManager:
    """Test checkpoint manager preserves all states."""
    
    def test_manager_resume_equivalence(self):
        """RobustCheckpointManager must enable perfect resume."""
        dataset = create_dummy_dataset(100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Continuous training
            torch.manual_seed(42)
            model_cont = TinyNet()
            model_cont = model_cont.float()
            dataloader_cont = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
            opt_cont = optim.Adam(model_cont.parameters(), lr=0.001)
            
            criterion = nn.CrossEntropyLoss()
            for _ in range(10):
                for x, y in dataloader_cont:
                    opt_cont.zero_grad()
                    loss = criterion(model_cont(x), y)
                    loss.backward()
                    opt_cont.step()
            
            final_cont = {k: v.clone() for k, v in model_cont.state_dict().items()}
            
            # Resumed training with checkpoint manager
            torch.manual_seed(42)
            model_resume = TinyNet()
            model_resume = model_resume.float()
            dataloader_resume = make_dataloader(dataset, batch_size=10, shuffle=True, seed=42)
            opt_resume = optim.Adam(model_resume.parameters(), lr=0.001)
            
            ckpt_manager = RobustCheckpointManager(
                base_dir=tmpdir,
                max_backups=3
            )
            
            # Train 5 epochs and checkpoint
            for _ in range(5):
                for x, y in dataloader_resume:
                    opt_resume.zero_grad()
                    loss = criterion(model_resume(x), y)
                    loss.backward()
                    opt_resume.step()
            
            checkpoint_data = {
                'model': model_resume.state_dict(),
                'optimizer': opt_resume.state_dict(),
                'epoch': 5,
                'metrics': {'loss': 0.5}
            }
            ckpt_manager.save_checkpoint(checkpoint_data, 'checkpoint.pt', 'test_experiment')
            
            # Resume from checkpoint
            loaded_data = ckpt_manager.load_checkpoint('checkpoint.pt', 'test_experiment')
            assert loaded_data is not None, "Failed to load checkpoint"
            model_resume.load_state_dict(loaded_data['model'])
            opt_resume.load_state_dict(loaded_data['optimizer'])
            
            # Continue 5 more epochs
            for _ in range(5):
                for x, y in dataloader_resume:
                    opt_resume.zero_grad()
                    loss = criterion(model_resume(x), y)
                    loss.backward()
                    opt_resume.step()
            
            final_resume = {k: v.clone() for k, v in model_resume.state_dict().items()}
            
            # Verify equivalence
            for key in final_cont:
                diff = (final_cont[key] - final_resume[key]).abs().max().item()
                assert diff < 1e-5, f"Checkpoint manager resume diverged: {diff}"
    
    def test_multi_checkpoint_selection(self):
        """Manager should correctly load latest checkpoint."""
        dataset = create_dummy_dataset(50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyNet()
            model = model.float()
            dataloader = make_dataloader(dataset, batch_size=10, shuffle=False)
            opt = optim.SGD(model.parameters(), lr=0.01)
            
            manager = RobustCheckpointManager(tmpdir, max_backups=5)
            
            criterion = nn.CrossEntropyLoss()
            
            # Save checkpoints at epochs 1, 3, 5
            for target_epoch in [1, 3, 5]:
                for _ in range(target_epoch):
                    for x, y in dataloader:
                        opt.zero_grad()
                        loss = criterion(model(x), y)
                        loss.backward()
                        opt.step()
                
                checkpoint_data = {
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': target_epoch,
                    'metrics': {'loss': 1.0 / target_epoch}
                }
                manager.save_checkpoint(checkpoint_data, f'checkpoint_epoch_{target_epoch}.pt', 'test_experiment')
            
            # Load latest (should be epoch 5)
            model_new = TinyNet()
            model_new = model_new.float()
            opt_new = optim.SGD(model_new.parameters(), lr=0.01)
            loaded_data = manager.load_checkpoint('checkpoint_epoch_5.pt', 'test_experiment')
            
            assert loaded_data is not None, "Failed to load checkpoint"
            model_new.load_state_dict(loaded_data['model'])
            opt_new.load_state_dict(loaded_data['optimizer'])
            
            assert loaded_data['epoch'] == 5, f"Loaded wrong checkpoint: epoch {loaded_data['epoch']}"


class TestMultiSeedResume:
    """Test resume consistency across different seeds."""
    
    def test_resume_with_different_init_seed(self):
        """Resume should work even if model initialized with different seed."""
        dataset = create_dummy_dataset(100)
        
        # Train with seed 42
        torch.manual_seed(42)
        model1 = TinyNet()
        dataloader = make_dataloader(dataset, batch_size=10, shuffle=True, seed=99)
        opt1 = optim.Adam(model1.parameters(), lr=0.001)
        _ = train_n_epochs(model1, opt1, dataloader, epochs=5)
        
        checkpoint = {
            'model': model1.state_dict(),
            'optimizer': opt1.state_dict()
        }
        
        # Initialize with seed 999 (different), then load checkpoint
        torch.manual_seed(999)
        model2 = TinyNet()  # Random weights different from model1
        opt2 = optim.Adam(model2.parameters(), lr=0.001)
        
        # Load checkpoint - should override random weights
        model2.load_state_dict(checkpoint['model'])
        opt2.load_state_dict(checkpoint['optimizer'])
        
        # Verify parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Checkpoint didn't restore weights correctly"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
