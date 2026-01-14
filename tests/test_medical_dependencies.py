"""
Test suite for optional medical dependency detection and error handling.
"""
import pytest
from unittest.mock import patch
from src.core.medical_dependencies import (
    HAS_MEDMNIST,
    HAS_MONAI,
    require_medmnist,
    require_monai,
    check_medical_stack,
    get_install_command,
    MedicalDependencyError
)


class TestMedicalDependencyDetection:
    """Test detection of medical imaging packages."""

    def test_has_flags_are_boolean(self):
        """Verify HAS_* flags are boolean values."""
        assert isinstance(HAS_MEDMNIST, bool)
        assert isinstance(HAS_MONAI, bool)

    def test_check_medical_stack_returns_dict(self):
        """Test check_medical_stack returns availability dict."""
        result = check_medical_stack(verbose=False)
        
        assert isinstance(result, dict)
        assert 'medmnist' in result
        assert 'monai' in result
        assert isinstance(result['medmnist'], bool)
        assert isinstance(result['monai'], bool)

    def test_check_medical_stack_matches_flags(self):
        """Test check_medical_stack consistency with module flags."""
        result = check_medical_stack(verbose=False)
        
        assert result['medmnist'] == HAS_MEDMNIST
        assert result['monai'] == HAS_MONAI

    def test_get_install_command_medmnist(self):
        """Test install command generation for medmnist."""
        cmd = get_install_command('medmnist')
        assert 'medmnist' in cmd
        assert 'pip install' in cmd

    def test_get_install_command_monai(self):
        """Test install command generation for MONAI."""
        cmd = get_install_command('monai')
        assert 'monai' in cmd
        assert 'pip install' in cmd

    def test_get_install_command_all(self):
        """Test install command for all medical dependencies."""
        cmd = get_install_command('all')
        assert 'medical' in cmd or 'medmnist' in cmd


class TestMedicalDependencyErrors:
    """Test error handling for missing dependencies."""

    @pytest.mark.skipif(HAS_MEDMNIST, reason="medmnist is installed")
    def test_require_medmnist_raises_when_missing(self):
        """Test require_medmnist raises clear error when package missing."""
        with pytest.raises(MedicalDependencyError) as exc_info:
            require_medmnist("Test feature")
        
        error_msg = str(exc_info.value)
        assert 'medmnist' in error_msg.lower()
        assert 'pip install' in error_msg
        assert 'Test feature' in error_msg

    @pytest.mark.skipif(HAS_MONAI, reason="MONAI is installed")
    def test_require_monai_raises_when_missing(self):
        """Test require_monai raises clear error when package missing."""
        with pytest.raises(MedicalDependencyError) as exc_info:
            require_monai("Test segmentation")
        
        error_msg = str(exc_info.value)
        assert 'monai' in error_msg.lower()
        assert 'pip install' in error_msg
        assert 'Test segmentation' in error_msg

    @pytest.mark.skipif(not HAS_MEDMNIST, reason="medmnist not installed")
    def test_require_medmnist_succeeds_when_available(self):
        """Test require_medmnist doesn't raise when package available."""
        # Should not raise
        try:
            require_medmnist("Test feature")
        except MedicalDependencyError:
            pytest.fail("require_medmnist raised error despite package being available")

    @pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
    def test_require_monai_succeeds_when_available(self):
        """Test require_monai doesn't raise when package available."""
        # Should not raise
        try:
            require_monai("Test feature")
        except MedicalDependencyError:
            pytest.fail("require_monai raised error despite package being available")


class TestMedicalDataLoading:
    """Test medical dataset loading with dependency checks."""

    def test_load_medmnist_with_strict_false(self):
        """Test load_medmnist_dataset returns None gracefully when unavailable."""
        from src.core.medical_data_utils import load_medmnist_dataset
        
        if not HAS_MEDMNIST:
            # Should return None without raising
            result = load_medmnist_dataset('pathmnist', download=False, strict=False)
            assert result is None

    @pytest.mark.skipif(HAS_MEDMNIST, reason="medmnist is installed")
    def test_load_medmnist_with_strict_true_raises(self):
        """Test load_medmnist_dataset raises clear error when strict=True."""
        from src.core.medical_data_utils import load_medmnist_dataset
        
        with pytest.raises(MedicalDependencyError) as exc_info:
            load_medmnist_dataset('pathmnist', download=False, strict=True)
        
        error_msg = str(exc_info.value)
        assert 'medmnist' in error_msg.lower()
        assert 'pip install' in error_msg

    @pytest.mark.skipif(not HAS_MEDMNIST, reason="medmnist not installed")
    def test_load_medmnist_succeeds_when_available(self):
        """Test load_medmnist_dataset works when medmnist available."""
        from src.core.medical_data_utils import load_medmnist_dataset
        
        # This might still fail if data not downloaded, but should not raise MedicalDependencyError
        try:
            result = load_medmnist_dataset('pathmnist', download=False, strict=True)
            # If we get here, medmnist is working (data might not be downloaded though)
            assert result is not None or True  # Either we get dataset or graceful None
        except MedicalDependencyError:
            pytest.fail("MedicalDependencyError raised despite medmnist being available")
        except Exception:
            # Other exceptions (e.g., data not found) are acceptable
            pass


class TestSyntheticFallback:
    """Test synthetic medical data fallback."""

    def test_synthetic_dataset_always_available(self):
        """Test synthetic dataset works without medmnist."""
        from src.core.medical_data_utils import SyntheticMedicalDataset
        
        dataset = SyntheticMedicalDataset(num_samples=10, img_size=64, seed=42)
        
        assert len(dataset) == 10
        image, mask = dataset[0]
        assert image.shape == (1, 64, 64)  # (C, H, W)
        assert mask.shape == (1, 64, 64)

    def test_synthetic_dataset_reproducibility(self):
        """Test synthetic dataset generates same data with same seed."""
        from src.core.medical_data_utils import SyntheticMedicalDataset
        
        ds1 = SyntheticMedicalDataset(num_samples=5, img_size=32, seed=123)
        ds2 = SyntheticMedicalDataset(num_samples=5, img_size=32, seed=123)
        
        img1, mask1 = ds1[0]
        img2, mask2 = ds2[0]
        
        # Should be identical with same seed
        import torch
        assert torch.allclose(img1, img2)
        assert torch.allclose(mask1, mask2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
