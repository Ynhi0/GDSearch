
import unittest
import torch
import os
import shutil
from src.core.checkpoint_manager import RobustCheckpointManager

class TestCheckpointManagerCompat(unittest.TestCase):
    def setUp(self):
        self.base_dir = "test_ckpt_compat"
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        self.manager = RobustCheckpointManager(self.base_dir)

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

    def test_validate_optimizer_compatibility_exact_match(self):
        checkpoint = {'opt_name': 'Adam'}
        self.assertTrue(self.manager.validate_optimizer_compatibility(checkpoint, 'Adam'))
        self.assertTrue(self.manager.validate_optimizer_compatibility(checkpoint, 'adam')) # Case insensitive

    def test_validate_optimizer_compatibility_mismatch(self):
        checkpoint = {'opt_name': 'Adam'}
        self.assertFalse(self.manager.validate_optimizer_compatibility(checkpoint, 'SGD'))

    def test_validate_optimizer_compatibility_fallback_structure(self):
        # No opt_name, but has optimizer state with param_groups
        checkpoint = {
            'optimizer': {
                'param_groups': [{'lr': 0.001}]
            }
        }
        self.assertTrue(self.manager.validate_optimizer_compatibility(checkpoint, 'Any'))

    def test_validate_optimizer_compatibility_invalid_checkpoint(self):
        self.assertFalse(self.manager.validate_optimizer_compatibility(None, 'Adam'))
        self.assertFalse(self.manager.validate_optimizer_compatibility({}, 'Adam'))

if __name__ == '__main__':
    unittest.main()
