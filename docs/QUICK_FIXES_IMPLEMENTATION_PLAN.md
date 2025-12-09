# Quick Fixes Implementation Plan
**Immediate Remediation for Critical Issues**  
*Date: December 9, 2025*

---

## 🎯 Goal

Implement **immediate, minimal-change fixes** for the 3 BLOCKER issues identified in the research validity audit. Each fix is designed to:
- Take ≤4 hours to implement
- Require minimal code changes
- Include verification tests
- Be safe to deploy immediately

---

## 🚨 BLOCKER-1: Fix Adaptive Overfitting (test_loader in tuning)

**Estimated Time:** 2 hours  
**Files to Edit:** 1  
**Risk Level:** LOW (rename + assertion)

### Changes Required

#### File: `run_all_kaggle.py`

**Change 1: Rename loader parameter (lines ~2050–2070)**
```python
# OLD
def create_tuning_objective(model_class, train_loader, test_loader, device, config):
    def objective(trial):
        # ... setup code ...
        for inputs, targets in test_loader:  # ← DANGER
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # ...

# NEW
def create_tuning_objective(model_class, train_loader, val_loader, device, config):
    """Create Optuna objective function.
    
    CRITICAL: Must use validation split for tuning, never test split.
    """
    # Add assertion
    assert val_loader is not None, "Validation loader required for tuning"
    
    def objective(trial):
        # ... setup code ...
        for inputs, targets in val_loader:  # ← FIXED
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # ...
```

**Change 2: Add runtime safety check (insert after function definition)**
```python
def create_tuning_objective(model_class, train_loader, val_loader, device, config):
    # SAFETY: Prevent test split leakage
    if 'test' in str(type(val_loader).__name__).lower():
        raise RuntimeError(
            "BLOCKER: Test loader passed to tuning objective. "
            "This constitutes adaptive overfitting. "
            "Use validation split only for hyperparameter tuning."
        )
    
    def objective(trial):
        # ... rest of function
```

**Change 3: Update all callsites (search for `create_tuning_objective`)**
```python
# OLD
objective = create_tuning_objective(model_class, train_loader, test_loader, device, config)

# NEW
objective = create_tuning_objective(model_class, train_loader, val_loader, device, config)
```

### Verification Steps

1. **Search and replace verification:**
   ```powershell
   # Verify no test_loader in tuning objectives
   Select-String -Path "run_all_kaggle.py" -Pattern "test_loader" -Context 2,2
   # Expected: 0 matches in tuning objective functions
   ```

2. **Run quick test:**
   ```python
   # tests/test_tuning_safety.py
   def test_tuning_objective_refuses_test_loader():
       """Verify tuning objective rejects test loaders."""
       from run_all_kaggle import create_tuning_objective
       
       # Create mock loaders
       train_loader = MagicMock(spec=DataLoader)
       test_loader = MagicMock(spec=DataLoader)
       type(test_loader).__name__ = 'TestDataLoader'  # ← Trigger check
       
       with pytest.raises(RuntimeError, match="adaptive overfitting"):
           create_tuning_objective(
               model_class=SimpleCNN,
               train_loader=train_loader,
               val_loader=test_loader,  # ← Should fail
               device='cpu',
               config={}
           )
   ```

3. **Integration test:**
   ```bash
   # Run a 1-trial tuning job
   python run_all_kaggle.py --experiments mnist --trials 1 --seeds 42
   # Expected: Completes without errors, uses val_loader
   ```

---

## 🚨 BLOCKER-2: Complete Checkpoint State

**Estimated Time:** 4 hours  
**Files to Edit:** 2  
**Risk Level:** MEDIUM (modifies checkpoint format)

### Changes Required

#### File: `run_all_kaggle.py` (checkpoint save section)

**Change 1: Extend checkpoint dictionary (lines ~2600–2650)**
```python
# OLD
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    },
    'loss': loss
}

# NEW
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    
    # ========== ADDED: Complete training state ==========
    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    'scaler_state_dict': scaler.state_dict() if hasattr(self, 'scaler') and scaler is not None else None,
    'ema_state_dict': ema.shadow.state_dict() if hasattr(self, 'ema') and ema is not None else None,
    # ====================================================
    
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    },
    
    # ========== ADDED: Metadata for verification ==========
    'metadata': {
        'current_lr': optimizer.param_groups[0]['lr'],
        'completed': epoch >= config.get('epochs', 100),
        'training_step': getattr(self, 'global_step', 0),
        'best_val_loss': getattr(self, 'best_val_loss', float('inf'))
    },
    # ======================================================
    
    'loss': loss
}
```

**Change 2: Extend checkpoint restore (lines ~2450–2480)**
```python
# OLD
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore RNG states
random.setstate(checkpoint['rng_states']['python'])
np.random.set_state(checkpoint['rng_states']['numpy'])
torch.set_rng_state(checkpoint['rng_states']['torch'])
if checkpoint['rng_states']['cuda'] is not None:
    torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])

start_epoch = checkpoint['epoch'] + 1

# NEW
# Load core state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ========== ADDED: Restore complete training state ==========
if checkpoint.get('scheduler_state_dict') and scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"✓ Restored scheduler state (last_epoch={scheduler.last_epoch})")

if checkpoint.get('scaler_state_dict') and hasattr(self, 'scaler') and scaler is not None:
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    print(f"✓ Restored AMP scaler state (scale={scaler.get_scale()})")

if checkpoint.get('ema_state_dict') and hasattr(self, 'ema') and ema is not None:
    ema.shadow.load_state_dict(checkpoint['ema_state_dict'])
    print(f"✓ Restored EMA shadow weights")
# ============================================================

# Restore RNG states
random.setstate(checkpoint['rng_states']['python'])
np.random.set_state(checkpoint['rng_states']['numpy'])
torch.set_rng_state(checkpoint['rng_states']['torch'])
if checkpoint['rng_states']['cuda'] is not None:
    torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])

# ========== ADDED: Restore metadata and verify ==========
metadata = checkpoint.get('metadata', {})
if metadata.get('completed'):
    print(f"⚠ Experiment already completed at epoch {checkpoint['epoch']}")
    return None  # Signal to skip

self.global_step = metadata.get('training_step', 0)
self.best_val_loss = metadata.get('best_val_loss', float('inf'))
print(f"✓ Restored metadata: LR={metadata.get('current_lr')}, step={self.global_step}")
# ========================================================

start_epoch = checkpoint['epoch'] + 1
```

### Verification Steps

1. **Unit test: Save and restore equivalence**
   ```python
   # tests/test_checkpoint.py
   def test_checkpoint_complete_state():
       """Verify all training state saved and restored."""
       from run_all_kaggle import ExperimentRunner
       
       # Setup
       runner = ExperimentRunner(config)
       runner.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(runner.optimizer, T_max=100)
       runner.scaler = torch.cuda.amp.GradScaler()
       
       # Train for 3 epochs
       runner.train(epochs=3)
       initial_lr = runner.optimizer.param_groups[0]['lr']
       
       # Save checkpoint
       runner.save_checkpoint('test_checkpoint.pt', epoch=2)
       
       # Create new runner and restore
       new_runner = ExperimentRunner(config)
       new_runner.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_runner.optimizer, T_max=100)
       new_runner.load_checkpoint('test_checkpoint.pt')
       
       # Verify state
       assert new_runner.optimizer.param_groups[0]['lr'] == initial_lr, "LR mismatch"
       assert new_runner.scheduler.last_epoch == 2, "Scheduler epoch mismatch"
       assert new_runner.scaler.get_scale() > 0, "Scaler not restored"
   ```

2. **Integration test: Interrupt and resume**
   ```python
   def test_interrupt_resume_equivalence():
       """Verify interrupted training produces identical results on resume."""
       # Run 1: Train for 10 epochs straight
       results_continuous = train_model(epochs=10, seed=42)
       
       # Run 2: Train for 5 epochs, checkpoint, resume for 5 more
       results_interrupted = train_model(epochs=5, seed=42)
       checkpoint_path = save_checkpoint()
       results_resumed = resume_training(checkpoint_path, additional_epochs=5)
       
       # Verify equivalence (within numerical tolerance)
       np.testing.assert_allclose(
           results_continuous['val_loss'][-1],
           results_resumed['val_loss'][-1],
           rtol=1e-5,
           err_msg="Resume altered training dynamics"
       )
   ```

3. **Manual verification (Kaggle notebook)**
   ```python
   # In run_benchmark.ipynb
   # 1. Run experiment for 5 epochs
   # 2. Copy checkpoint to Input Dataset
   # 3. Restart kernel
   # 4. Resume from checkpoint
   # 5. Verify epoch counter continues from 5 → 6
   # 6. Verify learning rate matches expected scheduler value
   ```

---

## 🚨 BLOCKER-3: JSON Schema Validation

**Estimated Time:** 3 hours  
**Files to Edit:** 5 (1 schema + 3 configs + 1 validator)  
**Risk Level:** LOW (validation only)

### Changes Required

#### File 1: Create schema definition

**New file: `configs/config_schema.json`**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GDSearch Experiment Configuration Schema",
  "type": "object",
  "required": ["sweeps"],
  "properties": {
    "sweeps": {
      "type": "object",
      "description": "Hyperparameter search spaces per optimizer",
      "patternProperties": {
        "^(SGD|Adam|AdamW|RMSprop|Adagrad|Adadelta|Lookahead|SAM|Lion)$": {
          "type": "object",
          "properties": {
            "learning_rate": {
              "type": "array",
              "items": {"type": "number", "exclusiveMinimum": 0},
              "minItems": 1
            },
            "weight_decay": {
              "type": "array",
              "items": {"type": "number", "minimum": 0}
            },
            "momentum": {
              "type": "array",
              "items": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "betas": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2
              }
            },
            "rho": {
              "type": "array",
              "items": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "adaptive": {
              "type": "array",
              "items": {"type": "boolean"}
            }
          },
          "required": ["learning_rate"],
          "additionalProperties": false
        }
      },
      "additionalProperties": false
    },
    "common": {
      "type": "object",
      "description": "Common hyperparameters (optional)",
      "properties": {
        "batch_size": {"type": "integer", "minimum": 1},
        "epochs": {"type": "integer", "minimum": 1}
      }
    }
  },
  "additionalProperties": false
}
```

#### File 2: Create validator script

**New file: `scripts/validate_configs.py`**
```python
#!/usr/bin/env python3
"""Validate all experiment configs against JSON schema."""

import json
import sys
from pathlib import Path
import jsonschema

def validate_config(config_path, schema_path):
    """Validate single config file."""
    with open(schema_path) as f:
        schema = json.load(f)
    
    with open(config_path) as f:
        config = json.load(f)
    
    try:
        jsonschema.validate(config, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)

def main():
    schema_path = Path(__file__).parent.parent / 'configs' / 'config_schema.json'
    configs_dir = Path(__file__).parent.parent / 'configs'
    
    config_files = [
        configs_dir / 'nn_tuning.json',
        configs_dir / 'cifar10_tuning.json',
        configs_dir / 'benchmark_hyperparameters.json'
    ]
    
    all_valid = True
    for config_file in config_files:
        if not config_file.exists():
            print(f"⚠ {config_file.name}: Not found (skipping)")
            continue
        
        valid, error = validate_config(config_file, schema_path)
        if valid:
            print(f"✓ {config_file.name}: Valid")
        else:
            print(f"✗ {config_file.name}: INVALID")
            print(f"  Error: {error}")
            all_valid = False
    
    if not all_valid:
        print("\n❌ Config validation FAILED")
        sys.exit(1)
    else:
        print("\n✅ All configs valid")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

#### File 3: Fix config files

**Update: `configs/nn_tuning.json`** (example — apply pattern to all configs)
```json
{
  "sweeps": {
    "SGD": {
      "learning_rate": [0.001, 0.01, 0.1],
      "weight_decay": [0.0, 1e-4, 1e-3],
      "momentum": [0.0, 0.9]
    },
    "Adam": {
      "learning_rate": [0.0001, 0.001, 0.01],
      "weight_decay": [0.0, 1e-4],
      "betas": [[0.9, 0.999]]
    },
    "AdamW": {
      "learning_rate": [0.0001, 0.001, 0.01],
      "weight_decay": [0.01, 0.1]
    }
  },
  "common": {
    "batch_size": 128,
    "epochs": 50
  }
}
```

**Key changes:**
- `weight_decay_values` → `weight_decay` (standardized)
- `lr_values` → `learning_rate` (standardized)
- Remove unknown keys like `sweep_params`

#### File 4: Add CI check

**New file: `.github/workflows/validate-configs.yml`**
```yaml
name: Validate Experiment Configs

on:
  push:
    paths:
      - 'configs/*.json'
      - 'configs/config_schema.json'
      - 'scripts/validate_configs.py'
  pull_request:
    paths:
      - 'configs/*.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install jsonschema
      
      - name: Validate configs
        run: python scripts/validate_configs.py
```

### Verification Steps

1. **Manual validation:**
   ```powershell
   # Run validator locally
   python scripts/validate_configs.py
   # Expected: All configs pass
   ```

2. **Test with intentional error:**
   ```json
   // Temporarily add unknown key to nn_tuning.json
   {
     "sweeps": {
       "SGD": {
         "learning_rate": [0.1],
         "unknown_param": [1, 2, 3]  // ← Should fail
       }
     }
   }
   ```
   ```powershell
   python scripts/validate_configs.py
   # Expected: INVALID - additionalProperties: false
   ```

3. **CI test:**
   - Push changes to GitHub
   - Verify workflow runs and passes
   - Intentionally break a config, verify workflow fails

---

## 📋 Implementation Checklist

### Pre-implementation
- [ ] Create feature branch: `git checkout -b fix/blocker-issues`
- [ ] Backup current configs: `cp -r configs configs.backup`
- [ ] Run existing tests to establish baseline: `pytest tests/`

### BLOCKER-1 (2 hours)
- [ ] Rename `test_loader` → `val_loader` in `run_all_kaggle.py`
- [ ] Add runtime assertion for loader type check
- [ ] Update all callsites (search for `create_tuning_objective`)
- [ ] Write unit test `test_tuning_safety.py`
- [ ] Run integration test: `python run_all_kaggle.py --experiments mnist --trials 1`
- [ ] Commit: `git commit -m "fix: prevent test loader in tuning (BLOCKER-1)"`

### BLOCKER-2 (4 hours)
- [ ] Extend checkpoint save dictionary (scheduler/scaler/EMA)
- [ ] Extend checkpoint restore logic
- [ ] Add metadata (current_lr, completed flag)
- [ ] Write unit test `test_checkpoint.py::test_checkpoint_complete_state`
- [ ] Write integration test `test_checkpoint.py::test_interrupt_resume_equivalence`
- [ ] Test Kaggle resume workflow manually
- [ ] Commit: `git commit -m "fix: complete checkpoint state (BLOCKER-2)"`

### BLOCKER-3 (3 hours)
- [ ] Create `configs/config_schema.json`
- [ ] Create `scripts/validate_configs.py`
- [ ] Update `configs/nn_tuning.json` (fix key names)
- [ ] Update `configs/cifar10_tuning.json`
- [ ] Update `configs/benchmark_hyperparameters.json`
- [ ] Run validator: `python scripts/validate_configs.py`
- [ ] Create CI workflow `.github/workflows/validate-configs.yml`
- [ ] Test CI locally or in PR
- [ ] Commit: `git commit -m "fix: add JSON schema validation (BLOCKER-3)"`

### Post-implementation
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Run quick validation: `python scripts/quick_validation_test.py`
- [ ] Update `CRITICAL_ISSUES_TRACKER.md` (mark blockers as fixed)
- [ ] Create PR with detailed description
- [ ] Request review from team

---

## 🧪 Testing Summary

### Unit Tests (3 new)
| Test | File | Purpose |
|------|------|---------|
| `test_tuning_objective_refuses_test_loader` | `test_tuning_safety.py` | Verify tuning rejects test loaders |
| `test_checkpoint_complete_state` | `test_checkpoint.py` | Verify scheduler/scaler/EMA saved |
| `test_interrupt_resume_equivalence` | `test_checkpoint.py` | Verify resume produces identical results |

### Integration Tests (2 new)
| Test | Command | Purpose |
|------|---------|---------|
| Quick tuning job | `python run_all_kaggle.py --experiments mnist --trials 1` | Verify tuning uses val_loader |
| Config validation | `python scripts/validate_configs.py` | Verify all configs pass schema |

### Manual Tests (1)
| Test | Location | Purpose |
|------|----------|---------|
| Kaggle resume | `kaggle/run_benchmark.ipynb` | Verify checkpoint restore in Kaggle environment |

---

## 📊 Expected Outcomes

### After BLOCKER-1 Fix
- ✅ No `test_loader` in tuning objective functions
- ✅ Runtime assertion prevents accidental test split usage
- ✅ CI lint check catches future violations

### After BLOCKER-2 Fix
- ✅ Checkpoints include scheduler/scaler/EMA state
- ✅ Resumed training continues with correct learning rate
- ✅ Gradient scaling and EMA preserved across resume
- ✅ Training curves identical between continuous and interrupted runs

### After BLOCKER-3 Fix
- ✅ All configs validated against schema
- ✅ Unknown keys fail CI immediately
- ✅ Standardized key names across all configs
- ✅ Documented canonical schema for future configs

---

## 🚨 Rollback Plan

If any fix causes issues:

```powershell
# Restore backup configs
cp -r configs.backup/* configs/

# Revert checkpoint changes
git checkout HEAD~1 run_all_kaggle.py

# Revert tuning changes
git checkout HEAD~1 run_all_kaggle.py

# Full revert
git revert HEAD
```

**Rollback triggers:**
- Checkpoint restore fails on existing checkpoints
- Config validation breaks existing experiments
- Integration tests fail consistently

---

## 📞 Support & Questions

If you encounter issues during implementation:

1. **Check test output:** `pytest tests/ -v --tb=short`
2. **Verify file changes:** `git diff`
3. **Run validation:** `python scripts/quick_validation_test.py`
4. **Check issue tracker:** `docs/CRITICAL_ISSUES_TRACKER.md`

---

**Estimated Total Time:** 9 hours  
**Recommended Schedule:** Implement over 2 days (1 blocker per session)  
**Target Completion:** End of Week 1

---

**Last Updated:** December 9, 2025  
**Status:** Ready for implementation
