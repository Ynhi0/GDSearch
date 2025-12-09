# Codebase Restructuring Plan

## Current Structure (Messy)
```
/workspaces/GDSearch/
 15+ .md files (scattered documentation)
 20+ .py files (mixed purpose)
 configs/, data/, plots/, results/, tests/
 PDF file (thesis registration)
```

## Target Structure (Clean)
```
/workspaces/GDSearch/
 src/
    core/              # Core implementations
       __init__.py
       optimizers.py
       test_functions.py
       models.py
       data_utils.py
    experiments/       # Experiment runners
       __init__.py
       run_experiment.py
       run_nn_experiment.py
       run_multi_seed.py
       run_full_analysis.py
    analysis/          # Analysis tools
       __init__.py
       statistical_analysis.py
       sensitivity_analysis.py
       ablation_study.py
    visualization/     # Plotting
        __init__.py
        plot_results.py
        plot_eigenvalues.py
        loss_landscape.py
 tests/
    __init__.py
    test_gradients.py
    test_optimizers.py
    test_validation.py
 configs/
    mnist_tuning.json
    cifar10_tuning.json
    test_functions.json
 scripts/               # Utility scripts
    run_all.py
    tune_nn.py
    generate_summaries.py
 docs/                  # Documentation
    README.md -> ../README.md
    LIMITATIONS.md
    MULTISEED_GUIDE.md
    QUICK_START.md
    CRITICAL_VALIDATION_REPORT.md
 data/                  # Datasets (gitignored)
 results/               # Experiment results (gitignored)
 plots/                 # Generated plots (gitignored)
 README.md              # Main readme
 requirements.txt
 .gitignore
 pyproject.toml         # Modern Python project config
```

## Migration Steps

### Phase 1: Move Core Modules
- [x] Create src/ structure
- [x] Move optimizers.py → src/core/
- [x] Move test_functions.py → src/core/
- [x] Move models.py → src/core/
- [x] Move data_utils.py → src/core/
- [x] Move validation.py → src/core/

### Phase 2: Move Experiment Scripts
- [x] Move run_experiment.py → src/experiments/
- [x] Move run_nn_experiment.py → src/experiments/
- [x] Move run_multi_seed.py → src/experiments/
- [x] Move run_full_analysis.py → src/experiments/

### Phase 3: Move Analysis Scripts
- [x] Move statistical_analysis.py → src/analysis/
- [x] Move sensitivity_analysis.py → src/analysis/
- [x] Move run_ablation_study.py → src/analysis/ablation_study.py
- [x] Move run_baseline_comparison.py → src/analysis/baseline_comparison.py

### Phase 4: Move Visualization
- [x] Move plot_results.py → src/visualization/
- [x] Move plot_eigenvalues.py → src/visualization/
- [x] Move loss_landscape.py → src/visualization/
- [x] Move run_loss_landscape.py → src/visualization/

### Phase 5: Move Utility Scripts
- [x] Move run_all.py → scripts/
- [x] Move tune_nn.py → scripts/
- [x] Move generate_summaries.py → scripts/
- [x] Move generate_appendix.py → scripts/
- [x] Move so_what_analysis.py → scripts/

### Phase 6: Consolidate Documentation
- [x] Keep README.md at root
- [x] Move important docs to docs/
- [x] Remove redundant/outdated .md files
- [x] Create docs/INDEX.md

### Phase 7: Update Imports
- [x] Create __init__.py with proper exports
- [ ] Fix all import statements in moved files
- [ ] Test all imports

### Phase 8: Clean Up
- [x] Remove __pycache__/
- [x] Remove .pytest_cache/
- [x] Remove duplicate files
- [x] Remove PDF (move to separate docs repo)

## Benefits

1. **Clear Separation of Concerns**: core → experiments → analysis → visualization
2. **Easier Navigation**: Logical grouping
3. **Better Imports**: `from src.core import optimizers`
4. **Professional Structure**: Follows Python best practices
5. **Scalability**: Easy to add new modules

## Breaking Changes

 All imports will change:
```python
# Before
from optimizers import SGD

# After
from src.core.optimizers import SGD
```

## Testing After Migration

```bash
# 1. Run all tests
pytest tests/ -v

# 2. Verify imports
python -c "from src.core import optimizers; print('')"

# 3. Run quick experiment
python scripts/run_all.py --quick
```

