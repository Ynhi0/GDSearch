# Documentation Index

Welcome to GDSearch documentation! This guide helps you navigate all available documents.

---

## 📚 Essential Reading (Start Here)

### For New Users:
1. **[README.md](../README.md)** - Project overview, installation, quick start
2. **[QUICK_START.md](QUICK_START.md)** - Step-by-step tutorial with examples
3. **[MULTISEED_GUIDE.md](MULTISEED_GUIDE.md)** - How to run statistically valid experiments

### For Researchers:
1. **[CRITICAL_VALIDATION_REPORT.md](CRITICAL_VALIDATION_REPORT.md)** - Scientific validation and methodology
2. **[LIMITATIONS.md](LIMITATIONS.md)** - Known limitations and assumptions
3. **[RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md)** - Experimental insights and findings

### For Developers:
1. **Code in `src/`** - Well-organized, documented source code
2. **Tests in `tests/`** - Unit test examples
3. **[IMPROVEMENT_PROGRESS.md](IMPROVEMENT_PROGRESS.md)** - Development history

---

## 📖 Documentation by Category

### 🎯 Getting Started
| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Project overview | Everyone |
| [QUICK_START.md](QUICK_START.md) | Quick tutorial | New users |
| [requirements.txt](../requirements.txt) | Dependencies | Setup |
| [pyproject.toml](../pyproject.toml) | Project config | Developers |

### 🔬 Scientific Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [CRITICAL_VALIDATION_REPORT.md](CRITICAL_VALIDATION_REPORT.md) | Validation & assessment | Researchers |
| [LIMITATIONS.md](LIMITATIONS.md) | Known issues | Researchers |
| [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) | Experimental insights | Researchers |
| [MULTISEED_GUIDE.md](MULTISEED_GUIDE.md) | Statistical methodology | Researchers |

### 💻 Developer Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [IMPROVEMENT_PROGRESS.md](IMPROVEMENT_PROGRESS.md) | Development history | Developers |
| [pyproject.toml](../pyproject.toml) | Build configuration | Developers |
| Source code in `src/` | Implementation | Developers |
| Tests in `tests/` | Test examples | Developers |

---

## 🗂️ Project Structure

```
/workspaces/GDSearch/
├── src/                          # Source code (organized by purpose)
│   ├── core/                     # Core implementations
│   │   ├── optimizers.py        # SGD, Adam, RMSProp, AdamW
│   │   ├── test_functions.py    # Rosenbrock, etc.
│   │   ├── models.py            # Neural network models
│   │   ├── data_utils.py        # MNIST/CIFAR-10 loaders
│   │   └── validation.py        # Input validation
│   ├── experiments/              # Experiment runners
│   │   ├── run_experiment.py    # Single experiment
│   │   ├── run_nn_experiment.py # Neural network exp
│   │   ├── run_multi_seed.py    # Multi-seed framework
│   │   └── run_full_analysis.py # Full pipeline
│   ├── analysis/                 # Analysis tools
│   │   ├── statistical_analysis.py  # T-tests, CI
│   │   ├── ablation_study.py        # Component analysis
│   │   ├── baseline_comparison.py   # PyTorch comparison
│   │   └── sensitivity_analysis.py  # Sensitivity study
│   └── visualization/            # Plotting
│       ├── plot_results.py      # Result plots
│       ├── plot_eigenvalues.py  # Eigenvalue plots
│       └── loss_landscape.py    # Loss surface viz
├── tests/                        # Unit tests
│   ├── test_gradients.py        # Gradient verification
│   ├── test_optimizers.py       # Optimizer correctness
│   └── test_validation.py       # Validation tests
├── configs/                      # Experiment configs
│   ├── mnist_tuning.json
│   └── cifar10_tuning.json
├── scripts/                      # Utility scripts
│   ├── run_all.py               # Run full pipeline
│   ├── tune_nn.py               # Hyperparameter tuning
│   └── generate_summaries.py    # Result summaries
├── docs/                         # Documentation (you are here!)
│   ├── INDEX.md                 # This file
│   ├── README.md                # Symlink to main README
│   ├── QUICK_START.md
│   ├── MULTISEED_GUIDE.md
│   ├── LIMITATIONS.md
│   ├── CRITICAL_VALIDATION_REPORT.md
│   ├── RESEARCH_JOURNAL.md
│   └── IMPROVEMENT_PROGRESS.md
├── data/                         # Datasets (auto-downloaded)
├── results/                      # Experiment results (gitignored)
├── plots/                        # Generated plots (gitignored)
├── README.md                     # Main readme
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Modern Python config
└── .gitignore                    # Git ignore rules
```

---

## 🎓 Learning Path

### Path 1: Quick User (30 minutes)
1. Read [README.md](../README.md) overview
2. Install dependencies: `pip install -r requirements.txt`
3. Run quick test: `pytest tests/ -v`
4. Try example: `python scripts/run_all.py --quick`

### Path 2: Researcher (2-3 hours)
1. Read [QUICK_START.md](QUICK_START.md) tutorial
2. Read [MULTISEED_GUIDE.md](MULTISEED_GUIDE.md) for methodology
3. Run multi-seed experiment: `python src/experiments/run_full_analysis.py --seeds 1,2,3`
4. Read [CRITICAL_VALIDATION_REPORT.md](CRITICAL_VALIDATION_REPORT.md) for validation
5. Consult [LIMITATIONS.md](LIMITATIONS.md) for scope

### Path 3: Developer (1 day)
1. Read all "Quick User" + "Researcher" docs
2. Study code structure in `src/`
3. Read tests in `tests/` for examples
4. Read [IMPROVEMENT_PROGRESS.md](IMPROVEMENT_PROGRESS.md) for context
5. Review [pyproject.toml](../pyproject.toml) for project setup

---

## 📊 Key Features Documentation

### Multi-Seed Experiments
- **Guide**: [MULTISEED_GUIDE.md](MULTISEED_GUIDE.md)
- **Code**: `src/experiments/run_multi_seed.py`
- **Example**: `python src/experiments/run_full_analysis.py --seeds 1,2,3,4,5`

### Statistical Analysis
- **Guide**: [MULTISEED_GUIDE.md](MULTISEED_GUIDE.md#statistical-comparison)
- **Code**: `src/analysis/statistical_analysis.py`
- **Functions**: `compare_optimizers_ttest()`, `print_ttest_results()`

### Unit Testing
- **Tests**: `tests/test_gradients.py`, `tests/test_optimizers.py`
- **Run**: `pytest tests/ -v`
- **Coverage**: All gradients & optimizers verified numerically

### Ablation Study
- **Code**: `src/analysis/ablation_study.py`
- **Run**: `python src/analysis/ablation_study.py`
- **Purpose**: Isolate effect of each optimizer component

### Baseline Comparison
- **Code**: `src/analysis/baseline_comparison.py`
- **Run**: `python src/analysis/baseline_comparison.py`
- **Purpose**: Validate against PyTorch implementations

---

## 🔍 Finding Information

### Want to...
| Goal | Document | Location |
|------|----------|----------|
| Get started quickly | [QUICK_START.md](QUICK_START.md) | docs/ |
| Run multi-seed experiments | [MULTISEED_GUIDE.md](MULTISEED_GUIDE.md) | docs/ |
| Understand methodology | [CRITICAL_VALIDATION_REPORT.md](CRITICAL_VALIDATION_REPORT.md) | docs/ |
| Know limitations | [LIMITATIONS.md](LIMITATIONS.md) | docs/ |
| See experimental insights | [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) | docs/ |
| Understand code | Source files in `src/` | src/ |
| Run tests | Test files in `tests/` | tests/ |
| Use configs | Config files | configs/ |

### Common Tasks
| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Test | `pytest tests/ -v` |
| Quick run | `python scripts/run_all.py --quick` |
| Multi-seed | `python src/experiments/run_full_analysis.py --seeds 1,2,3,4,5` |
| Ablation | `python src/analysis/ablation_study.py` |
| Baseline | `python src/analysis/baseline_comparison.py` |

---

## 📞 Getting Help

1. **Check this INDEX** - Find the right document
2. **Read QUICK_START.md** - Step-by-step tutorial
3. **Consult LIMITATIONS.md** - Known issues
4. **Run tests** - `pytest tests/ -v` to verify setup
5. **Check examples** - Look at `if __name__ == '__main__'` blocks in code

---

## 🔄 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| README.md | ✅ Current | Session 2.0 |
| QUICK_START.md | ✅ Current | Session 2.0 |
| MULTISEED_GUIDE.md | ✅ Current | Session 2.0 |
| CRITICAL_VALIDATION_REPORT.md | ✅ Current | Session 2.0 |
| LIMITATIONS.md | ✅ Current | Session 2.0 |
| RESEARCH_JOURNAL.md | ✅ Current | Session 2.0 |
| IMPROVEMENT_PROGRESS.md | ✅ Current | Session 2.0 |
| pyproject.toml | ✅ Current | Session 2.0 |

---

## 📝 Contributing to Documentation

When adding new documentation:
1. Place file in appropriate location (`docs/` for user docs)
2. Update this INDEX.md with link
3. Add to relevant section above
4. Update "Document Status" table
5. Follow existing formatting style

---

**Navigation**: [Back to README](../README.md) | [Quick Start](QUICK_START.md) | [Limitations](LIMITATIONS.md)

**Version**: 2.0.0  
**Last Updated**: Current Session  
**Maintainer**: GDSearch Team
