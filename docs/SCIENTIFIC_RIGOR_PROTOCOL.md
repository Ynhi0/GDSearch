# Scientific Rigor Protocol

This protocol defines how experiments in GDSearch are conducted to ensure scientific validity, reproducibility, and fairness across optimizers and tasks.

## 1) Reproducibility & Environments
- Log environment details (Python, library versions, hardware): use `scripts/generate_appendix.py` (Section A)
- Pin dependencies in `requirements.txt` (Kaggle: `requirements_kaggle.txt`)
- Seed all RNGs (NumPy, PyTorch, CUDA) and enforce deterministic behavior where possible

## 2) Experimental Design
- Multi-seed runs: at least 3 seeds per optimizer per setting; prefer 5–10 for publication
- Baseline → Ablation Ladder:
  - SGD → SGD+Momentum → RMSProp → Adam → AdamW → AMSGrad
- Convergence detection (when applicable): dual criteria
  - Gradient norm threshold: default 1e-6
  - Loss delta over window: default 1e-7 over 200 steps
  - Configurable per-run via `convergence_*` or `convergence.{grad_norm_threshold,loss_delta_threshold,loss_window}`
- Telemetry:
  - Wall-clock time per run (seconds)
  - Peak GPU memory per run (MiB)
  - Update norm and gradient norm (per step) when available

## 3) Datasets & Models
- 2D test functions: Rosenbrock, IllConditionedQuadratic, SaddlePoint, Ackley2D
- NN baselines: MNIST (MLP), CIFAR-10 (small CNN)
- Applications: NLP (Transformer fine-tuning), Medical imaging (3D U-Net)
- Keep architecture and training schedule minimal-yet-representative; document all hyperparameters

## 4) Metrics & Logging
- Per-step/per-epoch logs: loss, accuracy, grad_norm, update_norm, time_sec
- Periodic layer-wise gradient norms (for dynamics and gradient heterogeneity)
- Save CSVs under `results/` with consistent naming: `NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>[_tag].csv`

## 5) Statistical Analysis
- Normality checks (Shapiro–Wilk); choose tests accordingly:
  - Paired t-test if normal; Wilcoxon signed-rank otherwise
- Multiple comparisons: Holm–Bonferroni correction
- Effect sizes: Cohen’s d (parametric) or rank-biserial r (non-parametric)
- Power analysis to assess adequacy of sample size
- Tools:
  - Generic utilities: `src/analysis/statistical_analysis.py`
  - MNIST report: `scripts/generate_statistical_report.py`
  - CIFAR-10 report: `scripts/generate_cifar10_statistical_report.py`

## 6) Trade-off Analyses
- Accuracy vs. Wall-clock Time scatter and Pareto frontier
- Accuracy vs. Peak GPU Memory table
- Script: `scripts/compute_tradeoffs.py` → outputs to `plots/` and summary CSVs

## 7) Visualization & Publication Artifacts
- 2D trajectories and curvature (eigenvalues)
- Error-bar plots with individual points for NN results
- Publication tables (LaTeX/Excel): `scripts/generate_latex_tables.py`

## 8) Reporting & Appendices
- Outline mapping: `docs/OUTLINE_MAPPING.md`
- Practitioner handbook with decision heuristics: `docs/PRACTITIONER_HANDBOOK.md`
- Appendix with full environment and hyperparameters: `scripts/generate_appendix.py`

## 9) Kaggle Readiness
- Standalone scripts: `kaggle/mnist_publication/`, `kaggle/cifar10_publication/`
- Notebooks for figures, NLP and medical case studies
- Save outputs to `/kaggle/working/results`

## 10) Quality Gates
- Run unit tests (`pytest -v -m "not slow"`) before merging
- Ensure no syntax/type errors; keep public APIs stable
- Add targeted tests for new math/optimizer code
