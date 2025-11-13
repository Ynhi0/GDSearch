# Outline → Repository Mapping (Extended, Application-focused)

This document maps your extended research outline to concrete code, data, and scripts in this repository.

## Ch. 1–2: Introduction & Background
- Literature and theory are summarized in: `docs/INDEX.md`, `docs/LIMITATIONS.md`, `docs/PHASE14_STATISTICAL_SUMMARY.md`.
- Contribution summary and project scope: `README.md`, `FINAL_SUMMARY.md`.

## Ch. 3: Methodology
- 2D Test Functions: `src/core/test_functions.py` (Rosenbrock, IllConditionedQuadratic, SaddlePoint, Ackley)
- Optimizers (engine-first): `src/core/optimizers.py` and PyTorch wrappers in `src/core/pytorch_optimizers.py`
- 2D Experiments: `src/experiments/run_experiment.py`, `src/experiments/run_initial_condition_robustness.py`
- NN Baselines (MNIST): `kaggle/mnist_publication/mnist_publication.py` (Kaggle-ready, multi-seed)
- Ablation Design (SGD → SGDM → RMSProp → Adam → AdamW → AMSGrad):
  - 2D: `src/experiments/run_optimizer_ablation.py`
  - NN: `scripts/run_nn_ablation.py` (aggregates multi-seed MNIST runs)
- Application Case Studies (scaffolds):
  - NLP Transformer fine-tuning: `src/experiments/run_transformer_nlp.py`
  - Medical imaging segmentation (U-Net/MONAI): `src/experiments/run_medical_segmentation.py`

## Statistical Rigor
- Protocol: `docs/SCIENTIFIC_RIGOR_PROTOCOL.md`
- Statistical utilities: `src/analysis/statistical_analysis.py` (paired/independent tests, corrections, power)
- Paired tests + Holm–Bonferroni in pipeline: `src/experiments/run_full_analysis.py`
- MNIST publication stats: computed in `mnist_publication.py` and enriched by `scripts/generate_statistical_report.py`
- CIFAR-10 stats: `scripts/generate_cifar10_statistical_report.py` or Kaggle script output

## Ch. 4: Results & Discussion
- 2D kinetic analyses and visualizations: `src/visualization/plot_results.py`, `src/visualization/plot_eigenvalues.py`
- MNIST results: saved in `results/NN_SimpleMLP_MNIST_*_publication.csv`
- Ablation outputs:
  - 2D: `results/optimizer_ablation_summary.csv`, `plots/optimizer_ablation_study.png`
  - NN: `results/nn_ablation_summary.csv`, `plots/nn_ablation_*.png`
- Statistical comparisons: `results/nn_statistical_comparisons.csv`, `results/nn_statistical_report.md`
- Publication tables/Excel: `scripts/generate_latex_tables.py` (produces `.tex` and `.xlsx`)

## Ch. 5: Conclusions & Practice Guide
- Summaries and guidance drafts: `results/EXPERIMENT_REPORT.md`, `docs/PHASE14_STATISTICAL_SUMMARY.md`
- Practitioner handbook: `docs/PRACTITIONER_HANDBOOK.md`
- Actionable outputs for practitioners:
  - Error-bar plots: see `plots/`
  - Tables and Excel with highlights: `results/table_*.tex`, `results/table_*.xlsx`

## Reproducibility & Kaggle
- Kaggle-ready experiments: `kaggle/mnist_publication/`, `kaggle/cifar10_publication/`
- End-to-end pipeline: `scripts/run_all.py` (includes hooks for CIFAR-10, ablation, tables)

## What’s Missing / Next Steps
- Execute application case studies on real datasets (NLP, medical imaging) at larger scale and add results.
- Publication figures for application case studies (layer grad heterogeneity, qualitative segmentation visuals).
- Optional: larger models (ResNet-18 on CIFAR-10, DistilBERT) and longer training for stronger claims.
-- Trade-offs: use `scripts/compute_tradeoffs.py` to generate Accuracy vs Time/Memory plots.
