# Proposal Expectations Mapping

- Generated: 2026-03-14T11:04:59+00:00
- Results root: `results_longrun_20260314\experiments`
- Source docs:
  - `docs/CONVERGENCE_CRITERIA.md`
  - `docs/METRICS_HIERARCHY.md`
  - `docs/proposal.txt`

## Convergence/Metric Rules Applied
- Convergence criterion for optimizer-level summary plots uses stationarity-or-loss: (||grad|| < t) OR (loss < t).
- Practical threshold is t = 1e-3 and strict threshold is t = 1e-6 for 2D/robustness/ablation families.
- Convergence-profile denominator uses all attempted runs (error/non-finite rows count as non-converged).
- Error rows (run_status starts with 'error' or non-empty error column) are always forced non-converged.
- Convergence-rate analysis is based on optimization signals (loss/grad), not test accuracy curves.

## Experiment Mapping

### 2d_optimization
- Generator: `run_all_kaggle.py::_create_2d_visualizations`
- Summary CSV: `experiments/2d_optimization/2d_optimization_results.csv`
- Expected unique seeds: `10`
- Required columns: `optimizer, function, seed, final_loss, iterations, run_status, error, converged_strict, converged_practical`
- Required visualization artifacts:
  - `visualizations/static/2d_optimization/2d_rastrigin_loss_convergence.png`
  - `visualizations/static/2d_optimization/2d_rastrigin_grad_norm_convergence.png`
  - `visualizations/static/2d_optimization/2d_rastrigin_trajectory_overlay.png`
  - `visualizations/static/2d_optimization/2d_rosenbrock_loss_convergence.png`
  - `visualizations/static/2d_optimization/2d_rosenbrock_grad_norm_convergence.png`
  - `visualizations/static/2d_optimization/2d_rosenbrock_trajectory_overlay.png`
  - `visualizations/static/2d_optimization/2d_final_loss_by_optimizer.png`
  - `visualizations/static/2d_optimization/2d_convergence_rate_by_optimizer.png`
  - `visualizations/static/2d_optimization/2d_convergence_rate_strict_by_optimizer.png`
  - `visualizations/static/2d_optimization/2d_convergence_profile_by_threshold.png`

### ablation
- Generator: `run_all_kaggle.py::_create_ablation_visualizations`
- Summary CSV: `experiments/ablation/ablation_results.csv`
- Expected unique seeds: `10`
- Required columns: `optimizer, seed, final_loss, iterations, run_status, error, converged_strict, converged_practical`
- Required visualization artifacts:
  - `visualizations/static/ablation/ablation_loss_convergence.png`
  - `visualizations/static/ablation/ablation_final_loss_by_optimizer.png`
  - `visualizations/static/ablation/ablation_iterations_by_optimizer.png`
  - `visualizations/static/ablation/ablation_convergence_rate.png`
  - `visualizations/static/ablation/ablation_convergence_profile.png`

### beta_sensitivity_2d
- Generator: `run_beta_2d_demos.py::main / generate_*`
- Required visualization artifacts:
  - `experiments/beta_sensitivity_2d/rosenbrock/momentum/momentum_trajectories.png`
  - `experiments/beta_sensitivity_2d/rosenbrock/momentum/momentum_metrics.png`
  - `experiments/beta_sensitivity_2d/saddle_point/adam/adam_heatmaps.png`

### robustness
- Generator: `run_all_kaggle.py::_create_robustness_visualizations`
- Summary CSV: `experiments/robustness/robustness_results.csv`
- Expected unique seeds: `10`
- Required columns: `optimizer, seed, start_point, final_loss, iterations, run_status, error, converged, converged_iteration`
- Required visualization artifacts:
  - `visualizations/static/robustness/robustness_loss_convergence.png`
  - `visualizations/static/robustness/robustness_grad_norm_convergence.png`
  - `visualizations/static/robustness/robustness_start_point_sensitivity.png`
  - `visualizations/static/robustness/robustness_final_loss_by_optimizer.png`
  - `visualizations/static/robustness/robustness_convergence_rate.png`
  - `visualizations/static/robustness/robustness_convergence_rate_strict.png`
  - `visualizations/static/robustness/robustness_convergence_profile.png`

### sam_sensitivity
- Generator: `run_all_kaggle.py::_create_sam_sensitivity_visualizations`
- Summary CSV: `experiments/sam_sensitivity/sam_sensitivity_results.csv`
- Expected unique seeds: `10`
- Required columns: `rho, seed, final_loss, final_train_loss, final_test_accuracy, epochs_trained`
- Required visualization artifacts:
  - `visualizations/static/sam_sensitivity/sam_rho_sweep.png`
  - `visualizations/static/sam_sensitivity/sam_all_metrics.png`
  - `visualizations/static/sam_sensitivity/sam_train_loss_by_epoch.png`
  - `visualizations/static/sam_sensitivity/sam_test_accuracy_by_epoch.png`
