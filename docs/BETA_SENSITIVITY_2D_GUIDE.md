# Beta Sensitivity 2D Visualization - Quick Start Guide

## Purpose
Visualize the impact of β (Momentum) and β1/β2 (Adam) hyperparameters on 2D optimization trajectories.

**Research Alignment:**
- Vietnamese Proposal: *"khảo sát hệ thống và trực quan hóa ảnh hưởng của các siêu tham số đặc trưng (β, β1, β2) lên các khía cạnh động học như quỹ đạo, tốc độ tức thời và độ ổn định"*
- English: Systematically investigate and visualize the influence of characteristic hyperparameters (β, β1, β2) on dynamics aspects such as trajectory, instantaneous speed, and stability

## Usage Examples

### Momentum β Sweep on Rosenbrock
```bash
python src/experiments/beta_sensitivity_2d.py \
  --optimizer Momentum \
  --function rosenbrock \
  --beta-values 0.5,0.7,0.9,0.95,0.99
```

**Output:**
- `results/beta_sensitivity_2d/rosenbrock/momentum/momentum_trajectories.png` - 2D trajectory visualization
- `results/beta_sensitivity_2d/rosenbrock/momentum/momentum_metrics.png` - Metrics vs β
- `results/beta_sensitivity_2d/rosenbrock/momentum/momentum_beta_sweep.csv` - Numerical results

### Adam β1×β2 Sweep on Saddle Point
```bash
python src/experiments/beta_sensitivity_2d.py \
  --optimizer Adam \
  --function saddle_point \
  --beta1-values 0.8,0.9,0.95 \
  --beta2-values 0.9,0.99,0.999
```

**Output:**
- `results/beta_sensitivity_2d/saddle_point/adam/adam_heatmaps.png` - β1 vs β2 heatmaps
- `results/beta_sensitivity_2d/saddle_point/adam/adam_beta_sweep.csv` - Full results table

## Available Test Functions

| Function | Best For | Characteristics |
|----------|----------|-----------------|
| `rosenbrock` | Momentum analysis | Narrow valley, ill-conditioned |
| `saddle_point` | Escape dynamics | Mixed curvature (λ=[1,-1]) |
| `ill_conditioned_quadratic` | Convergence speed | High condition number κ=100 |
| `ackley2d` | Multi-modal | Many local minima |

## Metrics Tracked

1. **Final Loss** - Optimization quality
2. **Iterations** - Convergence speed
3. **Mean Speed** - Average step magnitude ||x_{t+1} - x_t||
4. **Smoothness** - Consistency of loss reduction
5. **Oscillation** - Directional changes (turning angle)
6. **Final Gradient Norm** - Proximity to critical point

## Thesis Integration

**For Trajectory Figures (Chapter 3):**
Use Momentum trajectory plots showing β impact on path smoothness.

**For Dynamics Analysis (Chapter 4):**
Use metrics plots showing β influence on convergence rate and stability.

**For Adam Comparison (Chapter 5):**
Use β1×β2 heatmaps showing joint impact on various metrics.

## Tips

- **Start with defaults:** The script uses well-tuned default learning rates per function
- **Custom LR:** Use `--lr 0.05` to override
- **More iterations:** Use `--max-iters 1000` for slower convergence
- **Quick test:** Try `rosenbrock` first (clearest visualization)
